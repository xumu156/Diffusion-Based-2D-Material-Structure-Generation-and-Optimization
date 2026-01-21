# project/models/diffusion_model.py
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        t = t.float()
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=device) * (-1))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class GNNLayer(nn.Module):
    """
    edge_index: (2, E)  src=j -> dst=i
    edge_mask:  (B, E) float 0/1, invalid edges should be 0
    """
    def __init__(self, node_dim: int, edge_dim: int, cond_dim: int, hidden: int, drop: float = 0.0):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + cond_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden + cond_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        h: torch.Tensor,              # (B, N, D)
        edge_index: torch.Tensor,     # (2, E)
        edge_attr: torch.Tensor,      # (B, E, De)
        cond: torch.Tensor,           # (B, Dc)
        mask: torch.Tensor,           # (B, N) bool
        edge_mask: Optional[torch.Tensor] = None,  # (B,E) float
    ) -> torch.Tensor:
        B, N, D = h.shape
        src, dst = edge_index[0], edge_index[1]  # (E,)

        h_src = h[:, src, :]  # (B, E, D)
        cond_e = cond[:, None, :].expand(B, h_src.shape[1], cond.shape[-1])  # (B,E,Dc)
        m_in = torch.cat([h_src, edge_attr, cond_e], dim=-1)
        msg = self.msg_mlp(m_in)  # (B, E, H)

        if edge_mask is not None:
            msg = msg * edge_mask[:, :, None]

        agg = torch.zeros((B, N, msg.shape[-1]), device=h.device, dtype=h.dtype)
        agg.scatter_add_(1, dst[None, :, None].expand(B, dst.shape[0], msg.shape[-1]), msg)

        cond_n = cond[:, None, :].expand(B, N, cond.shape[-1])
        u_in = torch.cat([h, agg, cond_n], dim=-1)
        dh = self.upd_mlp(u_in)
        out = self.norm(h + dh)

        out = torch.where(mask[:, :, None], out, h)
        return out


@dataclass
class DiffusionConfig:
    node_dim: int = 192
    edge_dim: int = 64
    time_dim: int = 128
    cond_dim: int = 128
    gnn_hidden: int = 256
    gnn_layers: int = 6
    dropout: float = 0.0
    enforce_2d: bool = True
    z_scale: float = 0.1


class PropertyConditioner(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.mlp(y)


class EdgeEmbedding(nn.Module):
    def __init__(self, rbf_bins: int = 32, cutoff: float = 5.0, out_dim: int = 64):
        super().__init__()
        self.rbf_bins = rbf_bins
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, rbf_bins)
        self.register_buffer("centers", centers)
        self.gamma = nn.Parameter(torch.tensor(10.0))
        self.proj = nn.Sequential(
            nn.Linear(rbf_bins + 1, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        d = dist.clamp(min=0.0, max=self.cutoff)
        rbf = torch.exp(-self.gamma * (d[:, :, None] - self.centers[None, None, :]) ** 2)
        d_norm = (d / self.cutoff)[:, :, None]
        feat = torch.cat([rbf, d_norm], dim=-1)
        return self.proj(feat)


class AtomEmbedding(nn.Module):
    def __init__(self, max_z: int = 100, emb_dim: int = 192):
        super().__init__()
        self.emb = nn.Embedding(max_z + 1, emb_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.emb(z.clamp(min=0, max=self.emb.num_embeddings - 1))


class DiffusionDenoiser(nn.Module):
    def __init__(self, cfg: DiffusionConfig, y_dim: int = 3, max_z: int = 100):
        super().__init__()
        self.cfg = cfg
        self.time_emb = SinusoidalTimeEmbedding(cfg.time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_dim, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )
        self.prop_cond = PropertyConditioner(y_dim, cfg.cond_dim, hidden=256)

        self.atom_emb = AtomEmbedding(max_z=max_z, emb_dim=cfg.node_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, cfg.node_dim),
            nn.SiLU(),
            nn.Linear(cfg.node_dim, cfg.node_dim),
        )

        self.edge_emb = EdgeEmbedding(rbf_bins=32, cutoff=5.0, out_dim=cfg.edge_dim)

        self.gnn = nn.ModuleList([
            GNNLayer(cfg.node_dim, cfg.edge_dim, cfg.cond_dim, cfg.gnn_hidden, cfg.dropout)
            for _ in range(cfg.gnn_layers)
        ])

        self.out_mlp = nn.Sequential(
            nn.Linear(cfg.node_dim + cfg.cond_dim, cfg.gnn_hidden),
            nn.SiLU(),
            nn.Linear(cfg.gnn_hidden, cfg.gnn_hidden),
            nn.SiLU(),
            nn.Linear(cfg.gnn_hidden, 3),
        )

    def forward(
        self,
        z: torch.Tensor,
        x_t: torch.Tensor,
        lattice: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        dist: torch.Tensor,              # (B,E)
        t: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,  # (B,E)
    ) -> torch.Tensor:
        B, N, _ = x_t.shape

        if self.cfg.enforce_2d:
            x_in = x_t.clone()
            x_in[..., 2] = x_in[..., 2] * self.cfg.z_scale
        else:
            x_in = x_t

        h = self.atom_emb(z) + self.coord_mlp(x_in)
        e = self.edge_emb(dist)  # (B,E,De)

        t_emb = self.time_mlp(self.time_emb(t))
        y_emb = self.prop_cond(y)
        cond = t_emb + y_emb

        for layer in self.gnn:
            h = layer(h, edge_index=edge_index, edge_attr=e, cond=cond, mask=mask, edge_mask=edge_mask)

        cond_n = cond[:, None, :].expand(B, N, cond.shape[-1])
        out = self.out_mlp(torch.cat([h, cond_n], dim=-1))
        out = torch.where(mask[:, :, None], out, torch.zeros_like(out))
        return out


class PropertyPredictor(nn.Module):
    def __init__(self, node_dim: int = 192, edge_dim: int = 64, cond_dim: int = 128, hidden: int = 256,
                 layers: int = 4, max_z: int = 100):
        super().__init__()
        self.atom_emb = AtomEmbedding(max_z=max_z, emb_dim=node_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.edge_emb = EdgeEmbedding(rbf_bins=32, cutoff=5.0, out_dim=edge_dim)

        self.null_cond = nn.Parameter(torch.zeros(cond_dim))

        self.gnn = nn.ModuleList([
            GNNLayer(node_dim, edge_dim, cond_dim, hidden, drop=0.0)
            for _ in range(layers)
        ])

        self.pool_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden, 3)

    def forward(self, z, frac_coords, edge_index, dist, mask, edge_mask: Optional[torch.Tensor] = None):
        B, N, _ = frac_coords.shape
        h = self.atom_emb(z) + self.coord_mlp(frac_coords)
        e = self.edge_emb(dist)
        cond = self.null_cond[None, :].expand(B, -1)

        for layer in self.gnn:
            h = layer(h, edge_index=edge_index, edge_attr=e, cond=cond, mask=mask, edge_mask=edge_mask)

        m = mask.float()
        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (h * m[:, :, None]).sum(dim=1) / denom
        feat = self.pool_mlp(pooled)
        out = self.head(feat)

        deltaG = out[:, 0:1]
        thermo = out[:, 1:2]
        synth = torch.sigmoid(out[:, 2:3])
        return torch.cat([deltaG, thermo, synth], dim=-1)
