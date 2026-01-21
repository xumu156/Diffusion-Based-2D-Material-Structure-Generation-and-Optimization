# project/utils/geo_utils.py
import math
from typing import Optional, Tuple

import torch


def lattice_from_params(a, b, c, alpha, beta, gamma):
    alpha_r = math.radians(alpha)
    beta_r = math.radians(beta)
    gamma_r = math.radians(gamma)

    va = torch.tensor([a, 0.0, 0.0], dtype=torch.float32)
    vb = torch.tensor([b * math.cos(gamma_r), b * math.sin(gamma_r), 0.0], dtype=torch.float32)

    cx = c * math.cos(beta_r)
    cy = c * (math.cos(alpha_r) - math.cos(beta_r) * math.cos(gamma_r)) / max(math.sin(gamma_r), 1e-8)
    cz_sq = c * c - cx * cx - cy * cy
    cz = math.sqrt(max(cz_sq, 1e-8))
    vc = torch.tensor([cx, cy, cz], dtype=torch.float32)

    lat = torch.stack([va, vb, vc], dim=0)
    return lat.numpy()


def pbc_diff(frac_i: torch.Tensor, frac_j: torch.Tensor) -> torch.Tensor:
    """
    minimal image in fractional space
    """
    d = frac_i - frac_j
    d = d - torch.round(d)
    return d


def build_fully_connected_edge_index(N: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    directed fully-connected edges excluding self-loops
    edge: src=j -> dst=i
    returns edge_index: (2,E)
    """
    idx = torch.arange(N, device=device)
    src = idx.repeat_interleave(N)  # [0,0,0,...,1,1,1,...]
    dst = idx.repeat(N)             # [0,1,2,...,0,1,2,...]
    m = src != dst
    src = src[m]
    dst = dst[m]
    edge_index = torch.stack([src, dst], dim=0).long()
    return edge_index


def compute_edge_distances_and_mask(
    frac: torch.Tensor,          # (B,N,3)
    lattice: torch.Tensor,       # (B,3,3)
    edge_index: torch.Tensor,    # (2,E) with node index in [0,N-1]
    mask: torch.Tensor,          # (B,N) bool
    cutoff: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      dist: (B,E) float
      edge_mask: (B,E) float in {0,1}  (both nodes valid AND dist<cutoff)
    """
    B, N, _ = frac.shape
    src, dst = edge_index[0], edge_index[1]  # (E,)

    x_src = frac[:, src, :]  # (B,E,3)
    x_dst = frac[:, dst, :]  # (B,E,3)

    d_frac = pbc_diff(x_dst, x_src)  # (B,E,3)
    # cart: (B,E,3) = (B,E,3) @ (B,3,3)
    d_cart = torch.einsum("bed,bdc->bec", d_frac, lattice)
    dist = torch.linalg.norm(d_cart, dim=-1)  # (B,E)

    valid_nodes = (mask[:, src] & mask[:, dst])  # (B,E)
    within = dist < cutoff
    edge_valid = valid_nodes & within

    edge_mask = edge_valid.float()
    # dist for invalid edges can be anything; keep it bounded to avoid weird embedding
    dist = torch.where(edge_valid, dist, torch.full_like(dist, float(cutoff)))
    return dist, edge_mask


def enforce_2d_constraints(frac: torch.Tensor, z_plane_center: float = 0.5, z_span: float = 0.02) -> torch.Tensor:
    out = frac.clone()
    z = out[..., 2]
    z = (z - 0.5)
    z = torch.tanh(z) * (z_span / 2.0)
    out[..., 2] = z_plane_center + z
    return out % 1.0


def plane_thickness_penalty(frac: torch.Tensor, mask: torch.Tensor, thickness_max: float = 0.06) -> torch.Tensor:
    z = frac[..., 2]
    z = torch.where(mask, z, torch.zeros_like(z))
    z_max = z.max(dim=1).values
    z_min = z.min(dim=1).values
    thick = (z_max - z_min)
    return torch.relu(thick - thickness_max).mean()


def min_distance_penalty(
    frac: torch.Tensor, lattice: Optional[torch.Tensor], mask: torch.Tensor, min_dist: float = 1.6
) -> torch.Tensor:
    device = frac.device
    B, N, _ = frac.shape
    if lattice is None:
        lattice = torch.eye(3, device=device)[None, :, :].expand(B, -1, -1)

    fi = frac[:, :, None, :]
    fj = frac[:, None, :, :]
    d_frac = pbc_diff(fi, fj)
    d_cart = torch.matmul(d_frac, lattice[:, None, None, :, :])
    dist = torch.norm(d_cart, dim=-1)

    valid = mask[:, :, None] & mask[:, None, :]
    dist = torch.where(valid, dist, torch.full_like(dist, 1e9))

    diag = torch.eye(N, device=device, dtype=torch.bool)[None, :, :]
    dist = torch.where(diag, torch.full_like(dist, 1e9), dist)

    pen = torch.relu(min_dist - dist)
    return pen.mean()


def coord_smoothness_penalty(frac: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = torch.where(mask[:, :, None], frac, torch.zeros_like(frac))
    mean = x.sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    var = ((x - mean[:, None, :]) ** 2).mean()
    return var


def simple_stability_metrics(frac: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    thick_pen = plane_thickness_penalty(frac, mask, thickness_max=0.06)
    smooth_pen = coord_smoothness_penalty(frac, mask)
    score = 1.0 - (thick_pen + 0.2 * smooth_pen)
    return score


def wrap_frac(frac: torch.Tensor) -> torch.Tensor:
    return frac % 1.0
