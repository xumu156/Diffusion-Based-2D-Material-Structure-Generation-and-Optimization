# project/models/structure_generator.py
from typing import Optional

import torch

from utils.geo_utils import enforce_2d_constraints, compute_edge_distances_and_mask


class DiffusionSchedule:
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2, device: str = "cpu"):
        self.T = T
        self.device = device
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)

    def to(self, device: str):
        self.__init__(self.T, float(self.betas[0].item()), float(self.betas[-1].item()), device=device)
        return self


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule, noise: Optional[torch.Tensor] = None):
    if noise is None:
        noise = torch.randn_like(x0)
    B = x0.shape[0]
    sqrt_ab = schedule.sqrt_alphas_bar[t].view(B, 1, 1)
    sqrt_omab = schedule.sqrt_one_minus_alphas_bar[t].view(B, 1, 1)
    return sqrt_ab * x0 + sqrt_omab * noise, noise


@torch.no_grad()
def p_sample_ddpm(
    denoiser,
    x_t: torch.Tensor,
    z: torch.Tensor,
    lattice: torch.Tensor,
    edge_index: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    schedule: DiffusionSchedule,
    cutoff: float = 5.0,
    guidance: Optional[object] = None,
    guidance_scale: float = 1.0,
    enforce_2d: bool = True,
):
    B = x_t.shape[0]
    t_int = t

    beta_t = schedule.betas[t_int].view(B, 1, 1)
    alpha_t = schedule.alphas[t_int].view(B, 1, 1)
    a_bar_t = schedule.alphas_bar[t_int].view(B, 1, 1)

    dist, edge_mask = compute_edge_distances_and_mask(x_t, lattice, edge_index, mask, cutoff=cutoff)

    eps = denoiser(
        z=z, x_t=x_t, lattice=lattice,
        edge_index=edge_index, dist=dist, t=t_int, y=y, mask=mask, edge_mask=edge_mask
    )

    if guidance is not None and guidance_scale > 0.0:
        # p_sample_ddpm 在 @torch.no_grad() 下运行，需要在这里显式开启梯度
        with torch.enable_grad():
            grad = guidance.grad(
                x_t=x_t, z=z, lattice=lattice,
                edge_index=edge_index, dist=dist, edge_mask=edge_mask, y=y, mask=mask
            )
        eps = eps - guidance_scale * grad

    coef = beta_t / torch.sqrt(1.0 - a_bar_t)
    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * eps)

    if (t_int == 0).all():
        x_prev = mean
    else:
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta_t)
        x_prev = mean + sigma * noise

    x_prev = x_prev % 1.0
    if enforce_2d:
        x_prev = enforce_2d_constraints(x_prev, z_plane_center=0.5, z_span=0.02)
    return x_prev


@torch.no_grad()
def sample_structures(
    denoiser,
    schedule: DiffusionSchedule,
    z: torch.Tensor,
    lattice: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    steps: Optional[int] = None,
    cutoff: float = 5.0,
    guidance: Optional[object] = None,
    guidance_scale: float = 1.0,
    enforce_2d: bool = True,
):
    device = z.device
    B, N = z.shape
    T = schedule.T if steps is None else steps

    x_t = torch.rand((B, N, 3), device=device)
    x_t = enforce_2d_constraints(x_t, z_plane_center=0.5, z_span=0.2) if enforce_2d else x_t

    for t in reversed(range(T)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        x_t = p_sample_ddpm(
            denoiser=denoiser,
            x_t=x_t,
            z=z,
            lattice=lattice,
            edge_index=edge_index,
            t=t_tensor,
            y=y,
            mask=mask,
            schedule=schedule,
            cutoff=cutoff,
            guidance=guidance,
            guidance_scale=guidance_scale,
            enforce_2d=enforce_2d,
        )
    return x_t
