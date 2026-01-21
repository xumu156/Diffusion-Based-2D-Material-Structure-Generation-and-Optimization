
# project/models/optimization.py
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class ObjectiveWeights:
    w_deltaG: float = 1.0
    w_thermo: float = 0.5
    w_synth: float = 0.5
    w_min_dist: float = 0.5
    w_thickness: float = 0.2
    w_smooth: float = 0.1


class GuidedObjective:
    def __init__(
        self,
        predictor,
        weights: ObjectiveWeights = ObjectiveWeights(),
        deltaG_target: float = 0.0,
        thermo_target_min: float = 0.0,
        synth_target_min: float = 0.7,
        min_dist: float = 1.6,
        thickness_max: float = 0.06,
    ):
        self.predictor = predictor
        self.w = weights
        self.deltaG_target = deltaG_target
        self.thermo_target_min = thermo_target_min
        self.synth_target_min = synth_target_min
        self.min_dist = min_dist
        self.thickness_max = thickness_max

    def loss(self, x, z, lattice, edge_index, dist, edge_mask, y, mask):
        pred = self.predictor(z=z, frac_coords=x, edge_index=edge_index, dist=dist, mask=mask, edge_mask=edge_mask)
        deltaG = pred[:, 0]
        thermo = pred[:, 1]
        synth = pred[:, 2]

        l_deltaG = (deltaG - self.deltaG_target).abs()
        l_thermo = F.relu(self.thermo_target_min - thermo)
        l_synth = F.relu(self.synth_target_min - synth)

        # 下面这些几何惩罚仍可用（如果你不想用可删）
        from utils.geo_utils import min_distance_penalty, plane_thickness_penalty, coord_smoothness_penalty
        l_min_dist = min_distance_penalty(x, lattice=lattice, mask=mask, min_dist=self.min_dist)
        l_thick = plane_thickness_penalty(x, mask=mask, thickness_max=self.thickness_max)
        l_smooth = coord_smoothness_penalty(x, mask=mask)

        total = (
            self.w.w_deltaG * l_deltaG.mean()
            + self.w.w_thermo * l_thermo.mean()
            + self.w.w_synth * l_synth.mean()
            + self.w.w_min_dist * l_min_dist
            + self.w.w_thickness * l_thick
            + self.w.w_smooth * l_smooth
        )
        return total


class DiffusionGuidance:
    def __init__(self, objective: GuidedObjective):
        self.obj = objective

    def grad(self, x_t, z, lattice, edge_index, dist, edge_mask, y, mask) -> torch.Tensor:
        x = x_t.detach().clone().requires_grad_(True)
        loss = self.obj.loss(x=x, z=z, lattice=lattice, edge_index=edge_index, dist=dist,
                             edge_mask=edge_mask, y=y, mask=mask)
        g = torch.autograd.grad(loss, x, create_graph=False, retain_graph=False)[0]
        g_norm = torch.norm(g.reshape(g.shape[0], -1), dim=-1).view(-1, 1, 1).clamp(min=1e-6)
        g = g / g_norm
        return g.detach()


@torch.no_grad()
def rank_candidates(pred: torch.Tensor) -> torch.Tensor:
    deltaG = pred[:, 0]
    thermo = pred[:, 1]
    synth = pred[:, 2]
    score = -deltaG.abs() + thermo + synth
    return score
