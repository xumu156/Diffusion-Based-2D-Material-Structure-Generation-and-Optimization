# project/test.py
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.material_dataset import MaterialDataset, collate_batch
from models.diffusion_model import DiffusionConfig, DiffusionDenoiser, PropertyPredictor
from models.structure_generator import DiffusionSchedule, sample_structures
from models.optimization import ObjectiveWeights, GuidedObjective, DiffusionGuidance, rank_candidates
from utils.geo_utils import compute_edge_distances_and_mask, enforce_2d_constraints
from utils.vis import plot_deltaG_hist, plot_generated_structures, plot_stability_synth


def load_ckpt(path: str, device: str):
    return torch.load(path, map_location=device)


@torch.no_grad()
def evaluate_and_save(
    predictor: PropertyPredictor,
    z: torch.Tensor,
    x: torch.Tensor,
    lattice: torch.Tensor,
    edge_index: torch.Tensor,
    mask: torch.Tensor,
    out_dir: str,
    cutoff: float,
):
    os.makedirs(out_dir, exist_ok=True)

    dist, edge_mask = compute_edge_distances_and_mask(x, lattice, edge_index, mask, cutoff=cutoff)
    pred = predictor(z=z, frac_coords=x, edge_index=edge_index, dist=dist, mask=mask, edge_mask=edge_mask)
    pred_np = pred.cpu().numpy()

    deltaG = pred_np[:, 0]
    thermo = pred_np[:, 1]
    synth = pred_np[:, 2]

    plot_deltaG_hist(deltaG, os.path.join(out_dir, "her_performance.png"))
    plot_stability_synth(thermo, synth, os.path.join(out_dir, "stability_curve.png"))
    plot_generated_structures(x, mask, os.path.join(out_dir, "generated_structures.png"))

    score = rank_candidates(pred).cpu().numpy()
    idx = np.argsort(-score)
    topk = idx[: min(50, len(idx))]

    records = []
    for i in topk:
        m = mask[i].cpu().numpy().astype(bool)
        rec = {
            "rank": int(np.where(topk == i)[0][0] + 1),
            "score": float(score[i]),
            "pred_deltaG": float(deltaG[i]),
            "pred_thermo": float(thermo[i]),
            "pred_synth": float(synth[i]),
            "atomic_numbers": z[i, m].cpu().tolist(),
            "frac_coords": x[i, m].cpu().tolist(),
            "lattice": lattice[i].cpu().tolist(),
        }
        records.append(rec)

    with open(os.path.join(out_dir, "top_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved plots and top_candidates.json to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results_gen")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_atoms", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=5.0)

    parser.add_argument("--ckpt_property", type=str, default="results/ckpt_property.pt")
    parser.add_argument("--ckpt_diffusion", type=str, default="results/ckpt_diffusion.pt")

    parser.add_argument("--num_generate", type=int, default=128)
    parser.add_argument("--T", type=int, default=200)  # 先用 200 更快，想更好可调回 1000
    parser.add_argument("--guidance_scale", type=float, default=1.5)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    predictor = PropertyPredictor(node_dim=192, edge_dim=64, cond_dim=128, hidden=256, layers=4, max_z=100).to(args.device)
    denoiser = DiffusionDenoiser(cfg=DiffusionConfig(), y_dim=3, max_z=100).to(args.device)

    ckpt_p = load_ckpt(args.ckpt_property, args.device)
    predictor.load_state_dict(ckpt_p["model"])
    predictor.eval()

    ckpt_d = load_ckpt(args.ckpt_diffusion, args.device)
    denoiser.load_state_dict(ckpt_d["model"])
    denoiser.eval()

    ds = MaterialDataset(data_dir=args.data_dir, max_atoms=args.max_atoms, cutoff=args.cutoff)
    dl = DataLoader(ds, batch_size=args.num_generate, shuffle=True, num_workers=0, collate_fn=collate_batch)

    batch = next(iter(dl))
    z = batch["z"].to(args.device)
    lattice = batch["lattice"].to(args.device)
    mask = batch["mask"].to(args.device)
    edge_index = batch["edge_index"].to(args.device)

    B = z.shape[0]
    y_target = torch.tensor([0.0, 1.0, 0.9], device=args.device)[None, :].expand(B, -1)

    obj = GuidedObjective(
        predictor=predictor,
        weights=ObjectiveWeights(w_deltaG=1.0, w_thermo=0.8, w_synth=0.8, w_min_dist=0.8, w_thickness=0.3, w_smooth=0.1),
        deltaG_target=0.0,
        thermo_target_min=0.5,
        synth_target_min=0.8,
        min_dist=1.6,
        thickness_max=0.06,
    )
    guidance = DiffusionGuidance(obj)

    schedule = DiffusionSchedule(T=args.T, device=args.device)

    x_gen = sample_structures(
        denoiser=denoiser,
        schedule=schedule,
        z=z,
        lattice=lattice,
        edge_index=edge_index,
        y=y_target,
        mask=mask,
        steps=args.T,
        cutoff=args.cutoff,
        guidance=guidance,
        guidance_scale=args.guidance_scale,
        enforce_2d=True,
    )
    x_gen = enforce_2d_constraints(x_gen, z_plane_center=0.5, z_span=0.02)

    evaluate_and_save(
        predictor=predictor,
        z=z,
        x=x_gen,
        lattice=lattice,
        edge_index=edge_index,
        mask=mask,
        out_dir=args.out_dir,
        cutoff=args.cutoff,
    )


if __name__ == "__main__":
    main()
