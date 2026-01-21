# project/train.py
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.material_dataset import MaterialDataset, collate_batch
from models.diffusion_model import DiffusionConfig, DiffusionDenoiser, PropertyPredictor
from models.structure_generator import DiffusionSchedule, q_sample
from utils.geo_utils import compute_edge_distances_and_mask
from utils.vis import plot_loss_curve


def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(path: str, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(kwargs, path)


def train_property_predictor(
    predictor: PropertyPredictor,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    out_dir: str,
    cutoff: float,
):
    predictor.train()
    opt = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=1e-4)

    losses = {"prop_total": [], "deltaG": [], "thermo": [], "synth": []}
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"[Property] epoch {ep+1}/{epochs}")
        for batch in pbar:
            z = batch["z"].to(device)
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            lattice = batch["lattice"].to(device)
            edge_index = batch["edge_index"].to(device)

            dist, edge_mask = compute_edge_distances_and_mask(x, lattice, edge_index, mask, cutoff=cutoff)

            pred = predictor(z=z, frac_coords=x, edge_index=edge_index, dist=dist, mask=mask, edge_mask=edge_mask)

            l_deltaG = F.mse_loss(pred[:, 0], y[:, 0])
            l_thermo = F.mse_loss(pred[:, 1], y[:, 1])
            l_synth = F.binary_cross_entropy(pred[:, 2].clamp(1e-6, 1 - 1e-6), y[:, 2].clamp(0, 1))
            loss = l_deltaG + 0.5 * l_thermo + 0.5 * l_synth

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            opt.step()

            losses["prop_total"].append(float(loss.item()))
            losses["deltaG"].append(float(l_deltaG.item()))
            losses["thermo"].append(float(l_thermo.item()))
            losses["synth"].append(float(l_synth.item()))
            pbar.set_postfix(loss=float(loss.item()))

        save_ckpt(os.path.join(out_dir, "ckpt_property.pt"), model=predictor.state_dict(), epoch=ep)

    plot_loss_curve(losses, os.path.join(out_dir, "loss_curve_property.png"))
    return losses


def train_diffusion(
    denoiser: DiffusionDenoiser,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    out_dir: str,
    T: int,
    cutoff: float,
):
    denoiser.train()
    opt = torch.optim.AdamW(denoiser.parameters(), lr=lr, weight_decay=1e-4)
    schedule = DiffusionSchedule(T=T, device=device)

    losses = {"diff_total": [], "mse_eps": []}
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"[Diffusion] epoch {ep+1}/{epochs}")
        for batch in pbar:
            z = batch["z"].to(device)
            x0 = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            lattice = batch["lattice"].to(device)
            edge_index = batch["edge_index"].to(device)

            B = z.shape[0]
            t = torch.randint(0, T, (B,), device=device).long()

            x_t, eps = q_sample(x0=x0, t=t, schedule=schedule, noise=None)

            dist, edge_mask = compute_edge_distances_and_mask(x_t, lattice, edge_index, mask, cutoff=cutoff)

            eps_pred = denoiser(
                z=z, x_t=x_t, lattice=lattice,
                edge_index=edge_index, dist=dist, t=t, y=y, mask=mask, edge_mask=edge_mask
            )

            mse = F.mse_loss(eps_pred, eps, reduction="none")
            mse = (mse * mask[:, :, None].float()).sum() / mask.float().sum().clamp(min=1.0)
            loss = mse

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()

            losses["diff_total"].append(float(loss.item()))
            losses["mse_eps"].append(float(mse.item()))
            pbar.set_postfix(loss=float(loss.item()))

        save_ckpt(os.path.join(out_dir, "ckpt_diffusion.pt"), model=denoiser.state_dict(), epoch=ep)

    plot_loss_curve(losses, os.path.join(out_dir, "loss_curve_diffusion.png"))
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_atoms", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs_prop", type=int, default=10)
    parser.add_argument("--epochs_diff", type=int, default=20)
    parser.add_argument("--lr_prop", type=float, default=2e-4)
    parser.add_argument("--lr_diff", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=1000)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = MaterialDataset(data_dir=args.data_dir, max_atoms=args.max_atoms, cutoff=args.cutoff)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)

    predictor = PropertyPredictor(node_dim=192, edge_dim=64, cond_dim=128, hidden=256, layers=4, max_z=100).to(args.device)
    train_property_predictor(predictor, dl, args.device, args.epochs_prop, args.lr_prop, args.out_dir, cutoff=args.cutoff)

    cfg = DiffusionConfig(node_dim=192, edge_dim=64, time_dim=128, cond_dim=128, gnn_hidden=256, gnn_layers=6, dropout=0.0)
    denoiser = DiffusionDenoiser(cfg=cfg, y_dim=3, max_z=100).to(args.device)
    train_diffusion(denoiser, dl, args.device, args.epochs_diff, args.lr_diff, args.out_dir, T=args.T, cutoff=args.cutoff)

    print("Done. Checkpoints saved in:", args.out_dir)


if __name__ == "__main__":
    main()
