# project/dataset/material_dataset.py

import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.geo_utils import lattice_from_params, build_fully_connected_edge_index


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class MaterialDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        max_atoms: int = 64,
        cutoff: float = 5.0,
        default_lattice_params=(3.0, 3.0, 20.0, 90.0, 90.0, 120.0),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.cutoff = cutoff
        self.max_atoms = max_atoms
        self.files = sorted([
            os.path.join(data_dir, x)
            for x in os.listdir(data_dir)
            if x.startswith("JVASP-") and x.endswith(".json")
        ])

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .json found under {data_dir}")

        self.default_lattice = lattice_from_params(*default_lattice_params)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = _read_json(self.files[idx])

        z = np.array(item["atomic_numbers"], dtype=np.int64)
        x = np.array(item["frac_coords"], dtype=np.float32)

        if "lattice" in item and item["lattice"] is not None:
            lattice = np.array(item["lattice"], dtype=np.float32)
        else:
            lattice = self.default_lattice.astype(np.float32)

        props = item.get("properties", {})
        deltaG = float(props.get("deltaG_H", 0.0))
        thermo = float(props.get("thermo_stability", 0.0))
        synth = float(props.get("synth_score", 0.5))

        n = min(len(z), self.max_atoms)
        z = z[:n]
        x = x[:n]

        sample = {
            "id": item.get("id", os.path.basename(self.files[idx]).replace(".json", "")),
            "z": torch.from_numpy(z).long(),                 # (n,)
            "x": torch.from_numpy(x).float(),                # (n,3)
            "lattice": torch.from_numpy(lattice).float(),    # (3,3)
            "y": torch.tensor([deltaG, thermo, synth], dtype=torch.float32),  # (3,)
        }
        return sample


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)
    n_list = [b["z"].shape[0] for b in batch]
    Nmax = max(n_list)

    z_pad = torch.zeros(B, Nmax, dtype=torch.long)
    x_pad = torch.zeros(B, Nmax, 3, dtype=torch.float32)
    mask = torch.zeros(B, Nmax, dtype=torch.bool)

    lattice = torch.stack([b["lattice"] for b in batch], dim=0)  # (B,3,3)
    y = torch.stack([b["y"] for b in batch], dim=0)              # (B,3)

    for i, b in enumerate(batch):
        n = b["z"].shape[0]
        z_pad[i, :n] = b["z"]
        x_pad[i, :n] = b["x"]
        mask[i, :n] = True

    # 每个 batch 使用同一个固定全连接图（Nmax）
    edge_index = build_fully_connected_edge_index(Nmax, device=None)  # CPU tensor

    ids = [b["id"] for b in batch]

    return {
        "ids": ids,
        "z": z_pad,
        "x": x_pad,
        "lattice": lattice,
        "y": y,
        "mask": mask,
        "edge_index": edge_index,
        "Nmax": Nmax,
    }
