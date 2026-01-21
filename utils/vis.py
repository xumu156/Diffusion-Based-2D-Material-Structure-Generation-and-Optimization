# project/utils/vis.py
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def ensure_dir(path: str):
    if path is None or path == "":
        return
    os.makedirs(path, exist_ok=True)


def plot_loss_curve(losses: Dict[str, List[float]], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    plt.figure()
    for k, v in losses.items():
        plt.plot(v, label=k)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_deltaG_hist(deltaG: np.ndarray, out_path: str, bins: int = 40):
    ensure_dir(os.path.dirname(out_path))
    plt.figure()
    plt.hist(deltaG, bins=bins)
    plt.xlabel("ΔG_H (eV)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_stability_synth(stab: np.ndarray, synth: np.ndarray, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    plt.figure()
    plt.scatter(stab, synth, s=10)
    plt.xlabel("Thermo stability (proxy / predicted)")
    plt.ylabel("Synth score (predicted)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _supercell_xy_3x3(xy01: np.ndarray) -> np.ndarray:
    """
    Build 3x3 supercell projection for 2D fractional xy points in [0,1).
    Input:
      xy01: (M,2) in [0,1] (or close)
    Output:
      xy_sc: (9*M,2) shifted by dx,dy in {-1,0,1}
      Range roughly [-1,2] if original is [0,1]
    """
    shifts = []
    for dx in (-1.0, 0.0, 1.0):
        for dy in (-1.0, 0.0, 1.0):
            shifts.append([dx, dy])
    shifts = np.array(shifts, dtype=np.float32)  # (9,2)

    # (9,1,2) + (1,M,2) -> (9,M,2) -> (9*M,2)
    xy_sc = shifts[:, None, :] + xy01[None, :, :]
    xy_sc = xy_sc.reshape(-1, 2)
    return xy_sc


def plot_generated_structures(
    frac: torch.Tensor,
    mask: torch.Tensor,
    out_path: str,
    max_show: int = 16,
):
    """
    3×3 supercell projection scatter of x-y (ignore z)
    frac: (B,N,3) fractional coords
    mask: (B,N) boolean/0-1 mask
    """
    ensure_dir(os.path.dirname(out_path))

    if not isinstance(frac, torch.Tensor):
        raise TypeError("frac must be a torch.Tensor")
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if frac.ndim != 3 or frac.shape[-1] != 3:
        raise ValueError(f"frac must have shape (B,N,3), got {tuple(frac.shape)}")
    if mask.ndim != 2 or mask.shape[0] != frac.shape[0] or mask.shape[1] != frac.shape[1]:
        raise ValueError(f"mask must have shape (B,N), got {tuple(mask.shape)} vs frac {tuple(frac.shape)}")

    B = frac.shape[0]
    show = min(B, max_show)
    cols = int(np.ceil(np.sqrt(show)))
    rows = int(np.ceil(show / cols))

    # 每个子图略大一点，3×3 更容易看清
    plt.figure(figsize=(3.6 * cols, 3.6 * rows))

    for i in range(show):
        ax = plt.subplot(rows, cols, i + 1)

        m = mask[i].detach().cpu().numpy().astype(bool)
        xy = frac[i, m, :2].detach().cpu().numpy()

        # 容错：有些点可能略超出 [0,1]，做一下 wrap
        # 注意：这仍然是“分数坐标”的周期意义
        xy = xy - np.floor(xy)

        if xy.shape[0] == 0:
            ax.set_title(f"sample {i} (empty)")
            ax.set_xlim(-1, 2)
            ax.set_ylim(-1, 2)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)
            continue

        xy_sc = _supercell_xy_3x3(xy)

        # 画 3×3 超胞
        ax.scatter(xy_sc[:, 0], xy_sc[:, 1], s=10)

        # 画出单胞边界（0~1）以及周围胞的网格线（可选）
        # 单胞边界
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], linewidth=0.8)

        # 3×3 范围
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"sample {i}")

        # 让周期性更直观：画整数网格线
        ax.set_xticks([-1, 0, 1, 2])
        ax.set_yticks([-1, 0, 1, 2])
        ax.grid(True, linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
