import os
import json
import math
import argparse
import numpy as np


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def pbc_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    d = d - np.round(d)
    return d


def pairwise_min_dist(frac: np.ndarray, lattice: np.ndarray) -> float:
    """
    frac: (N,3) fractional
    lattice: (3,3)
    returns minimal interatomic distance in Angstrom (excluding i=j)
    """
    N = frac.shape[0]
    if N <= 1:
        return 1e9
    min_d = 1e9
    for i in range(N):
        for j in range(i + 1, N):
            df = pbc_diff(frac[i], frac[j])
            dc = df @ lattice
            d = float(np.linalg.norm(dc))
            if d < min_d:
                min_d = d
    return min_d


def thickness(frac: np.ndarray) -> float:
    z = frac[:, 2]
    return float(z.max() - z.min())


def spread_xy(frac: np.ndarray) -> float:
    """
    分散度：xy 方差的均值（越分散越大）
    """
    xy = frac[:, :2]
    v = float(np.mean(np.var(xy, axis=0)))
    return v


def elem_proxy(z: np.ndarray) -> float:
    """
    粗糙“HER 倾向”proxy：让 deltaG_H 有分布。
    思路：不同元素族给不同偏置（只为产生可学习信号，不代表真实ΔG_H）
    """
    z = z.astype(int).tolist()
    if len(z) == 0:
        return 0.0
    # 简单分组：过渡金属/非金属/卤素/氧氮等
    tm = set(list(range(21, 31)) + list(range(39, 49)) + list(range(57, 81)) + list(range(89, 113)))
    hal = {9, 17, 35, 53, 85}
    chalc = {8, 16, 34, 52, 84}
    pnict = {7, 15, 33, 51, 83}

    s = 0.0
    for a in z:
        if a in tm:
            s += -0.15
        if a in hal:
            s += 0.10
        if a in chalc:
            s += -0.05
        if a in pnict:
            s += -0.03
    s /= len(z)
    return float(s)


def compute_proxies(z: np.ndarray, frac: np.ndarray, lattice: np.ndarray) -> dict:
    """
    输出:
      thermo_stability: 连续值（越大越稳定）
      synth_score: 0~1（越大越可合成）
      deltaG_H: 有分布的启发式 proxy（越接近0越好）
    """
    N = len(z)
    if N == 0:
        return {"deltaG_H": 0.0, "thermo_stability": 0.0, "synth_score": 0.5}

    dmin = pairwise_min_dist(frac, lattice)               # Å
    thick = thickness(frac)                               # frac units (0~1)
    spread = spread_xy(frac)                              # ~0~0.1
    elem = elem_proxy(z)

    # 2D 厚度：希望小
    thick_pen = max(0.0, thick - 0.06) / 0.06             # >0 惩罚
    # 最小距离：希望大于 1.6 Å
    d_pen = max(0.0, 1.6 - dmin) / 1.6                    # >0 惩罚
    # 分散度：过小表示挤在一起/塌缩，过大表示太散；希望在中间（0.03左右）
    spread_target = 0.03
    spread_pen = abs(spread - spread_target) / spread_target

    # thermo：把惩罚组合成一个“越大越好”的分数
    thermo = 1.0 - (1.5 * d_pen + 1.0 * thick_pen + 0.3 * spread_pen)
    # 防止数值爆炸
    thermo = float(np.clip(thermo, -2.0, 2.0))

    # synth：映射到 0~1，thermo 越大越可合成
    synth = sigmoid(2.0 * thermo)

    # deltaG_H proxy：由元素偏置 + 稳定性/几何产生小幅变化（单位 eV）
    # 目标是接近 0，所以我们生成一个围绕 0 的分布
    deltaG = 0.10 * elem + 0.02 * (0.5 - thick) + 0.02 * (dmin - 2.0)
    deltaG = float(np.clip(deltaG, -0.6, 0.6))

    return {"deltaG_H": deltaG, "thermo_stability": thermo, "synth_score": float(synth)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="JVASP-", help="only rewrite files starting with this")
    ap.add_argument("--backup", action="store_true", help="save old file as *.bak.json")
    args = ap.parse_args()

    data_dir = args.data_dir
    files = [x for x in os.listdir(data_dir) if x.startswith(args.pattern) and x.endswith(".json")]
    files.sort()

    if not files:
        raise RuntimeError(f"No files found under {data_dir} with pattern {args.pattern}*.json")

    changed = 0
    for fn in files:
        path = os.path.join(data_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        if "atomic_numbers" not in d or "frac_coords" not in d:
            continue

        z = np.array(d["atomic_numbers"], dtype=np.int64)
        frac = np.array(d["frac_coords"], dtype=np.float32)

        lattice = d.get("lattice", None)
        if lattice is None:
            # 默认一个典型 2D 晶格（仅用于 proxy）
            lattice = np.array([[3.0, 0.0, 0.0],
                                [1.5, 2.6, 0.0],
                                [0.0, 0.0, 20.0]], dtype=np.float32)
        else:
            lattice = np.array(lattice, dtype=np.float32)

        props = compute_proxies(z, frac, lattice)
        d["properties"] = props

        if args.backup:
            bak = path.replace(".json", ".bak.json")
            if not os.path.exists(bak):
                with open(bak, "w", encoding="utf-8") as f:
                    json.dump(d, f, ensure_ascii=False, indent=2)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

        changed += 1

    print(f"[DONE] Rewrote properties for {changed} files in {data_dir}")


if __name__ == "__main__":
    main()
