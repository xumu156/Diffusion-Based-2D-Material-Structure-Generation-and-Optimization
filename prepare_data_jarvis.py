import os
import json
import time
import argparse
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def pick_first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def is_valid_zip(path: str) -> bool:
    try:
        return os.path.isfile(path) and zipfile.is_zipfile(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def read_head_as_text(path: str, nbytes: int = 512) -> str:
    try:
        with open(path, "rb") as f:
            b = f.read(nbytes)
        # 尝试按 utf-8/gbk 解码，方便你判断是不是 HTML/错误页
        for enc in ("utf-8", "gbk", "latin1"):
            try:
                return b.decode(enc, errors="replace")
            except Exception:
                pass
        return repr(b)
    except Exception as e:
        return f"<failed to read head: {e}>"


def url_variants(url: str) -> List[str]:
    """
    JARVIS 的 db_info 里通常是：
      https://figshare.com/ndownloader/files/<id>

    但很多网络环境下：
      https://ndownloader.figshare.com/files/<id>
    更稳定。

    这里自动尝试多个等价 URL。
    """
    urls = [url]

    # figshare.com/ndownloader -> ndownloader.figshare.com
    if "https://figshare.com/ndownloader/files/" in url:
        alt = url.replace("https://figshare.com/ndownloader/files/", "https://ndownloader.figshare.com/files/")
        urls.append(alt)

    # 老写法：ndownloader.figshare.com -> figshare.com/ndownloader
    if "https://ndownloader.figshare.com/files/" in url:
        alt = url.replace("https://ndownloader.figshare.com/files/", "https://figshare.com/ndownloader/files/")
        urls.append(alt)

    # 去重保持顺序
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def download_with_retry(
    urls: List[str],
    out_path: str,
    timeout: int = 60,
    retries: int = 3,
    sleep_base: float = 1.5,
) -> Tuple[bool, str]:
    """
    下载到 out_path。若不是 zip，会自动删掉并尝试下一个 URL / 下一次重试。
    返回 (ok, debug_message)
    """
    headers = {
        # Figshare/某些网关对默认 UA 有时会返回 HTML 或 403
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }

    for u in urls:
        for attempt in range(1, retries + 1):
            try:
                # 确保目录存在
                ensure_dir(os.path.dirname(out_path))

                # 若已有旧文件，先删掉（避免“坏缓存”）
                if os.path.isfile(out_path):
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass

                with requests.get(u, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as r:
                    status = r.status_code
                    ctype = r.headers.get("content-type", "")
                    clen = r.headers.get("content-length", "")
                    if status != 200:
                        msg = f"[WARN] GET {u} -> status={status}, content-type={ctype}, content-length={clen}"
                        time.sleep(sleep_base * attempt)
                        continue

                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 64):
                            if chunk:
                                f.write(chunk)

                size = os.path.getsize(out_path) if os.path.isfile(out_path) else 0
                if not is_valid_zip(out_path):
                    head = read_head_as_text(out_path, 512)
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
                    msg = (
                        f"[WARN] Downloaded file is NOT a zip. url={u} size={size} bytes\n"
                        f"       First bytes (decoded):\n{head}\n"
                        f"       This usually means the download was blocked or returned an HTML error page."
                    )
                    time.sleep(sleep_base * attempt)
                    continue

                return True, f"[INFO] Download ok. url={u} size={size} bytes"

            except Exception as e:
                # 清理坏文件
                if os.path.isfile(out_path):
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
                msg = f"[WARN] Download failed. url={u} attempt={attempt}/{retries} err={e}"
                time.sleep(sleep_base * attempt)
                continue

    return False, "[ERROR] All download attempts failed (all URLs / retries)."


def load_json_from_zip(zip_path: str, member_name: str) -> Any:
    with zipfile.ZipFile(zip_path, "r") as zf:
        if member_name not in zf.namelist():
            # 有些 zip 里路径不同，做一次兜底查找
            candidates = [x for x in zf.namelist() if x.endswith(member_name)]
            if not candidates:
                raise KeyError(f"Member '{member_name}' not found in zip. members={zf.namelist()[:10]} ...")
            member_name = candidates[0]
        raw = zf.read(member_name)
    return json.loads(raw)


def map_properties(entry: Dict[str, Any]) -> Dict[str, float]:
    """
    统一映射到你的训练字段：
      - deltaG_H: 结构库通常没有（先置 0 或后续再补标签）
      - thermo_stability: 用 -ehull（越大越稳定）
      - synth_score: 若无标签，用 ehull 的 proxy（越接近 0 越可能可合成）
    """
    deltaG = safe_float(
        pick_first(entry, ["deltaG_H", "delta_g_h", "deltag_h", "deltaGH", "delta_g"], default=0.0),
        default=0.0,
    )

    ehull = pick_first(entry, ["ehull", "e_hull", "energy_above_hull", "e_above_hull"], default=None)
    ehull = safe_float(ehull, default=0.0)
    thermo = -ehull

    synth = pick_first(entry, ["synth_score", "synthesis_score", "is_synthesizable"], default=None)
    if synth is None:
        synth = float(sigmoid((-ehull) / 0.08))
    else:
        synth = safe_float(synth, default=0.5)
        if synth > 1.0:
            synth = float(max(0.0, min(1.0, synth / 100.0)))
    synth = float(max(0.0, min(1.0, synth)))

    return {
        "deltaG_H": float(deltaG),
        "thermo_stability": float(thermo),
        "synth_score": float(synth),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dft_2d_2021",
                        help="JARVIS dataset name, e.g. dft_2d_2021 / dft_2d / twod_matpd / c2db")
    parser.add_argument("--out_dir", type=str, required=True, help="Output dir for *.json samples")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Where to store downloaded zip (recommended). Default: <out_dir>/_jarvis_cache")
    parser.add_argument("--max_atoms", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit; otherwise export first N entries")
    parser.add_argument("--force_redownload", action="store_true", help="Force delete existing zip and re-download")
    parser.add_argument("--offline_zip", type=str, default=None,
                        help="If provided, skip download and load dataset from this local zip file")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    if args.cache_dir is None:
        args.cache_dir = os.path.join(args.out_dir, "_jarvis_cache")
    ensure_dir(args.cache_dir)

    # 读取 JARVIS dataset 的 (url, js_tag, message, reference)
    from jarvis.db.figshare import get_db_info  # noqa
    from jarvis.core.atoms import Atoms  # noqa

    db_info = get_db_info()
    if args.dataset not in db_info:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Available examples: {list(db_info.keys())[:20]} ...")

    url, js_tag, message, reference = db_info[args.dataset][0], db_info[args.dataset][1], db_info[args.dataset][2], db_info[args.dataset][3]
    print(message)
    print("Reference:" + str(reference))

    zip_name = js_tag + ".zip"   # JARVIS 官方就是这样命名缓存 zip 的:contentReference[oaicite:1]{index=1}
    zip_path = os.path.join(args.cache_dir, zip_name)

    if args.offline_zip:
        zip_path = args.offline_zip
        print(f"[INFO] Using offline zip: {zip_path}")
        if not is_valid_zip(zip_path):
            raise RuntimeError(f"offline_zip is not a valid zip: {zip_path}")

    else:
        if args.force_redownload and os.path.isfile(zip_path):
            try:
                os.remove(zip_path)
                print(f"[INFO] Removed cached zip: {zip_path}")
            except Exception as e:
                print(f"[WARN] Failed to remove cached zip: {zip_path}, err={e}")

        if not is_valid_zip(zip_path):
            print(f"[INFO] Downloading zip to: {zip_path}")
            urls = url_variants(url)
            ok, dbg = download_with_retry(urls, zip_path, timeout=90, retries=3)
            print(dbg)
            if not ok:
                raise RuntimeError(
                    "Download failed. If you're in a restricted network, try VPN/proxy or manually download the zip.\n"
                    f"Expected zip filename: {zip_name}\n"
                    f"Put it at: {zip_path}\n"
                    "Then rerun with: --offline_zip <path_to_zip>"
                )

    print("[INFO] Loading dataset JSON from zip ...")
    try:
        data_list = load_json_from_zip(zip_path, js_tag)
    except Exception as e:
        head = read_head_as_text(zip_path, 256)
        raise RuntimeError(
            f"Failed to load json from zip: {zip_path}\n"
            f"js_tag={js_tag}\n"
            f"err={e}\n"
            f"zip head (decoded):\n{head}"
        )

    if not isinstance(data_list, list):
        raise RuntimeError(f"Unexpected dataset type: {type(data_list)} (expected list).")

    print(f"[INFO] Loaded entries: {len(data_list)}")
    n_total = len(data_list)
    n_export = n_total if args.limit <= 0 else min(args.limit, n_total)

    ok_cnt = 0
    skipped = 0

    for i in range(n_export):
        entry = data_list[i]

        mid = pick_first(entry, ["jid", "id", "material_id", "mpid"], default=None)
        if mid is None:
            mid = f"{args.dataset}_{i:06d}"
        else:
            mid = str(mid)

        atoms_dict = entry.get("atoms", None)
        if atoms_dict is None:
            skipped += 1
            continue

        try:
            atoms = Atoms.from_dict(atoms_dict)
        except Exception:
            skipped += 1
            continue

        z = list(atoms.atomic_numbers)
        if len(z) == 0:
            skipped += 1
            continue

        frac = np.array(atoms.frac_coords, dtype=np.float32)
        lattice = np.array(atoms.lattice_mat, dtype=np.float32)

        if len(z) > args.max_atoms:
            z = z[: args.max_atoms]
            frac = frac[: args.max_atoms, :]

        props = map_properties(entry)

        out = {
            "id": mid,
            "atomic_numbers": [int(x) for x in z],
            "frac_coords": frac.tolist(),
            "lattice": lattice.tolist(),
            "properties": props,
        }

        out_path = os.path.join(args.out_dir, f"{mid}.json")
        # Windows 文件名过长兜底
        if len(os.path.basename(out_path)) > 200:
            out_path = os.path.join(args.out_dir, f"{args.dataset}_{i:06d}.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        ok_cnt += 1
        if (i + 1) % 100 == 0:
            print(f"[INFO] exported {ok_cnt}/{i+1}, skipped={skipped}")

    meta = {
        "source": "jarvis-tools (robust downloader)",
        "dataset": args.dataset,
        "exported": ok_cnt,
        "skipped": skipped,
        "max_atoms": args.max_atoms,
        "cache_zip": zip_path,
        "note": "deltaG_H is often missing in structure databases and may be placeholder (0.0) here.",
    }
    with open(os.path.join(args.out_dir, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE] Export finished.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
