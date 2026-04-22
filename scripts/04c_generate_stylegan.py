"""Phase 4.c: generate per-class synthetic images from fine-tuned StyleGAN pickles,
compute FID vs real pool, write metadata and preview grids.
"""
from __future__ import annotations

import argparse
import math
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.utils.config import load_with_base
from src.utils.logging import get_logger


def _load_G(pkl_path: Path, device: str):
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    G = data["G_ema"].to(device).eval()
    return G


@torch.no_grad()
def _generate(G, n: int, device: str, truncation: float = 0.7, batch: int = 16):
    imgs = []
    label = torch.zeros([1, G.c_dim], device=device) if G.c_dim > 0 else None
    for start in range(0, n, batch):
        cur = min(batch, n - start)
        z = torch.randn(cur, G.z_dim, device=device)
        c = label.repeat(cur, 1) if label is not None else None
        img = G(z, c, truncation_psi=truncation, noise_mode="const")
        img = (img.clamp(-1, 1) + 1) * 127.5
        img = img.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        imgs.append(img)
    return np.concatenate(imgs, axis=0)


def _compute_fid(real_dir: Path, fake_dir: Path) -> float:
    try:
        result = subprocess.run(
            ["python", "-m", "pytorch_fid", str(real_dir), str(fake_dir), "--batch-size", "32"],
            capture_output=True, text=True, check=True,
        )
        for tok in result.stdout.split():
            try:
                return float(tok)
            except ValueError:
                pass
    except Exception as e:
        print(f"FID failure: {e}")
    return float("nan")


def _save_grid(imgs: np.ndarray, path: Path, n: int = 36) -> None:
    k = int(math.sqrt(n))
    sel = imgs[:k * k]
    h, w = sel.shape[1:3]
    grid = np.zeros((k * h, k * w) + sel.shape[3:], dtype=sel.dtype)
    for i, im in enumerate(sel):
        r, c = divmod(i, k)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    if grid.ndim == 3 and grid.shape[-1] == 1:
        grid = grid[..., 0]
    Image.fromarray(grid).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gan/base_stylegan2_ada.yaml")
    args = ap.parse_args()
    cfg = load_with_base(args.config, base_path=ROOT / "configs" / "base.yaml")
    log = get_logger()
    classes = cfg["classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gan_out = Path(cfg["paths"]["outputs_root"]) / cfg["run"]["out_subdir"]
    synth_out = Path(cfg["paths"]["outputs_root"]) / "05_stylegan_synth"
    synth_out.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(Path(cfg["paths"]["outputs_root"]) / "01_audit" / "train_labels.csv")
    real_counts = {c: int(audit[c].sum()) for c in classes}

    fid_rows = []
    meta_rows = []
    for c in classes:
        safe = c.replace(" ", "_")
        pkls = sorted((gan_out / f"finetune_{safe}").rglob("network-snapshot-*.pkl"))
        if not pkls:
            log.warning(f"{c}: no fine-tuned pkl — skipping")
            continue
        pkl = pkls[-1]
        G = _load_G(pkl, device)
        n_target = min(
            max(cfg["generation"]["target_per_class"] - real_counts[c], 0),
            int(cfg["generation"]["cap_factor"] * real_counts[c]),
        )
        n_gen = max(n_target, cfg["generation"]["samples_for_fid"])
        log.info(f"{c}: generating {n_gen} samples (target inject {n_target})")
        imgs = _generate(G, n_gen, device)

        cls_dir = synth_out / "by_class" / safe
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(imgs):
            im_arr = im[..., 0] if im.shape[-1] == 1 else im
            Image.fromarray(im_arr).save(cls_dir / f"{safe}_{i:05d}.png")

        _save_grid(imgs, cls_dir / "preview.png", cfg["generation"]["preview_grid"])
        real_pool = ROOT / cfg["pool"]["per_class_dir"] / safe
        fid = _compute_fid(real_pool, cls_dir)
        fid_rows.append({"class": c, "fid": fid, "n_real": real_counts[c], "n_generated": n_gen})
        log.info(f"  {c}: FID={fid:.2f}")

        if fid > cfg["stylegan"]["fid_max_accept"] or not np.isfinite(fid):
            log.warning(f"  {c}: FID above threshold — NOT added to metadata")
            continue
        for i in range(n_target):
            row = {"image_id": f"stylegan_{safe}_{i:05d}",
                   "path": str(cls_dir / f"{safe}_{i:05d}.png"),
                   "source": "stylegan"}
            for cc in classes:
                row[cc] = 1 if cc == c else 0
            meta_rows.append(row)

    pd.DataFrame(fid_rows).to_csv(synth_out / "fid_report.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(synth_out / "synthetic_metadata.csv", index=False)
    log.info(f"Wrote {synth_out/'synthetic_metadata.csv'}")


if __name__ == "__main__":
    main()
