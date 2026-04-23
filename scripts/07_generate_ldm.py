"""Phase 5.b: class-conditional sampling + FID + metadata."""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.audit import lesion_classes
from src.eval.fid_gate import resolve_fid_threshold
from src.train.ldm_trainer import build_unet, build_vae, sample_ldm
from src.utils.config import load_with_base
from src.utils.logging import get_logger


def _compute_fid(real_dir: Path, fake_dir: Path) -> float:
    try:
        result = subprocess.run(
            ["python", "-m", "pytorch_fid", str(real_dir), str(fake_dir), "--batch-size", "32"],
            capture_output=True, text=True, check=True)
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
    Image.fromarray(grid).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ldm-config", default="configs/diffusion/ldm.yaml")
    ap.add_argument("--vae-config", default="configs/diffusion/vae.yaml")
    args = ap.parse_args()
    cfg_ldm = load_with_base(args.ldm_config, base_path=ROOT / "configs" / "base.yaml")
    cfg_vae = load_with_base(args.vae_config, base_path=ROOT / "configs" / "base.yaml")
    cfg = {**cfg_ldm, "vae": cfg_vae["vae"]}
    log = get_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = cfg["classes"]

    vae = build_vae(cfg).to(device).eval()
    vae.load_state_dict(torch.load(cfg["ldm"]["vae_ckpt"], map_location=device)["state_dict"])
    unet = build_unet(cfg, num_classes=len(classes)).to(device).eval()
    unet_ckpt = Path(cfg["paths"]["outputs_root"]) / cfg["run"]["out_subdir"] / "best.pth"
    unet.load_state_dict(torch.load(unet_ckpt, map_location=device)["state_dict"])

    synth_out = Path(cfg["paths"]["outputs_root"]) / "07_ldm_synth"
    synth_out.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(Path(cfg["paths"]["outputs_root"]) / "01_audit" / "train_labels.csv")

    fid_rows = []
    meta_rows = []
    lesions = lesion_classes(classes)
    for c in lesions:
        ci = classes.index(c)
        safe = c.replace(" ", "_")
        real_count = int(audit[c].sum())
        n_target = min(
            max(cfg["generation"]["target_per_class"] - real_count, 0),
            int(cfg["generation"]["cap_factor"] * real_count),
        )
        n_gen = max(n_target, cfg["generation"]["samples_for_fid"])
        log.info(f"{c}: sampling {n_gen}")
        imgs = []
        batch = 16
        for s in range(0, n_gen, batch):
            cur = min(batch, n_gen - s)
            imgs.append(sample_ldm(cfg, unet, vae, ci, cur, device))
        imgs = np.concatenate(imgs, axis=0)

        cls_dir = synth_out / "by_class" / safe
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(imgs):
            arr = im[..., 0] if im.shape[-1] == 1 else im
            Image.fromarray(arr).save(cls_dir / f"{safe}_{i:05d}.png")
        _save_grid(imgs, cls_dir / "preview.png", cfg["generation"]["preview_grid"])

        # real pool for FID — reuse stylegan pool if present
        real_pool = ROOT / "outputs/04_stylegan/pools" / safe
        fid = _compute_fid(real_pool, cls_dir) if real_pool.exists() else float("nan")
        threshold = resolve_fid_threshold(cfg["ldm"]["fid_max_accept"], real_count)
        fid_rows.append({"class": c, "fid": fid, "n_real": real_count,
                          "n_generated": n_gen, "fid_threshold": threshold})
        log.info(f"  FID={fid:.2f}  threshold={threshold:.1f}")

        if fid > threshold or not np.isfinite(fid):
            log.warning(f"  {c}: FID above threshold ({threshold:.1f}) — NOT added to metadata")
            continue
        for i in range(n_target):
            row = {"image_id": f"ldm_{safe}_{i:05d}",
                   "path": str(cls_dir / f"{safe}_{i:05d}.png"),
                   "source": "ldm"}
            for cc in classes:
                row[cc] = 1 if cc == c else 0
            meta_rows.append(row)

    pd.DataFrame(fid_rows).to_csv(synth_out / "fid_report.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(synth_out / "synthetic_metadata.csv", index=False)
    log.info(f"Wrote {synth_out/'synthetic_metadata.csv'}")


if __name__ == "__main__":
    main()
