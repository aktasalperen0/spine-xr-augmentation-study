"""Phase 4.a: build per-class single-label real pools for StyleGAN training."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import pandas as pd
from tqdm import tqdm

from src.data.audit import lesion_classes
from src.utils.config import load_with_base
from src.utils.logging import get_logger


def _save_resized(src: Path, dst: Path, size: int) -> None:
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    h, w = img.shape
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = 0 * img[:1, :1].repeat(size, axis=0).repeat(size, axis=1)  # zero-pad square
    import numpy as np
    canvas = np.zeros((size, size), dtype=img.dtype)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = img
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), canvas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gan/base_stylegan2_ada.yaml")
    args = ap.parse_args()
    cfg = load_with_base(args.config, base_path=ROOT / "configs" / "base.yaml")
    log = get_logger()
    classes = lesion_classes(cfg["classes"])  # GAN pools never cover the No finding aux class
    resolution = cfg["stylegan"]["resolution"]

    audit_dir = Path(cfg["paths"]["outputs_root"]) / "01_audit"
    df = pd.read_csv(audit_dir / "train_labels.csv")
    df = df[df["source"] == "abnormal"].reset_index(drop=True)

    if cfg["pool"]["single_label_only"]:
        n_labels = df[classes].sum(axis=1)
        df = df[n_labels == 1].reset_index(drop=True)
    log.info(f"Abnormal single-label images: {len(df)}")

    all_dir = ROOT / cfg["pool"]["all_abnormal_dir"]
    if all_dir.exists():
        shutil.rmtree(all_dir)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="all_abnormal"):
        _save_resized(Path(row["path"]), all_dir / f"{row['image_id']}.png", resolution)

    for c in classes:
        sub = df[df[c] == 1]
        cls_dir = ROOT / cfg["pool"]["per_class_dir"] / c.replace(" ", "_")
        if cls_dir.exists():
            shutil.rmtree(cls_dir)
        for _, row in tqdm(sub.iterrows(), total=len(sub), desc=c):
            _save_resized(Path(row["path"]), cls_dir / f"{row['image_id']}.png", resolution)
        log.info(f"  {c}: {len(sub)} images → {cls_dir}")


if __name__ == "__main__":
    main()
