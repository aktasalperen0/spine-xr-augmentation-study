"""Phase 5: train VAE then LDM UNet."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.audit import lesion_classes
from src.utils.config import load_with_base
from src.utils.logging import get_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/diffusion/vae.yaml or ldm.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_with_base(args.config, base_path=ROOT / "configs" / "base.yaml")
    log = get_logger()

    if "vae" in cfg and "ldm" not in cfg:
        from src.train.ldm_trainer import train_vae as trainer
        stage = "vae"
    else:
        from src.train.ldm_trainer import train_ldm as trainer
        stage = "ldm"

    if args.smoke:
        if stage == "vae":
            cfg["vae"]["epochs"] = 2
        else:
            cfg["ldm"]["max_steps"] = 200

    audit = Path(cfg["paths"]["outputs_root"]) / "01_audit" / "train_labels.csv"
    df = pd.read_csv(audit)
    # For VAE/UNet training, exclude multi-label images (single-label + normal only)
    if stage == "ldm":
        # Count lesion positives only — "No finding" aux column must not break the filter.
        single = df[lesion_classes(cfg["classes"])].sum(axis=1)
        df = df[(single == 1) | (single == 0)].reset_index(drop=True)
    out_dir = Path(cfg["paths"]["outputs_root"]) / cfg["run"]["out_subdir"]
    ckpt = trainer(cfg, df, out_dir)
    log.info(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
