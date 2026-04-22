"""Phase 2/3/6: train classifier for one fold under a given config."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.synth_mix import mix_synthetic_into_train
from src.train.classifier_trainer import train_one_fold
from src.utils.config import load_with_base
from src.utils.logging import get_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="Override epochs=3 and patience=2 for quick sanity run.")
    args = ap.parse_args()

    cfg = load_with_base(args.config, base_path=ROOT / "configs" / "base.yaml")
    log = get_logger()

    if args.smoke:
        cfg["train"]["epochs"] = 3
        cfg["train"]["patience"] = 2
        cfg["run"]["out_subdir"] = cfg["run"]["out_subdir"] + "_smoke"
        log.info("SMOKE MODE: epochs=3, patience=2")

    splits_dir = Path(cfg["paths"]["outputs_root"]) / "02_splits"
    train_df = pd.read_csv(splits_dir / f"fold_{args.fold}_train.csv")
    val_df = pd.read_csv(splits_dir / f"fold_{args.fold}_val.csv")

    synth_mix_cfg = cfg["train"].get("synth_mix")
    if synth_mix_cfg:
        meta = ROOT / synth_mix_cfg["metadata_csv"]
        if meta.exists():
            log.info(f"Mixing synthetic samples from {meta}")
            train_df = mix_synthetic_into_train(
                train_df, meta, cfg["classes"],
                ratio_per_class=synth_mix_cfg["ratio_per_class"],
                target_per_class=synth_mix_cfg.get("target_per_class", 1500),
                cap_factor=synth_mix_cfg.get("cap_factor", 3.0),
            )
            log.info(f"Train size after synth mix: {len(train_df)}")
        else:
            log.warning(f"Synth metadata not found: {meta} — training without synthetic samples")

    out_dir = Path(cfg["paths"]["outputs_root"]) / cfg["run"]["out_subdir"] / f"fold_{args.fold}"
    result = train_one_fold(cfg, train_df, val_df, out_dir)
    log.info(f"Fold {args.fold}: best macro F1 = {result.best_val_macro_f1:.4f} @ epoch {result.best_epoch}")


if __name__ == "__main__":
    main()
