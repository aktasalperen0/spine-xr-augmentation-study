"""Phase 1: write 3-fold multilabel-stratified split CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.splits import make_folds
from src.utils.config import load_config
from src.utils.logging import get_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger()
    classes = cfg["classes"]

    audit_dir = Path(cfg["paths"]["outputs_root"]) / "01_audit"
    out_dir = Path(cfg["paths"]["outputs_root"]) / "02_splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(audit_dir / "train_labels.csv")
    df = make_folds(df, classes, cfg["cv"]["n_splits"], cfg["cv"]["random_state"])
    df.to_csv(out_dir / "train_with_folds.csv", index=False)

    summary_rows = []
    for k in range(1, cfg["cv"]["n_splits"] + 1):
        tr = df[df["fold"] != k]
        va = df[df["fold"] == k]
        tr.to_csv(out_dir / f"fold_{k}_train.csv", index=False)
        va.to_csv(out_dir / f"fold_{k}_val.csv", index=False)
        row = {"fold": k, "n_train": len(tr), "n_val": len(va)}
        for c in classes:
            row[f"train_{c}"] = int(tr[c].sum())
            row[f"val_{c}"] = int(va[c].sum())
            assert va[c].sum() >= 1, f"fold {k} val has 0 positives for {c}"
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "fold_summary.csv", index=False)
    log.info(f"Wrote 3 folds + fold_summary.csv to {out_dir}")


if __name__ == "__main__":
    main()
