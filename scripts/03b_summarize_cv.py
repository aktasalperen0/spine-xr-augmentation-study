"""Aggregate per-fold metrics into cv_summary.{csv,md}."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.config import load_with_base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-folds", type=int, default=3)
    args = ap.parse_args()

    cfg = load_with_base(args.config, base_path=ROOT / "configs" / "base.yaml")
    run_dir = Path(cfg["paths"]["outputs_root"]) / cfg["run"]["out_subdir"]

    all_rows = []
    for k in range(1, args.n_folds + 1):
        path = run_dir / f"fold_{k}" / "metrics_at_tuned.csv"
        if not path.exists():
            print(f"missing: {path}")
            continue
        m = pd.read_csv(path)
        m["fold"] = k
        all_rows.append(m)
    if not all_rows:
        print("No fold metrics found.")
        return
    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(run_dir / "cv_per_fold.csv", index=False)

    agg = full.groupby("class").agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        ap_mean=("ap", "mean"), ap_std=("ap", "std"),
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
    ).reset_index()
    agg.to_csv(run_dir / "cv_summary.csv", index=False)

    lines = [f"# CV Summary — {cfg['run']['name']}", ""]
    cols = ["class", "f1_mean", "f1_std", "ap_mean", "auroc_mean", "precision_mean", "recall_mean"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in agg.iterrows():
        lines.append("| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols
        ) + " |")
    (run_dir / "cv_summary.md").write_text("\n".join(lines))
    print(f"Wrote {run_dir/'cv_summary.md'}")


if __name__ == "__main__":
    main()
