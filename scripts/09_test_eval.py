"""Phase 7: evaluate every classifier config on held-out test set, with bootstrap CI."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.eval.metrics import compute_metrics
from src.eval.test_eval import bootstrap_macro_f1, ensemble_predict
from src.utils.config import load_with_base
from src.utils.logging import get_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True,
                    help="classifier config paths (baseline, traditional, stylegan_aug, ...)")
    ap.add_argument("--n-folds", type=int, default=3)
    args = ap.parse_args()
    log = get_logger()

    base_cfg = load_with_base(args.configs[0], base_path=ROOT / "configs" / "base.yaml")
    test_df = pd.read_csv(Path(base_cfg["paths"]["outputs_root"]) / "01_audit" / "test_labels.csv")
    classes = base_cfg["classes"]
    y_true = test_df[classes].values.astype(int)

    comparison_rows = []
    for cfg_path in args.configs:
        cfg = load_with_base(cfg_path, base_path=ROOT / "configs" / "base.yaml")
        run_dir = Path(cfg["paths"]["outputs_root"]) / cfg["run"]["out_subdir"]
        out = Path(cfg["paths"]["outputs_root"]) / "09_test" / cfg["run"]["name"]
        out.mkdir(parents=True, exist_ok=True)

        log.info(f"[{cfg['run']['name']}] ensemble predict...")
        scores, thresh = ensemble_predict(run_dir, test_df, cfg, n_folds=args.n_folds)

        m_tuned = compute_metrics(y_true, scores, classes, thresh)
        m_05 = compute_metrics(y_true, scores, classes)
        m_tuned.to_csv(out / "test_metrics_at_tuned.csv", index=False)
        m_05.to_csv(out / "test_metrics_at_0.5.csv", index=False)

        mean, lo, hi = bootstrap_macro_f1(y_true, scores, thresh, classes, n_boot=1000)
        summary = {
            "config": cfg["run"]["name"],
            "macro_f1_tuned": float(m_tuned.loc[m_tuned["class"] == "macro", "f1"].iloc[0]),
            "weighted_f1_tuned": float(m_tuned.loc[m_tuned["class"] == "weighted", "f1"].iloc[0]),
            "macro_ap": float(m_tuned.loc[m_tuned["class"] == "macro", "ap"].iloc[0]),
            "weighted_ap": float(m_tuned.loc[m_tuned["class"] == "weighted", "ap"].iloc[0]),
            "macro_auroc": float(m_tuned.loc[m_tuned["class"] == "macro", "auroc"].iloc[0]),
            "bootstrap_macro_f1_mean": mean,
            "bootstrap_macro_f1_ci_low": lo,
            "bootstrap_macro_f1_ci_high": hi,
        }
        pd.Series(summary).to_json(out / "summary.json", indent=2)
        comparison_rows.append(summary)

    cmp_df = pd.DataFrame(comparison_rows)
    cmp_out = Path(base_cfg["paths"]["outputs_root"]) / "09_test" / "comparison.csv"
    cmp_df.to_csv(cmp_out, index=False)

    lines = ["# Test Set Comparison", "", cmp_df.to_markdown(index=False, floatfmt=".4f")]
    (cmp_out.with_suffix(".md")).write_text("\n".join(lines))
    log.info(f"Wrote {cmp_out} and {cmp_out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
