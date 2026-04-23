"""Per-class + macro + weighted metrics for multi-label classification."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.audit import NO_FINDING_CLASS


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return a dataframe: one row per class + macro/weighted rows."""
    if thresholds is None:
        thresholds = np.full(len(class_names), 0.5)
    y_pred = (y_score >= thresholds[None, :]).astype(int)

    rows = []
    for i, c in enumerate(class_names):
        yt = y_true[:, i]
        ys = y_score[:, i]
        yp = y_pred[:, i]
        pos = int(yt.sum())
        if pos == 0:
            ap = auroc = float("nan")
        else:
            ap = float(average_precision_score(yt, ys))
            auroc = float(roc_auc_score(yt, ys)) if pos < len(yt) else float("nan")
        rows.append({
            "class": c,
            "n_pos": pos,
            "threshold": float(thresholds[i]),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "ap": ap,
            "auroc": auroc,
        })

    df = pd.DataFrame(rows)

    # Macro / weighted aggregates cover lesion classes only — the "No finding"
    # auxiliary head is an easy majority class and would inflate scores.
    lesion_mask = df["class"].ne(NO_FINDING_CLASS).values
    lesion_df = df[lesion_mask]
    support = lesion_df["n_pos"].values.astype(float)
    total = support.sum()
    weights = support / total if total > 0 else np.ones_like(support) / len(support)
    macro_w = np.ones(len(lesion_df)) / max(len(lesion_df), 1)

    for name, w in [("macro", macro_w), ("weighted", weights)]:
        def wavg(col):
            vals = lesion_df[col].values
            mask = ~np.isnan(vals)
            if not mask.any():
                return float("nan")
            return float(np.sum(vals[mask] * w[mask]) / max(w[mask].sum(), 1e-12))
        df = pd.concat([df, pd.DataFrame([{
            "class": name,
            "n_pos": int(total),
            "threshold": float("nan"),
            "f1": wavg("f1"),
            "precision": wavg("precision"),
            "recall": wavg("recall"),
            "ap": wavg("ap"),
            "auroc": wavg("auroc"),
        }])], ignore_index=True)
    return df
