"""Per-class F1-maximizing threshold search on val predictions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def find_best_thresholds(y_true: np.ndarray, y_score: np.ndarray,
                          grid: np.ndarray | None = None) -> np.ndarray:
    if grid is None:
        grid = np.linspace(0.05, 0.95, 37)
    C = y_true.shape[1]
    thresh = np.full(C, 0.5)
    for c in range(C):
        if y_true[:, c].sum() == 0:
            continue
        best_f1 = -1.0
        best_t = 0.5
        for t in grid:
            pred = (y_score[:, c] >= t).astype(int)
            f = f1_score(y_true[:, c], pred, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = float(t)
        thresh[c] = best_t
    return thresh
