"""3-fold multilabel stratified split."""
from __future__ import annotations

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def make_folds(df: pd.DataFrame, class_names: list[str], n_splits: int, seed: int) -> pd.DataFrame:
    y = df[class_names].values.astype(int)
    # Guarantee that rows with no positives in any column get stratified too.
    if (y.sum(axis=1) == 0).any():
        y_aug = np.concatenate([y, (y.sum(axis=1) == 0).astype(int)[:, None]], axis=1)
    else:
        y_aug = y
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_col = np.full(len(df), -1, dtype=int)
    for i, (_, val_idx) in enumerate(mskf.split(np.zeros(len(df)), y_aug), start=1):
        fold_col[val_idx] = i
    out = df.copy()
    out["fold"] = fold_col
    return out
