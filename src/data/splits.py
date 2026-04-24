"""Patient-level multilabel stratified split.

Groups images by `study_id` so all images from the same patient/study land in the
same fold — prevents leakage between train and val.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def make_folds(df: pd.DataFrame, class_names: list[str], n_splits: int, seed: int) -> pd.DataFrame:
    if "study_id" not in df.columns:
        raise ValueError(
            "make_folds requires a 'study_id' column. Re-run scripts/01_audit.py "
            "to regenerate label tables with study-level metadata."
        )

    # Study-level label table: a study is positive for class c if any of its
    # images is. This preserves multi-label structure at the patient level.
    study_df = (
        df.groupby("study_id")[class_names]
        .max()
        .reset_index()
    )
    y = study_df[class_names].values.astype(int)
    # Keep the "no positives" stratification trick so fully-normal studies
    # are balanced across folds too.
    if (y.sum(axis=1) == 0).any():
        y_aug = np.concatenate([y, (y.sum(axis=1) == 0).astype(int)[:, None]], axis=1)
    else:
        y_aug = y

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    study_fold = np.full(len(study_df), -1, dtype=int)
    for i, (_, val_idx) in enumerate(mskf.split(np.zeros(len(study_df)), y_aug), start=1):
        study_fold[val_idx] = i
    mapping = dict(zip(study_df["study_id"].values, study_fold))

    out = df.copy()
    out["fold"] = out["study_id"].map(mapping).astype(int)

    # Sanity: no study spans multiple folds.
    nunique = out.groupby("study_id")["fold"].nunique()
    assert nunique.max() == 1, "Patient-level split invariant violated (study spans folds)"
    assert (out["fold"] > 0).all(), "Some rows did not receive a fold assignment"
    return out
