"""Inject synthetic single-label samples into a train fold."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def mix_synthetic_into_train(
    train_df: pd.DataFrame,
    synth_meta_csv: Path,
    class_names: list[str],
    ratio_per_class: dict[str, float],
    target_per_class: int = 1500,
    cap_factor: float = 3.0,
) -> pd.DataFrame:
    """
    ratio_per_class[c] ∈ [0,1]: fraction of (target - n_real) to fill from synth.
    Cap injection at cap_factor × n_real_positive_c.
    """
    synth = pd.read_csv(synth_meta_csv)
    keep_rows = []
    for c in class_names:
        ratio = float(ratio_per_class.get(c, 0.0))
        if ratio <= 0:
            continue
        n_real = int(train_df[c].sum())
        gap = max(target_per_class - n_real, 0)
        n_want = int(round(min(gap * ratio, cap_factor * n_real)))
        if n_want <= 0:
            continue
        pool = synth[synth[c] == 1]
        if len(pool) == 0:
            continue
        sampled = pool.sample(n=min(n_want, len(pool)), replace=False, random_state=42)
        keep_rows.append(sampled)
    if not keep_rows:
        return train_df
    synth_chosen = pd.concat(keep_rows, ignore_index=True).drop_duplicates(subset=["image_id"])
    cols = ["image_id", "path", "source"] + class_names
    synth_chosen = synth_chosen.reindex(columns=cols, fill_value=0)
    return pd.concat([train_df, synth_chosen], ignore_index=True)
