"""Build multi-label tables from VinDr-SpineXR CSVs."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_label_table(
    annotations_csv: Path,
    normal_metadata_csv: Path,
    abnormal_pngs_dir: Path,
    normal_pngs_dir: Path,
    class_names: list[str],
) -> pd.DataFrame:
    """Return one row per image with path + one-hot labels + split tag."""
    ann = pd.read_csv(annotations_csv)
    ann = ann[ann["lesion_type"].isin(class_names)]

    one_hot = (
        ann.assign(v=1)
        .pivot_table(index="image_id", columns="lesion_type", values="v", aggfunc="max", fill_value=0)
        .reindex(columns=class_names, fill_value=0)
        .reset_index()
    )
    one_hot["source"] = "abnormal"
    one_hot["path"] = one_hot["image_id"].apply(lambda i: str(abnormal_pngs_dir / f"{i}.png"))

    normal = pd.read_csv(normal_metadata_csv)[["image_id"]].drop_duplicates()
    for c in class_names:
        normal[c] = 0
    normal["source"] = "normal"
    normal["path"] = normal["image_id"].apply(lambda i: str(normal_pngs_dir / f"{i}.png"))

    df = pd.concat([one_hot, normal], ignore_index=True)
    cols = ["image_id", "path", "source"] + class_names
    return df[cols]


def class_counts(df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    rows = []
    for c in class_names:
        rows.append({"class": c, "n_positive": int(df[c].sum())})
    rows.append({"class": "__any__", "n_positive": int(df[class_names].sum(axis=1).gt(0).sum())})
    rows.append({"class": "__none__ (normal)", "n_positive": int(df[class_names].sum(axis=1).eq(0).sum())})
    return pd.DataFrame(rows)


def co_occurrence(df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    mat = df[class_names].T.dot(df[class_names])
    return mat
