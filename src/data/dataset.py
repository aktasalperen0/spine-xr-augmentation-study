"""Multi-label Dataset for spine X-ray classification."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SpineXRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        class_names: list[str],
        transform: Callable | None,
    ):
        self.df = df.reset_index(drop=True)
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._load_image(row["path"])
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        label = torch.tensor([row[c] for c in self.class_names], dtype=torch.float32)
        return img, label


def compute_pos_weight(df: pd.DataFrame, class_names: list[str], clip: float = 20.0) -> torch.Tensor:
    weights = []
    for c in class_names:
        pos = int(df[c].sum())
        neg = len(df) - pos
        w = neg / max(pos, 1)
        weights.append(min(w, clip))
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(df: pd.DataFrame, class_names: list[str]) -> np.ndarray:
    """Per-sample weight for WeightedRandomSampler (class-balanced). Normal rows get moderate weight."""
    class_freq = {c: max(int(df[c].sum()), 1) for c in class_names}
    sample_w = np.zeros(len(df), dtype=np.float64)
    labels = df[class_names].values
    for i in range(len(df)):
        row = labels[i]
        if row.sum() == 0:
            sample_w[i] = 1.0 / max(int((df[class_names].sum(axis=1) == 0).sum()), 1)
        else:
            active = np.where(row == 1)[0]
            inv = np.mean([1.0 / class_freq[class_names[j]] for j in active])
            sample_w[i] = inv
    sample_w /= sample_w.sum()
    return sample_w
