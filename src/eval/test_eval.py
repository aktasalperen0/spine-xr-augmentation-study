"""3-fold ensemble test evaluation + bootstrap CI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SpineXRDataset
from src.data.transforms import build_transform
from src.eval.metrics import compute_metrics
from src.models.classifier import SpineClassifier


@torch.no_grad()
def _predict_one(ckpt_path: Path, test_df: pd.DataFrame, cfg: dict, device: str) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location=device)
    run_cfg = ckpt["cfg"]
    model = SpineClassifier(run_cfg["train"]["backbone"], num_classes=len(cfg["classes"]),
                             dropout=run_cfg["train"].get("dropout", 0.3),
                             pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["ema_state_dict"])
    ds = SpineXRDataset(test_df, cfg["classes"],
                        build_transform("val", run_cfg["train"]["image_size"]))
    loader = DataLoader(ds, batch_size=run_cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["project"]["num_workers"], pin_memory=True)
    tta = cfg.get("eval", {}).get("tta", "hflip")
    scores = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast(enabled=device == "cuda"):
            probs = torch.sigmoid(model(imgs))
            if tta == "hflip":
                probs = 0.5 * (probs + torch.sigmoid(model(imgs.flip(-1))))
        scores.append(probs.float().cpu().numpy())
    return np.concatenate(scores, axis=0)


def ensemble_predict(run_dir: Path, test_df: pd.DataFrame, cfg: dict,
                      n_folds: int = 3) -> tuple[np.ndarray, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_fold_scores = []
    per_fold_thresh = []
    for k in range(1, n_folds + 1):
        ckpt = run_dir / f"fold_{k}" / "best_ema.pth"
        if not ckpt.exists():
            continue
        per_fold_scores.append(_predict_one(ckpt, test_df, cfg, device))
        per_fold_thresh.append(torch.load(ckpt, map_location="cpu")["thresholds"])
    scores = np.mean(per_fold_scores, axis=0)
    thresh = np.mean(per_fold_thresh, axis=0)
    return scores, thresh


def bootstrap_macro_f1(y_true: np.ndarray, y_score: np.ndarray, thresh: np.ndarray,
                       classes: list[str], n_boot: int = 1000,
                       seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = compute_metrics(y_true[idx], y_score[idx], classes, thresh)
        vals.append(float(m.loc[m["class"] == "macro", "f1"].iloc[0]))
    vals = np.array(vals)
    return float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
