"""Multi-label MixUp / CutMix.

timm's default Mixup coerces targets to one-hot for CE-style losses; our labels
are multi-hot (multi-label BCE), so we implement the sample mixing directly.
The soft-label convex combination of two multi-hot vectors stays a valid
Bernoulli target and BCEWithLogitsLoss handles it natively.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MixupConfig:
    enabled: bool = False
    alpha: float = 0.2         # MixUp Beta(alpha, alpha); 0 disables MixUp
    cutmix_alpha: float = 1.0  # CutMix Beta(alpha, alpha); 0 disables CutMix
    prob: float = 0.5          # probability of choosing CutMix over MixUp when both enabled
    apply_prob: float = 1.0    # probability of applying ANY mix on a batch

    @classmethod
    def from_dict(cls, d: dict | None) -> "MixupConfig":
        if not d:
            return cls(enabled=False)
        return cls(
            enabled=bool(d.get("enabled", False)),
            alpha=float(d.get("alpha", 0.2)),
            cutmix_alpha=float(d.get("cutmix_alpha", 1.0)),
            prob=float(d.get("prob", 0.5)),
            apply_prob=float(d.get("apply_prob", 1.0)),
        )


def _rand_bbox(h: int, w: int, lam: float, rng: np.random.Generator):
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    cy = rng.integers(0, h)
    cx = rng.integers(0, w)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, h)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    return y1, y2, x1, x2


def apply_mixup(imgs: torch.Tensor, labels: torch.Tensor, cfg: MixupConfig,
                rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MixUp or CutMix in-place on a batch. labels must be float multi-hot."""
    if not cfg.enabled or rng.random() >= cfg.apply_prob:
        return imgs, labels

    use_cutmix = cfg.cutmix_alpha > 0 and (cfg.alpha <= 0 or rng.random() < cfg.prob)
    alpha = cfg.cutmix_alpha if use_cutmix else cfg.alpha
    if alpha <= 0:
        return imgs, labels

    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(imgs.size(0), device=imgs.device)

    if use_cutmix:
        h, w = imgs.shape[-2:]
        y1, y2, x1, x2 = _rand_bbox(h, w, lam, rng)
        imgs[:, :, y1:y2, x1:x2] = imgs[perm, :, y1:y2, x1:x2]
        # Recompute lam from actual patched area for label mixing accuracy.
        lam = 1.0 - ((y2 - y1) * (x2 - x1)) / float(h * w)
    else:
        imgs = lam * imgs + (1.0 - lam) * imgs[perm]

    labels = lam * labels + (1.0 - lam) * labels[perm]
    return imgs, labels
