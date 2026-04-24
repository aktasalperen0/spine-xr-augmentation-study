"""Loss functions for multi-label imbalanced classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Ben-Baruch et al. 2021. Default hyperparams from the paper."""

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss *= (1 - pt) ** gamma
        return -loss.mean()


class BootstrappedASL(nn.Module):
    """Soft-bootstrapping wrapper around AsymmetricLoss.

    Replaces hard targets with ``beta * targets + (1 - beta) * sigmoid(logits)``
    so the model can self-correct noisy radiologist labels. Use after an initial
    warmup phase — bootstrapping a cold model drags training toward its own
    random predictions.
    """

    def __init__(self, beta: float = 0.85, gamma_neg: float = 4.0,
                 gamma_pos: float = 0.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.beta = beta
        self.asl = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos,
                                  clip=clip, eps=eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            soft = torch.sigmoid(logits).detach()
        new_targets = self.beta * targets + (1.0 - self.beta) * soft
        return self.asl(logits, new_targets)


def build_loss(name: str, pos_weight: torch.Tensor | None = None) -> nn.Module:
    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "asymmetric":
        return AsymmetricLoss()
    if name == "bootstrapped_asl":
        return BootstrappedASL()
    raise ValueError(f"unknown loss: {name}")
