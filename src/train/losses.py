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

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        
        # İhtimalleri güvenli alana hapset (Log(0) imkansızlaşır)
        p = torch.clamp(p, min=self.eps, max=1.0 - self.eps)
        
        p_pos = p
        p_neg = 1 - p
        
        if self.clip > 0:
            p_neg = torch.clamp(p_neg - self.clip, min=0.0)
            
        # === DÜZELTME BURADA: Ağırlık çarpanı p_neg değil, p_pos olmalı! ===
        loss_pos = -targets * torch.log(p_pos) * ((1 - p_pos) ** self.gamma_pos)
        loss_neg = -(1 - targets) * torch.log(torch.clamp(p_neg, min=self.eps)) * (p_pos ** self.gamma_neg)
        # ====================================================================
        
        loss = loss_pos + loss_neg
        return loss.mean()


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

    def forward(self, logits, targets):
        # === KRİTİK DÜZELTME 1: .detach() ile sonsuz gradyan döngüsünü kır ===
        p = torch.sigmoid(logits).detach()
        
        # Hedefler artık gradyan taşımayan, sabit ve güvenli sayılar oldu
        bootstrapped_targets = self.beta * targets + (1.0 - self.beta) * p
        
        return self.asl(logits, bootstrapped_targets)


def build_loss(name: str, pos_weight: torch.Tensor | None = None) -> nn.Module:
    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "asymmetric":
        return AsymmetricLoss()
    if name == "bootstrapped_asl":
        return BootstrappedASL()
    raise ValueError(f"unknown loss: {name}")
