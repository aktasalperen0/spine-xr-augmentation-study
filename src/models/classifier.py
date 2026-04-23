"""Multi-label classifier: timm backbone + GeM pooling + linear head."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import is_radimagenet_backbone, load_radimagenet_backbone


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x.flatten(1)


class SpineClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, dropout: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        if is_radimagenet_backbone(backbone_name):
            self.backbone = load_radimagenet_backbone(backbone_name, pretrained=pretrained)
        else:
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
            )
        feat_dim = self.backbone.num_features
        self.pool = GeM()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)
        pooled = self.pool(feat)
        return self.head(pooled)


def param_groups(model: SpineClassifier, backbone_lr: float, head_lr: float, wd: float):
    return [
        {"params": model.backbone.parameters(), "lr": backbone_lr, "weight_decay": wd},
        {"params": list(model.pool.parameters()) + list(model.head.parameters()),
         "lr": head_lr, "weight_decay": wd},
    ]
