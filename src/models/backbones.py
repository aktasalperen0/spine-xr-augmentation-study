"""RadImageNet backbone loader.

RadImageNet (https://github.com/BMEII-AI/RadImageNet) releases ImageNet-equivalent
pretrained weights trained on ~1.35M medical images (CT/MRI/US). For spine X-ray
(out of domain for RadImageNet but still medical) the features transfer much
better than generic ImageNet. timm does not ship these weights, so we load
torchvision backbones and re-hydrate with a local state_dict.

Expected weight files (download manually and place under ``weights/``):
    weights/radimagenet_resnet50.pth      — state_dict from RadImageNet-ResNet50
    weights/radimagenet_densenet121.pth   — state_dict from RadImageNet-DenseNet121

The backbone is exposed with a ``forward_features`` method and a
``num_features`` attribute so that :class:`SpineClassifier` can treat it
identically to a timm model.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"


class _RadImageNetBackbone(nn.Module):
    def __init__(self, feature_extractor: nn.Module, num_features: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_features = num_features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.forward_features(x)


def _load_state_dict(name: str, pretrained: bool) -> dict | None:
    if not pretrained:
        return None
    path = WEIGHTS_DIR / f"radimagenet_{name}.pth"
    if not path.exists():
        raise FileNotFoundError(
            f"RadImageNet weights not found at {path}. Download from "
            "https://github.com/BMEII-AI/RadImageNet and place the state_dict "
            f"file as 'radimagenet_{name}.pth'."
        )
    
    blob = torch.load(path, map_location="cpu", weights_only=False)
    
    if hasattr(blob, "state_dict"):
        blob = blob.state_dict()
    elif isinstance(blob, dict) and "state_dict" in blob:
        blob = blob["state_dict"]
        
    # RADIMAGENET ÇEVİRMEN SÖZLÜĞÜ (Mapping)
    # Adamların kullandığı rakamlı sistemi, PyTorch'un beklediği isimlere çeviriyoruz
    prefix_map = {
        "backbone.0.": "conv1.",
        "backbone.1.": "bn1.",
        "backbone.4.": "layer1.",
        "backbone.5.": "layer2.",
        "backbone.6.": "layer3.",
        "backbone.7.": "layer4."
    }
    
    cleaned_state = {}
    for k, v in blob.items():
        new_k = k
        # Sözlükte eşleşen var mı diye bak, varsa değiştir
        for old_prefix, new_prefix in prefix_map.items():
            if new_k.startswith(old_prefix):
                new_k = new_k.replace(old_prefix, new_prefix, 1)
                break
                
        cleaned_state[new_k] = v
        
    return cleaned_state


def _build_resnet50(pretrained: bool) -> _RadImageNetBackbone:
    net = models.resnet50(weights=None)
    state = _load_state_dict("resnet50", pretrained)
    if state is not None:
        # Drop classifier head if present; tolerate both prefixed and unprefixed keys.
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        missing, unexpected = net.load_state_dict(state, strict=False)
        if any(not m.startswith("fc.") for m in missing):
            raise RuntimeError(f"Unexpected missing keys when loading RadImageNet-ResNet50: {missing}")
    feat = nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool,
        net.layer1, net.layer2, net.layer3, net.layer4,
    )
    return _RadImageNetBackbone(feat, num_features=2048)


def _build_densenet121(pretrained: bool) -> _RadImageNetBackbone:
    net = models.densenet121(weights=None)
    state = _load_state_dict("densenet121", pretrained)
    if state is not None:
        state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
        missing, unexpected = net.load_state_dict(state, strict=False)
        if any(not m.startswith("classifier.") for m in missing):
            raise RuntimeError(f"Unexpected missing keys when loading RadImageNet-DenseNet121: {missing}")
    return _RadImageNetBackbone(net.features, num_features=1024)


_BUILDERS = {
    "radimagenet_resnet50": _build_resnet50,
    "radimagenet_densenet121": _build_densenet121,
}


def is_radimagenet_backbone(name: str) -> bool:
    return name in _BUILDERS


def load_radimagenet_backbone(name: str, pretrained: bool = True) -> _RadImageNetBackbone:
    if name not in _BUILDERS:
        raise KeyError(f"Unknown RadImageNet backbone: {name}. Known: {list(_BUILDERS)}")
    return _BUILDERS[name](pretrained=pretrained)
