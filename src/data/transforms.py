"""Train/val transform presets. Grayscale X-ray → 3-channel (ImageNet stats)."""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _clahe_preprocess() -> A.BasicTransform:
    return A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)


def val_transform(image_size: int) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
        _clahe_preprocess(),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def baseline_train_transform(image_size: int) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
        _clahe_preprocess(),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def traditional_train_transform(image_size: int) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
        _clahe_preprocess(),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_transform(name: str, image_size: int) -> A.Compose:
    if name == "val":
        return val_transform(image_size)
    if name == "baseline":
        return baseline_train_transform(image_size)
    if name == "traditional":
        return traditional_train_transform(image_size)
    raise ValueError(f"unknown transform preset: {name}")
