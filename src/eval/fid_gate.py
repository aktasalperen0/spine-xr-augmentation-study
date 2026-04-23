"""Dynamic FID acceptance threshold based on real-image pool size.

Rationale: smaller real pools produce noisier FID estimates and synthesis quality
is fundamentally capped by the training set size. A single fixed threshold
either over-rejects minority classes (starving the classifier of augmentation)
or over-accepts majority classes (admitting low-quality synthetic samples).
"""
from __future__ import annotations


def dynamic_fid_threshold(real_count: int) -> float:
    if real_count < 300:
        return 250.0
    if real_count < 1000:
        return 200.0
    return 150.0


def resolve_fid_threshold(cfg_value, real_count: int) -> float:
    """Accept either a numeric config value (legacy) or the literal 'dynamic'."""
    if isinstance(cfg_value, (int, float)):
        return float(cfg_value)
    if isinstance(cfg_value, str) and cfg_value.lower() == "dynamic":
        return dynamic_fid_threshold(real_count)
    raise ValueError(f"Unsupported fid_max_accept value: {cfg_value!r}")
