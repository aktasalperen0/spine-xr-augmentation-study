from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_configs(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_configs(out[k], v)
        else:
            out[k] = v
    return out


def load_with_base(path: str | Path, base_path: str | Path = "configs/base.yaml") -> dict:
    base = load_config(base_path)
    cfg = load_config(path)
    return merge_configs(base, cfg)
