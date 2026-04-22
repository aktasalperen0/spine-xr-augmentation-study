"""Phase 8: assemble final_report.md with all tables + image grids."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.config import load_config


def _maybe_read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_root = Path(cfg["paths"]["outputs_root"])

    lines = [
        "# Spine XR Augmentation — Final Report",
        "",
        "## 1. Dataset",
    ]
    audit_md = out_root / "01_audit" / "audit_report.md"
    if audit_md.exists():
        lines += [audit_md.read_text(), ""]

    lines += ["## 2. Cross-validation results (per config)"]
    for sub in ["03_baseline", "03_traditional", "06_stylegan_aug", "07_ldm_aug", "08_combined_aug"]:
        cv = out_root / sub / "cv_summary.md"
        if cv.exists():
            lines += [f"### {sub}", cv.read_text(), ""]

    lines += ["## 3. Test set comparison (3-fold ensemble + bootstrap CI)"]
    cmp_md = out_root / "09_test" / "comparison.md"
    if cmp_md.exists():
        lines += [cmp_md.read_text(), ""]

    lines += ["## 4. Generative model quality (FID per class)"]
    for name, path in [("StyleGAN2-ADA", out_root / "05_stylegan_synth" / "fid_report.csv"),
                        ("Latent Diffusion", out_root / "07_ldm_synth" / "fid_report.csv")]:
        df = _maybe_read(path)
        if df is not None:
            lines += [f"### {name}", df.to_markdown(index=False, floatfmt=".2f"), ""]

    lines += ["## 5. Preview grids"]
    for name, d in [("StyleGAN2-ADA", out_root / "05_stylegan_synth" / "by_class"),
                     ("Latent Diffusion", out_root / "07_ldm_synth" / "by_class")]:
        if not d.exists():
            continue
        lines.append(f"### {name}")
        for sub in sorted(d.iterdir()):
            grid = sub / "preview.png"
            if grid.exists():
                rel = grid.relative_to(out_root.parent)
                lines.append(f"- **{sub.name}**: `{rel}`")
        lines.append("")

    lines += [
        "## 6. Tartışma (TR)",
        "- Her konfig için macro + weighted F1 test setinde üstte raporlandı.",
        "- Per-class tabloları rare sınıflarda augmentation katkısını gösterir (Vertebral collapse, Surgical implant, Foraminal stenosis).",
        "- FID eşiği (150) üzerindeki sınıfların sentetikleri classifier'a enjekte edilmemiştir; şeffaflık için FID tablosu raporda.",
        "- Bootstrap CI macro F1 için istatistiksel belirsizliği verir.",
    ]
    (out_root / "final_report.md").write_text("\n".join(lines))
    print(f"Wrote {out_root/'final_report.md'}")


if __name__ == "__main__":
    main()
