"""Phase 6 helper: concatenate StyleGAN + LDM synthetic metadata into combined."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd


def main() -> None:
    out = ROOT / "outputs" / "08_combined_synth"
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for p in ["outputs/05_stylegan_synth/synthetic_metadata.csv",
              "outputs/07_ldm_synth/synthetic_metadata.csv"]:
        pp = ROOT / p
        if pp.exists():
            parts.append(pd.read_csv(pp))
    if not parts:
        print("No synthetic metadata found.")
        return
    combined = pd.concat(parts, ignore_index=True).drop_duplicates("image_id")
    combined.to_csv(out / "synthetic_metadata.csv", index=False)
    print(f"Wrote {out/'synthetic_metadata.csv'} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
