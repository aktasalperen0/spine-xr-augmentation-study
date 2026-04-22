"""Phase 4.b: invoke stylegan2-ada-pytorch train.py for base and per-class fine-tune.

Assumes NVIDIA's stylegan2-ada-pytorch repo has been cloned to cfg.stylegan.repo_dir.
Run on Colab A100. Locally this prints the commands without executing.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_with_base
from src.utils.logging import get_logger


def _dataset_tool(repo: Path, source: Path, dest: Path) -> list[str]:
    return ["python", str(repo / "dataset_tool.py"),
            "--source", str(source), "--dest", str(dest)]


def _train_cmd(repo: Path, data_zip: Path, outdir: Path, scfg: dict,
               kimg: int, resume: str | None) -> list[str]:
    cmd = ["python", str(repo / "train.py"),
           "--outdir", str(outdir),
           "--data", str(data_zip),
           "--cfg", scfg["cfg"],
           "--gpus", str(scfg["gpus"]),
           "--kimg", str(kimg),
           "--snap", str(scfg["snap"]),
           "--mirror", str(scfg["mirror"]),
           "--augpipe", scfg["augpipe"]]
    if resume:
        cmd += ["--resume", resume]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gan/base_stylegan2_ada.yaml")
    ap.add_argument("--stage", choices=["base", "per_class"], required=True)
    ap.add_argument("--class", dest="cls", default=None, help="required if stage=per_class")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_with_base(args.config, base_path=ROOT / "configs" / "base.yaml")
    log = get_logger()
    scfg = cfg["stylegan"]
    repo = Path(scfg["repo_dir"])
    if not repo.exists():
        log.warning(f"StyleGAN repo not found at {repo}. Clone it on Colab:\n"
                    f"  git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git {repo}")

    out_root = ROOT / cfg["run"]["out_subdir"] if "run" in cfg else ROOT / "outputs/04_stylegan"
    out_root = Path(cfg["paths"]["outputs_root"]) / cfg["run"]["out_subdir"]
    out_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "base":
        src = ROOT / cfg["pool"]["all_abnormal_dir"]
        zip_path = out_root / "base_dataset.zip"
        outdir = out_root / "base_train"
        resume = None
        kimg = scfg["kimg_base"]
    else:
        assert args.cls, "--class required for per_class"
        src = ROOT / cfg["pool"]["per_class_dir"] / args.cls.replace(" ", "_")
        zip_path = out_root / f"{args.cls.replace(' ', '_')}_dataset.zip"
        outdir = out_root / f"finetune_{args.cls.replace(' ', '_')}"
        base_pkl = sorted((out_root / "base_train").rglob("network-snapshot-*.pkl"))
        resume = str(base_pkl[-1]) if base_pkl else None
        kimg = scfg["kimg_finetune"]

    cmds = [_dataset_tool(repo, src, zip_path),
            _train_cmd(repo, zip_path, outdir, scfg, kimg, resume)]
    for cmd in cmds:
        log.info("> " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
