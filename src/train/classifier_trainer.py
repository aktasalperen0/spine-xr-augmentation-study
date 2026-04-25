"""Single source of truth for all classifier training runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import SpineXRDataset, compute_pos_weight, compute_sample_weights
from src.data.mixup import MixupConfig, apply_mixup
from src.data.transforms import build_transform
from src.models.classifier import SpineClassifier, param_groups
from src.models.ema import ModelEMA
from src.eval.metrics import compute_metrics
from src.eval.threshold import find_best_thresholds
from src.train.losses import BootstrappedASL, build_loss
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything


@dataclass
class TrainOutcome:
    best_val_macro_f1: float
    best_epoch: int
    thresholds: np.ndarray


def _linear_warmup_cosine(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    ys, ps = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=device == "cuda", dtype=torch.bfloat16):
            logits = model(imgs)
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_one_fold(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   out_dir: Path) -> TrainOutcome:
    log = get_logger()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["project"]["seed"])

    classes = cfg["classes"]
    tcfg = cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = SpineXRDataset(train_df, classes,
                               build_transform(tcfg["train_transform"], tcfg["image_size"]))
    val_ds = SpineXRDataset(val_df, classes, build_transform("val", tcfg["image_size"]))

    if tcfg.get("weighted_sampler", False):
        w = compute_sample_weights(train_df, classes)
        sampler = WeightedRandomSampler(w, num_samples=len(train_df), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=shuffle,
                               sampler=sampler, num_workers=cfg["project"]["num_workers"],
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                             num_workers=cfg["project"]["num_workers"], pin_memory=True)

    model = SpineClassifier(tcfg["backbone"], num_classes=len(classes),
                            dropout=tcfg.get("dropout", 0.3),
                            pretrained=tcfg.get("pretrained", True)).to(device)

    optimizer = torch.optim.AdamW(
        param_groups(model, tcfg["backbone_lr"], tcfg["head_lr"], tcfg["weight_decay"])
    )
    total_steps = tcfg["epochs"] * len(train_loader)
    warmup_steps = tcfg.get("warmup_epochs", 3) * len(train_loader)
    scheduler = _linear_warmup_cosine(optimizer, total_steps, warmup_steps)

    if tcfg["loss"] == "bce":
        pos_w = compute_pos_weight(train_df, classes).to(device)
        criterion = build_loss("bce", pos_weight=pos_w)
    else:
        criterion = build_loss(tcfg["loss"])

    ema = ModelEMA(model, decay=tcfg.get("ema_decay", 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    mixup_cfg = MixupConfig.from_dict(tcfg.get("mixup"))
    mixup_rng = np.random.default_rng(cfg["project"]["seed"])
    if mixup_cfg.enabled:
        log.info(f"Mixup enabled: alpha={mixup_cfg.alpha} cutmix_alpha={mixup_cfg.cutmix_alpha} "
                 f"prob={mixup_cfg.prob} apply_prob={mixup_cfg.apply_prob}")

    best_f1 = -1.0
    best_epoch = -1
    patience = tcfg.get("patience", 15)
    no_improve = 0
    history = []

    boot_enabled = bool(tcfg.get("bootstrapped_asl", False)) and tcfg["loss"] == "asymmetric"
    boot_switch_epoch = tcfg.get("warmup_epochs", 3) + 2

    for epoch in range(1, tcfg["epochs"] + 1):
        if boot_enabled and epoch > boot_switch_epoch and not isinstance(criterion, BootstrappedASL):
            criterion = build_loss("bootstrapped_asl")
            log.info(f"[epoch {epoch}] Switched to Bootstrapped ASL (beta={criterion.beta})")
            if mixup_cfg.enabled:
                mixup_cfg.enabled = False
                log.info(f"[epoch {epoch}] Disabled Mixup/CutMix to avoid double label smoothing")
        model.train()
        running = 0.0
        n = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            if mixup_cfg.enabled:
                imgs, labels = apply_mixup(imgs, labels, mixup_cfg, mixup_rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device == "cuda", dtype=torch.bfloat16):
                logits = model(imgs)
                loss = criterion(logits, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Zehirli Batch Yakalandı (NaN)! Model korunuyor ve bu batch çöpe atılıyor...")
                optimizer.zero_grad()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            running += loss.item() * imgs.size(0)
            n += imgs.size(0)
        train_loss = running / max(n, 1)

        y_true, y_score = _evaluate(ema.module, val_loader, device)
        thresh = find_best_thresholds(y_true, y_score)
        metrics_tuned = compute_metrics(y_true, y_score, classes, thresh)
        macro_f1_tuned = float(metrics_tuned.loc[metrics_tuned["class"] == "macro", "f1"].iloc[0])
        macro_f1_05 = float(compute_metrics(y_true, y_score, classes).loc[lambda d: d["class"] == "macro", "f1"].iloc[0])

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_macro_f1_tuned": macro_f1_tuned,
                        "val_macro_f1_at_0.5": macro_f1_05})
        log.info(f"[epoch {epoch:03d}] loss={train_loss:.4f}  "
                 f"val_macro_f1(tuned)={macro_f1_tuned:.4f}  (@0.5)={macro_f1_05:.4f}")

        if macro_f1_tuned > best_f1:
            best_f1 = macro_f1_tuned
            best_epoch = epoch
            no_improve = 0
            torch.save({"ema_state_dict": ema.module.state_dict(),
                        "thresholds": thresh.tolist(),
                        "cfg": cfg, "epoch": epoch},
                       out_dir / "best_ema.pth")
            metrics_tuned.to_csv(out_dir / "metrics_at_tuned.csv", index=False)
            compute_metrics(y_true, y_score, classes).to_csv(out_dir / "metrics_at_0.5.csv", index=False)
            np.savez(out_dir / "val_preds.npz", y_true=y_true, y_score=y_score, thresholds=thresh)
            (out_dir / "thresholds.json").write_text(
                pd.Series(dict(zip(classes, thresh.tolist()))).to_json(indent=2))
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stop: no improvement in {patience} epochs.")
                break

    pd.DataFrame(history).to_csv(out_dir / "train_log.csv", index=False)
    return TrainOutcome(best_val_macro_f1=best_f1, best_epoch=best_epoch,
                        thresholds=np.load(out_dir / "val_preds.npz")["thresholds"])
