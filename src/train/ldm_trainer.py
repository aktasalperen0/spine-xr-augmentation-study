"""MONAI-generative Latent Diffusion Model: VAE + class-conditional UNet.

Designed to run on Colab A100. Keeps MONAI API usage in one place so that
version issues can be fixed here without touching scripts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler

from src.data.dataset import SpineXRDataset
from src.data.transforms import build_transform
from src.utils.logging import get_logger


# ---------- VAE ----------

def build_vae(cfg: dict) -> AutoencoderKL:
    v = cfg["vae"]
    return AutoencoderKL(
        spatial_dims=2, in_channels=3, out_channels=3,
        num_channels=v["channels"],
        latent_channels=v["latent_channels"],
        num_res_blocks=v["num_res_blocks"],
        attention_levels=v["attention_levels"],
        with_encoder_nonlocal_attn=True,
        with_decoder_nonlocal_attn=True,
    )


def train_vae(cfg: dict, train_df: pd.DataFrame, out_dir: Path) -> Path:
    log = get_logger()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v = cfg["vae"]

    ds = SpineXRDataset(train_df, cfg["classes"], build_transform("val", v["image_size"]))
    loader = DataLoader(ds, batch_size=v["batch_size"], shuffle=True,
                        num_workers=cfg["project"]["num_workers"], pin_memory=True, drop_last=True)

    vae = build_vae(cfg).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=v["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    best = float("inf")
    ckpt_path = out_dir / "best.pth"
    for epoch in range(1, v["epochs"] + 1):
        vae.train()
        total = 0.0
        n = 0
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                recon, mu, logvar = vae(imgs)
                recon_loss = F.l1_loss(recon, imgs)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + v["kl_weight"] * kl
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * imgs.size(0)
            n += imgs.size(0)
        avg = total / max(n, 1)
        log.info(f"[vae epoch {epoch:03d}] loss={avg:.4f}")
        if avg < best:
            best = avg
            torch.save({"state_dict": vae.state_dict(), "cfg": cfg}, ckpt_path)
    return ckpt_path


# ---------- LDM UNet ----------

def build_unet(cfg: dict, num_classes: int) -> DiffusionModelUNet:
    m = cfg["ldm"]
    return DiffusionModelUNet(
        spatial_dims=2,
        in_channels=m["latent_channels"],
        out_channels=m["latent_channels"],
        num_channels=m["channels"],
        attention_levels=m["attention_levels"],
        num_res_blocks=m["num_res_blocks"],
        num_head_channels=m["num_head_channels"],
        with_conditioning=True,
        cross_attention_dim=num_classes,
    )


def train_ldm(cfg: dict, train_df: pd.DataFrame, out_dir: Path) -> Path:
    log = get_logger()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = cfg["ldm"]
    classes = cfg["classes"]
    # VAE (frozen)
    vae = build_vae({"vae": {**cfg["vae"]}}).to(device).eval()
    vae_ckpt = torch.load(m["vae_ckpt"], map_location=device)
    vae.load_state_dict(vae_ckpt["state_dict"])
    for p in vae.parameters():
        p.requires_grad_(False)

    ds = SpineXRDataset(train_df, classes, build_transform("val", m["image_size"]))
    loader = DataLoader(ds, batch_size=m["batch_size"], shuffle=True,
                        num_workers=cfg["project"]["num_workers"], pin_memory=True, drop_last=True)

    unet = build_unet(cfg, num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(unet.parameters(), lr=m["lr"])
    scheduler = DDPMScheduler(
        num_train_timesteps=m["num_train_timesteps"],
        schedule=m["schedule"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    step = 0
    best = float("inf")
    ckpt_path = out_dir / "best.pth"
    while step < m["max_steps"]:
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                latents = vae.encode_stage_2_inputs(imgs)
            # classifier-free conditioning dropout
            drop_mask = (torch.rand(labels.size(0), device=device) < m["classifier_free_prob"])
            cond = labels.clone()
            cond[drop_mask] = 0
            cond = cond.unsqueeze(1)  # (B, 1, C) for cross-attention

            noise = torch.randn_like(latents)
            t = torch.randint(0, scheduler.num_train_timesteps, (latents.size(0),),
                              device=device).long()
            noisy = scheduler.add_noise(original_samples=latents, noise=noise, timesteps=t)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                pred = unet(x=noisy, timesteps=t, context=cond)
                loss = F.mse_loss(pred, noise)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            step += 1
            if step % 200 == 0:
                log.info(f"[ldm step {step:06d}] loss={loss.item():.4f}")
            if step % m["val_every"] == 0 or step == m["max_steps"]:
                if loss.item() < best:
                    best = loss.item()
                    torch.save({"state_dict": unet.state_dict(), "cfg": cfg,
                                "step": step}, ckpt_path)
            if step >= m["max_steps"]:
                break
    return ckpt_path


# ---------- Sampling ----------

@torch.no_grad()
def sample_ldm(cfg: dict, unet: nn.Module, vae: AutoencoderKL,
               class_idx: int, n: int, device: str) -> np.ndarray:
    m = cfg["ldm"]
    classes = cfg["classes"]
    scheduler = DDIMScheduler(num_train_timesteps=m["num_train_timesteps"],
                               schedule=m["schedule"])
    scheduler.set_timesteps(num_inference_steps=m["ddim_steps"])

    cond = torch.zeros(1, 1, len(classes), device=device)
    cond[0, 0, class_idx] = 1.0
    uncond = torch.zeros_like(cond)
    cond_batch = cond.repeat(n, 1, 1)
    uncond_batch = uncond.repeat(n, 1, 1)

    shape = (n, m["latent_channels"], m["latent_size"], m["latent_size"])
    x = torch.randn(shape, device=device)

    for t in scheduler.timesteps:
        tt = torch.tensor([t] * n, device=device).long()
        with torch.cuda.amp.autocast(enabled=device == "cuda"):
            e_cond = unet(x=x, timesteps=tt, context=cond_batch)
            e_uncond = unet(x=x, timesteps=tt, context=uncond_batch)
            eps = e_uncond + m["guidance_scale"] * (e_cond - e_uncond)
        x = scheduler.step(eps, t, x).prev_sample

    imgs = vae.decode_stage_2_outputs(x)
    imgs = (imgs.clamp(-1, 1) + 1) * 127.5
    imgs = imgs.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
    return imgs
