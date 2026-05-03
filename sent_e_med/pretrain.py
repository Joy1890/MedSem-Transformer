"""
pretrain.py
===========
Pretraining loop

Objectives:
  1. MLM  — Masked Language Modeling (always active)
  2. NVP  — Next Visit Prediction    (active when config.use_nvp=True)

Training setup:
  - Optimizer: AdamW with weight decay
  - LR: 1e-5
  - Gradient clipping: max_norm=1.0
  - "Pretraining took around 5 days" on a 32 GB V100

The best model (lowest validation loss) is saved to
    config.output_dir / checkpoint_name
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Main pretraining function
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(
    model: nn.Module,
    config,
    train_dataset,
    val_dataset=None,
    device: str = "cuda",
    checkpoint_name: str = "pretrained_sent_e_med.pt",
) -> nn.Module:
    """
    Run the pretraining loop.

    Args:
        model:            SenteMed instance
        config:           SenteMedConfig instance
        train_dataset:    PretrainDataset for the ~80K pretraining patients
        val_dataset:      Optional PretrainDataset for validation
        device:           "cuda" or "cpu"
        checkpoint_name:  Filename (not full path) to save the best checkpoint.
                          Pass a variant-specific name (e.g.
                          "pretrained_sent_e_med_sbert_maskcode.pt") so that
                          different runs don't overwrite each other.

    Returns:
        model with best (lowest val loss) weights loaded
    """
    model = model.to(device)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.pretrain_batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.pretrain_batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=(device == "cuda"),
        )

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    # Only optimize parameters that require gradients (excludes SBERT buffer)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.pretrain_lr,
        weight_decay=config.pretrain_weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.pretrain_epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_state     = None
    save_path      = Path(config.output_dir) / checkpoint_name

    param_info = model.count_parameters()
    print(
        f"\nTrainable parameters: {param_info['trainable_params']:,}  "
        f"| SBERT buffer (frozen): {param_info['sbert_buffer_elements']:,} elements\n"
    )
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'MLM':>8}  {'NVP':>8}")
    print("─" * 60)

    for epoch in range(config.pretrain_epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_mlm, train_nvp, train_total = [], [], []

        for batch in train_loader:
            code_ids       = batch["code_ids"].to(device)
            visit_ids      = batch["visit_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            is_masked      = batch["is_masked"].to(device)
            mlm_labels     = batch["mlm_labels"].to(device)
            nvp_labels     = batch["nvp_labels"].to(device)

            optimizer.zero_grad()

            losses = model.pretrain_step(
                code_ids=code_ids,
                visit_ids=visit_ids,
                attention_mask=attention_mask,
                is_masked=is_masked,
                mlm_labels=mlm_labels,
                nvp_labels=nvp_labels,
                use_nvp=config.use_nvp,
            )

            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.pretrain_grad_clip
            )
            optimizer.step()

            train_mlm.append(losses["mlm_loss"].item())
            train_nvp.append(losses["nvp_loss"].item())
            train_total.append(losses["total_loss"].item())

        scheduler.step()

        avg_train = np.mean(train_total)
        avg_mlm   = np.mean(train_mlm)
        avg_nvp   = np.mean(train_nvp)

        # ── Validate ──────────────────────────────────────────────────────────
        if val_loader is not None:
            val_loss = _eval_pretrain(model, val_loader, config, device)
            print(
                f"{epoch+1:6d}  {avg_train:12.4f}  {val_loss:10.4f}"
                f"  {avg_mlm:8.4f}  {avg_nvp:8.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": best_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "config": config,
                    },
                    save_path,
                )
                print(f"         ✓ New best model saved  (val_loss={best_val_loss:.4f})")
        else:
            print(
                f"{epoch+1:6d}  {avg_train:12.4f}  {'—':>10}"
                f"  {avg_mlm:8.4f}  {avg_nvp:8.4f}"
            )
            # Save every epoch when no validation
            torch.save(model.state_dict(), save_path)

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nPretraining complete. Best val loss: {best_val_loss:.4f}")
    else:
        print(f"\nPretraining complete.")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def _eval_pretrain(model: nn.Module, val_loader: DataLoader, config, device: str) -> float:
    """Evaluate pretraining loss on a validation set."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in val_loader:
            code_ids       = batch["code_ids"].to(device)
            visit_ids      = batch["visit_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            is_masked      = batch["is_masked"].to(device)
            mlm_labels     = batch["mlm_labels"].to(device)
            nvp_labels     = batch["nvp_labels"].to(device)

            losses = model.pretrain_step(
                code_ids=code_ids,
                visit_ids=visit_ids,
                attention_mask=attention_mask,
                is_masked=is_masked,
                mlm_labels=mlm_labels,
                nvp_labels=nvp_labels,
                use_nvp=config.use_nvp,
            )
            total_loss += losses["total_loss"].item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Load pretrained weights
# ─────────────────────────────────────────────────────────────────────────────

def load_pretrained(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """
    Load pretrained weights into a SenteMed model.

    Handles both:
        - Full checkpoint dict (with 'model_state_dict' key)
        - Raw state_dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(
            f"Loading checkpoint from epoch {checkpoint.get('epoch', '?')} "
            f"(val_loss={checkpoint.get('val_loss', 'N/A'):.4f})"
        )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    print(f"Loaded pretrained weights from {checkpoint_path}")
    return model
