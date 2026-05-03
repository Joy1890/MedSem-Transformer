"""
finetune.py
===========
Fine-tuning and evaluation for Sent-e-Med binary risk prediction.

Setup matches Section 5.1 and Section 7.2 of the paper:
  - Metrics: ROC AUC and Precision-Recall AUC (PR AUC)
  - 5 independent runs, report mean ± std  (Table 4)
  - Loss: binary cross-entropy
  - Imbalanced data → WeightedRandomSampler to balance batches
  - Best checkpoint per run selected by validation ROC AUC
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import FinetuneDataset


# ─────────────────────────────────────────────────────────────────────────────
# Sampler for imbalanced datasets
# ─────────────────────────────────────────────────────────────────────────────

def make_balanced_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler that up-samples the minority class.

    Positive (case) patients are typically ~10% of the dataset (Table 2),
    so we up-sample them to create balanced mini-batches.
    """
    labels_arr = np.array(labels, dtype=float)
    n_pos = labels_arr.sum()
    n_neg = len(labels_arr) - n_pos

    # Avoid division by zero
    w_pos = len(labels_arr) / max(n_pos, 1)
    w_neg = len(labels_arr) / max(n_neg, 1)

    sample_weights = np.where(labels_arr == 1, w_pos, w_neg)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(labels_arr),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def _single_finetune_run(
    model: nn.Module,
    config,
    train_samples: List[dict],
    val_samples:   List[dict],
    test_samples:  List[dict],
    code_to_idx:   Dict,
    condition:     str,
    device:        str,
    run_idx:       int,
) -> Dict[str, float]:
    """
    One complete fine-tuning run: train → best checkpoint → test evaluation.
    """
    # Fresh copy of pretrained model for each run (isolated weights)
    run_model = copy.deepcopy(model).to(device)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = FinetuneDataset(train_samples, code_to_idx,
                               config.max_seq_len, config.max_visits)
    val_ds   = FinetuneDataset(val_samples,   code_to_idx,
                               config.max_seq_len, config.max_visits)
    test_ds  = FinetuneDataset(test_samples,  code_to_idx,
                               config.max_seq_len, config.max_visits)

    train_labels = [s["label"] for s in train_samples]
    sampler      = make_balanced_sampler(train_labels)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=config.finetune_batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=config.finetune_batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=(device == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=config.finetune_batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=(device == "cuda"))

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = AdamW(
        run_model.parameters(),
        lr=config.finetune_lr,
        weight_decay=config.finetune_weight_decay,
        betas=(0.9, 0.999),
    )

    best_val_auc = 0.0
    best_state   = None

    # ── Training epochs ────────────────────────────────────────────────────────
    for epoch in range(config.finetune_epochs):
        run_model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            code_ids       = batch["code_ids"].to(device)
            visit_ids      = batch["visit_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            logits = run_model.finetune_step(code_ids, visit_ids, attention_mask)
            loss   = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                run_model.parameters(), max_norm=config.finetune_grad_clip
            )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        # Validate after each epoch
        val_metrics = _evaluate(run_model, val_loader, device)

        print(
            f"    Run {run_idx+1} | Epoch {epoch+1:2d}/{config.finetune_epochs} | "
            f"Loss={epoch_loss/n_batches:.4f} | "
            f"Val ROC-AUC={val_metrics['roc_auc']:.4f} | "
            f"Val PR-AUC={val_metrics['pr_auc']:.4f}"
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_state   = copy.deepcopy(run_model.state_dict())

    # ── Test with best checkpoint ─────────────────────────────────────────────
    if best_state is not None:
        run_model.load_state_dict(best_state)
    test_metrics = _evaluate(run_model, test_loader, device)

    print(
        f"  → Run {run_idx+1} TEST | "
        f"ROC-AUC={test_metrics['roc_auc']:.4f} | "
        f"PR-AUC={test_metrics['pr_auc']:.4f}"
    )
    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    """
    Compute ROC AUC and PR AUC on a DataLoader.

    "our main emphasis was on evaluating outputs in the form of probability
     scores … we selected Area under the ROC Curve (ROC AUC) and
     Precision-Recall AUC (PR AUC) as our preferred evaluation metrics."
    — Section 5.1
    """
    model.eval()
    all_probs:  List[float] = []
    all_labels: List[float] = []

    with torch.no_grad():
        for batch in loader:
            code_ids       = batch["code_ids"].to(device)
            visit_ids      = batch["visit_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].cpu().numpy()

            logits = model.finetune_step(code_ids, visit_ids, attention_mask)
            probs  = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Guard against single-class edge case (e.g., very small batches)
    if len(np.unique(all_labels)) < 2:
        return {"roc_auc": 0.5, "pr_auc": float(all_labels.mean())}

    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc  = average_precision_score(all_labels, all_probs)

    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


# ─────────────────────────────────────────────────────────────────────────────
# Multi-run fine-tuning (as per Table 4: "average over 5 runs")
# ─────────────────────────────────────────────────────────────────────────────

def finetune(
    model: nn.Module,
    config,
    train_samples: List[dict],
    val_samples:   List[dict],
    test_samples:  List[dict],
    code_to_idx:   Dict,
    condition:     str,
    device:        str = "cuda",
    n_runs:        Optional[int] = None,
) -> Dict:
    """
    Run n_runs independent fine-tuning experiments and report mean ± std.

    This mirrors the paper's evaluation protocol (Table 4):
    "The results depict the average performances across 5 different runs."

    Args:
        model:          Pretrained SenteMed (weights are NOT modified in-place;
                        each run uses a deep copy)
        config:         SenteMedConfig
        train_samples:  from create_condition_dataset + split_finetune_samples
        val_samples:    validation split
        test_samples:   test split
        code_to_idx:    vocabulary mapping
        condition:      "SUD" | "OUD" | "Diabetes" (for logging)
        device:         "cuda" | "cpu"
        n_runs:         number of independent runs (default: config.n_eval_runs)

    Returns:
        dict with mean/std ROC-AUC and PR-AUC, plus per-run results
    """
    if n_runs is None:
        n_runs = config.n_eval_runs

    n_cases    = sum(1 for s in train_samples + val_samples + test_samples if s["label"] == 1)
    n_controls = sum(1 for s in train_samples + val_samples + test_samples if s["label"] == 0)
    print(f"\n{'='*65}")
    print(f"Fine-tuning: {condition}  |  cases={n_cases:,}  controls={n_controls:,}")
    print(
        f"  Train={len(train_samples):,}  Val={len(val_samples):,}  "
        f"Test={len(test_samples):,}  Runs={n_runs}"
    )
    print(f"{'='*65}\n")

    all_run_results: List[Dict[str, float]] = []

    for run_idx in range(n_runs):
        print(f"\n── Run {run_idx+1}/{n_runs} ──────────────────────────────")
        metrics = _single_finetune_run(
            model=model,
            config=config,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            code_to_idx=code_to_idx,
            condition=condition,
            device=device,
            run_idx=run_idx,
        )
        all_run_results.append(metrics)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    roc_aucs = [r["roc_auc"] for r in all_run_results]
    pr_aucs  = [r["pr_auc"]  for r in all_run_results]

    summary = {
        "condition":    condition,
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std":  float(np.std(roc_aucs)),
        "pr_auc_mean":  float(np.mean(pr_aucs)),
        "pr_auc_std":   float(np.std(pr_aucs)),
        "all_runs":     all_run_results,
    }

    print(f"\n{'─'*65}")
    print(f"FINAL  {condition}")
    print(
        f"  ROC-AUC : {summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f}"
    )
    print(
        f"  PR-AUC  : {summary['pr_auc_mean']:.4f} ± {summary['pr_auc_std']:.4f}"
    )
    print(f"{'─'*65}\n")

    return summary
