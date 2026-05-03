"""
baselines/run_traditional_ml.py
================================
Traditional ML baselines for clinical risk prediction on MIMIC-IV.

Two feature representations:
  1. Bag-of-codes (BOC)  — binary vector over the ICD vocabulary.
     Each patient's visit history becomes a sparse binary vector:
     1 if a code appeared at least once, 0 otherwise.
     Captures presence/absence but ignores visit order and frequency.

  2. Mean-SBERT          — average the pre-computed SBERT embeddings
     over all codes in a patient's history → dense 384-dim vector.
     This is a direct ablation of the Transformer component: same
     frozen code representations as Sent-e-Med, but no pretraining
     and no attention across visits.  If Sent-e-Med beats this, it
     validates the value of EHR-specific pretraining.

Two classifiers per representation:
  - Logistic Regression  (linear, interpretable)
  - Random Forest        (non-linear, handles interactions)

Evaluation mirrors Sent-e-Med exactly:
  - 5 independent runs with different train/val/test splits
  - ROC-AUC and PR-AUC reported as mean ± std
  - Results saved as {condition}_{baseline}_results.pkl

Usage:
    python baselines/run_traditional_ml.py --mimic_dir data/mimic-iv

    # Skip SBERT features (faster, no embedding cache needed)
    python baselines/run_traditional_ml.py --mimic_dir data/mimic-iv --no_sbert

    # Single condition
    python baselines/run_traditional_ml.py --mimic_dir data/mimic-iv --conditions SUD
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# ── Allow imports from parent directory ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from sent_e_med.data_processing import (
    load_mimic_data,
    build_patient_sequences,
    build_vocabulary,
    split_patients,
    create_condition_dataset,
    split_finetune_samples,
)
from sent_e_med.icd_utils import load_icd_descriptions


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_bag_of_codes(
    samples: List[dict],
    code_to_idx: Dict[Tuple[str, int], int],
    vocab_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binary bag-of-codes feature matrix.

    Each row is a patient; each column is a vocab code.
    Value = 1 if the code appeared anywhere in the patient's input visits.

    Returns:
        X: (n_samples, vocab_size) float32 sparse-ish array
        y: (n_samples,) int labels
    """
    X = np.zeros((len(samples), vocab_size), dtype=np.float32)
    y = np.zeros(len(samples), dtype=np.int32)

    for i, sample in enumerate(samples):
        for visit in sample["visits"]:
            for code, version in visit["codes"]:
                idx = code_to_idx.get((code, version))
                if idx is not None:
                    X[i, idx] = 1.0
        y[i] = sample["label"]

    return X, y


def extract_sbert_features(
    samples: List[dict],
    code_to_idx: Dict[Tuple[str, int], int],
    sbert_embeddings: np.ndarray,   # (vocab_size, 384)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean-pooled SBERT feature matrix.

    Each patient's history is represented as the average of the SBERT
    embeddings of all ICD codes across all their input visits.
    This removes visit structure and Transformer context, serving as
    a direct ablation of the pretraining + attention mechanism.

    Returns:
        X: (n_samples, 384) float32 array
        y: (n_samples,) int labels
    """
    emb_dim = sbert_embeddings.shape[1]
    X = np.zeros((len(samples), emb_dim), dtype=np.float32)
    y = np.zeros(len(samples), dtype=np.int32)

    for i, sample in enumerate(samples):
        code_vecs = []
        for visit in sample["visits"]:
            for code, version in visit["codes"]:
                idx = code_to_idx.get((code, version))
                if idx is not None:
                    code_vecs.append(sbert_embeddings[idx])
        if code_vecs:
            X[i] = np.mean(code_vecs, axis=0)
        y[i] = sample["label"]

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Single run: train + evaluate
# ─────────────────────────────────────────────────────────────────────────────

def _run_once(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    classifier: str,
    seed: int,
) -> Dict[str, float]:
    """Train one classifier and return ROC-AUC and PR-AUC on the test set."""

    if classifier == "lr":
        # Scale features for LR (important for BOC high-dim sparse features)
        scaler  = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        model = LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",   # handles case/control imbalance
            C=1.0,
            solver="lbfgs",
        )
    else:  # "rf"
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc":  average_precision_score(y_test, probs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-run evaluation (mirrors finetune.py interface)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_baseline(
    samples:     List[dict],
    code_to_idx: Dict[Tuple[str, int], int],
    vocab_size:  int,
    feature:     str,            # "boc" or "sbert"
    classifier:  str,            # "lr" or "rf"
    sbert_embeddings: np.ndarray = None,
    n_runs:      int = 5,
    seed:        int = 42,
) -> Dict:
    """
    Run n_runs independent experiments and return mean ± std metrics.

    Each run uses a different stratified train/val/test split to match
    the evaluation protocol in Sent-e-Med (Section 3.4).
    """
    roc_aucs, pr_aucs = [], []

    for run in range(n_runs):
        run_seed = seed + run

        train_s, val_s, test_s = split_finetune_samples(
            samples, seed=run_seed
        )
        # Merge train + val for traditional ML (no early stopping needed)
        train_all = train_s + val_s

        if feature == "boc":
            X_tr, y_tr = extract_bag_of_codes(train_all, code_to_idx, vocab_size)
            X_te, y_te = extract_bag_of_codes(test_s,    code_to_idx, vocab_size)
        else:  # "sbert"
            X_tr, y_tr = extract_sbert_features(train_all, code_to_idx, sbert_embeddings)
            X_te, y_te = extract_sbert_features(test_s,    code_to_idx, sbert_embeddings)

        metrics = _run_once(X_tr, y_tr, X_te, y_te, classifier, seed=run_seed)
        roc_aucs.append(metrics["roc_auc"])
        pr_aucs.append(metrics["pr_auc"])
        print(
            f"  Run {run+1}/{n_runs}  "
            f"ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
        )

    results = {
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std":  float(np.std(roc_aucs)),
        "pr_auc_mean":  float(np.mean(pr_aucs)),
        "pr_auc_std":   float(np.std(pr_aucs)),
        "roc_aucs":     roc_aucs,
        "pr_aucs":      pr_aucs,
    }
    print(
        f"  → ROC-AUC: {results['roc_auc_mean']:.4f}±{results['roc_auc_std']:.4f}  "
        f"PR-AUC: {results['pr_auc_mean']:.4f}±{results['pr_auc_std']:.4f}"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Traditional ML baselines (LR + RF) for Sent-e-Med comparison."
    )
    p.add_argument("--mimic_dir",  type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument(
        "--conditions", nargs="+", default=["SUD", "OUD", "Diabetes"],
        choices=["SUD", "OUD", "Diabetes"],
    )
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument(
        "--no_sbert", action="store_true",
        help="Skip mean-SBERT features (faster; use if no embedding cache exists).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading MIMIC-IV data …")
    admissions, diagnoses = load_mimic_data(args.mimic_dir)
    patient_visits = build_patient_sequences(admissions, diagnoses, min_visits=2)
    descriptions   = load_icd_descriptions(args.mimic_dir)
    code_to_idx, idx_to_code = build_vocabulary(patient_visits)
    vocab_size = len(idx_to_code)

    all_ids = list(patient_visits.keys())
    _, val_ids, test_ids = split_patients(all_ids, seed=args.seed)
    finetune_ids = val_ids + test_ids

    # ── Optionally load SBERT embeddings from existing cache ──────────────────
    sbert_embeddings = None
    if not args.no_sbert:
        sbert_cache = Path(args.output_dir) / "sbert_embeddings.pt"
        if sbert_cache.exists():
            import torch
            sbert_embeddings = torch.load(sbert_cache, map_location="cpu").numpy()
            print(f"Loaded SBERT embeddings: {sbert_embeddings.shape}")
        else:
            print(
                f"WARNING: SBERT cache not found at {sbert_cache}. "
                "Skipping mean-SBERT features. Run main.py first or use --no_sbert."
            )

    # ── Baselines to run ──────────────────────────────────────────────────────
    baselines = [
        ("boc",   "lr",  "boc_lr"),
        ("boc",   "rf",  "boc_rf"),
    ]
    if sbert_embeddings is not None:
        baselines += [
            ("sbert", "lr", "sbert_mean_lr"),
            ("sbert", "rf", "sbert_mean_rf"),
        ]

    # ── Run for each condition × baseline ─────────────────────────────────────
    all_results: Dict = {}

    for condition in args.conditions:
        print(f"\n{'='*65}")
        print(f"Condition: {condition}")
        print(f"{'='*65}")

        samples = create_condition_dataset(
            patient_visits=patient_visits,
            patient_ids=finetune_ids,
            condition=condition,
            seed=args.seed,
        )
        if len(samples) < 50:
            print(f"Insufficient samples ({len(samples)}). Skipping.")
            continue

        all_results[condition] = {}

        for feature, clf, tag in baselines:
            print(f"\n── {tag} ──")
            results = evaluate_baseline(
                samples=samples,
                code_to_idx=code_to_idx,
                vocab_size=vocab_size,
                feature=feature,
                classifier=clf,
                sbert_embeddings=sbert_embeddings,
                n_runs=args.n_runs,
                seed=args.seed,
            )
            all_results[condition][tag] = results

            out_path = Path(args.output_dir) / f"{condition}_{tag}_results.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(results, f)
            print(f"Saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY — Traditional ML Baselines")
    print(f"{'='*65}")
    print(f"{'Baseline':<20} {'Condition':<12} {'ROC-AUC':>12} {'PR-AUC':>12}")
    print("─" * 60)
    for condition, res_dict in all_results.items():
        for tag, res in res_dict.items():
            print(
                f"{tag:<20} {condition:<12} "
                f"{res['roc_auc_mean']:.4f}±{res['roc_auc_std']:.4f}  "
                f"{res['pr_auc_mean']:.4f}±{res['pr_auc_std']:.4f}"
            )


if __name__ == "__main__":
    main()
