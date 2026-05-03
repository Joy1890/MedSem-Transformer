"""
main.py
=======
Entry point for training and evaluating Sent-e-Med on MIMIC-IV.

Usage:
    # SBERT encoder + code-level masking
    python main.py --mimic_dir data/mimic-iv

    # Variant 1: Bio_ClinicalBERT encoder (768-dim → projected to 384)
    python main.py --mimic_dir data/mimic-iv --encoder bio_clinical_bert

    # Variant 2: Visit-level masking
    python main.py --mimic_dir data/mimic-iv --masking visit

    # Both variants combined
    python main.py --mimic_dir data/mimic-iv --encoder bio_clinical_bert --masking visit

    # MLM-only (no Next Visit Prediction)
    python main.py --mimic_dir data/mimic-iv --mlm_only

    # Skip pretraining, load existing weights
    python main.py --mimic_dir data/mimic-iv --skip_pretrain

    # Fine-tune specific conditions only
    python main.py --mimic_dir data/mimic-iv --skip_pretrain --conditions SUD Diabetes

    # Specify GPU
    python main.py --mimic_dir data/mimic-iv --device cuda:0

Expected MIMIC-IV directory layout:
    data/mimic-iv/
        hosp/
            admissions.csv
            diagnoses_icd.csv
            d_icd_diagnoses.csv
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from sent_e_med.config import SenteMedConfig
from sent_e_med.enrich_icd import load_or_build_enriched
from sent_e_med.phecode_utils import (
    build_phecode_embeddings,
    extend_phecode_embeddings,
    load_phecode_map,
)
from sent_e_med.data_processing import (
    build_patient_sequences,
    build_vocabulary,
    create_condition_dataset,
    extend_vocabulary,
    load_mimic_data,
    split_finetune_samples,
    split_patients,
)
from sent_e_med.dataset import PretrainDataset
from sent_e_med.finetune import finetune
from sent_e_med.icd_utils import load_icd_descriptions
from sent_e_med.model import SenteMed
from sent_e_med.pretrain import load_pretrained, pretrain
from sent_e_med.sbert_utils import build_code_embeddings, extend_embeddings, update_model_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate Sent-e-Med on MIMIC-IV EHR data."
    )
    p.add_argument(
        "--mimic_dir", type=str, required=True,
        help="Path to MIMIC-IV directory (must contain hosp/admissions.csv etc.)"
    )
    p.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory for saving models, embeddings, and results."
    )
    p.add_argument(
        "--skip_pretrain", action="store_true",
        help="Skip pretraining; load weights from output_dir/pretrained_sent_e_med_<variant>.pt"
    )
    p.add_argument(
        "--mlm_only", action="store_true",
        help="Use MLM-only pretraining (disable Next Visit Prediction objective). "
             "The paper notes this variant performed slightly better overall."
    )
    p.add_argument(
        "--conditions", nargs="+", default=["SUD", "OUD", "Diabetes"],
        choices=["SUD", "OUD", "Diabetes"],
        help="Which conditions to fine-tune for."
    )
    p.add_argument(
        "--encoder", type=str, default="sbert",
        choices=["sbert", "bio_clinical_bert"],
        help=(
            "Code encoder type. "
            "'sbert' (default) = all-MiniLM-L6-v2, 384-dim (original paper). "
            "'bio_clinical_bert' = emilyalsentzer/Bio_ClinicalBERT, 768→384-dim."
        ),
    )
    p.add_argument(
        "--masking", type=str, default="code",
        choices=["code", "visit"],
        help=(
            "Pretraining masking strategy. "
            "'code' (default) = mask individual ICD codes (15%%, original paper). "
            "'visit' = mask entire visits (15%% of visits, all their codes)."
        ),
    )
    p.add_argument(
        "--enrich", action="store_true",
        help=(
            "Replace MIMIC ICD descriptions with GPT-enriched clinical summaries "
            "before SBERT encoding. Requires OPENAI_API_KEY env variable. "
            "Enriched texts are cached to output_dir/enriched_descriptions.json "
            "after the first run — no API calls on subsequent runs."
        ),
    )
    p.add_argument(
        "--openai_model", type=str, default="gpt-4o-mini",
        help="OpenAI model for ICD description enrichment (default: gpt-4o-mini).",
    )
    p.add_argument(
        "--phecode", action="store_true",
        help=(
            "Enable PheCode dual-embedding: each ICD token is represented by "
            "the fusion of its ICD description embedding and its coarse PheCode "
            "phenotype embedding. Requires --phecode_map."
        ),
    )
    p.add_argument(
        "--phecode_map", type=str, default=None,
        help="Path to phecodes_cm_rolled.csv (required when --phecode is set).",
    )
    p.add_argument(
        "--phecode_fusion", type=str, default="concat",
        choices=["concat", "gate"],
        help=(
            "How to fuse ICD and PheCode embeddings. "
            "'concat' (default) = Linear(2H→H) on their concatenation. "
            "'gate' = learned sigmoid gate: gate*icd + (1-gate)*phecode."
        ),
    )
    p.add_argument(
        "--n_runs", type=int, default=5,
        help="Number of independent fine-tuning runs (paper uses 5)."
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda', 'cuda:0', 'cpu'. Auto-detected if not set."
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ── Config ────────────────────────────────────────────────────────────────
    # ── Validate PheCode args ─────────────────────────────────────────────────
    if args.phecode and not args.phecode_map:
        raise ValueError(
            "--phecode requires --phecode_map pointing to phecodes_cm_rolled.csv"
        )

    config = SenteMedConfig(
        mimic_dir=args.mimic_dir,
        output_dir=args.output_dir,
        use_nvp=not args.mlm_only,
        encoder_type=args.encoder,
        masking_strategy=args.masking,
        use_phecode=args.phecode,
        phecode_fusion=args.phecode_fusion,
        phecode_map_path=args.phecode_map or "",
        seed=args.seed,
        n_eval_runs=args.n_runs,
    )
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Variant tag used in filenames so checkpoints don't collide across variants.
    # Includes encoder, masking strategy, NVP flag, and phecode fusion mode.
    obj_tag    = "nvpmlm" if not args.mlm_only else "mlm"
    ph_tag     = f"_phecode{args.phecode_fusion}" if args.phecode else ""
    enrich_tag = "_enriched" if args.enrich else ""
    variant_tag = f"{args.encoder}_{obj_tag}_mask{args.masking}{ph_tag}{enrich_tag}"
    print(f"Variant: encoder={args.encoder}  masking={args.masking}")
    if not args.mlm_only:
        print("         + Next Visit Prediction objective")
    if args.phecode:
        print(f"         + PheCode dual-embedding (fusion={args.phecode_fusion})")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Load MIMIC-IV tables
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 1: Loading MIMIC-IV data")
    print("="*65)
    admissions, diagnoses = load_mimic_data(config.mimic_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Build patient visit sequences
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 2: Building patient visit sequences")
    print("="*65)
    patient_visits = build_patient_sequences(
        admissions, diagnoses, min_visits=config.min_visits
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Load ICD descriptions (for SBERT encoding)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 3: Loading ICD code descriptions")
    print("="*65)
    descriptions = load_icd_descriptions(config.mimic_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Build vocabulary & patient splits
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 4: Building vocabulary and splitting patients")
    print("="*65)
    code_to_idx, idx_to_code = build_vocabulary(patient_visits)

    all_patient_ids = list(patient_visits.keys())
    pretrain_ids, val_ids, test_ids = split_patients(
        all_patient_ids,
        pretrain_split=config.pretrain_split,
        val_split=config.val_split,
        test_split=config.test_split,
        seed=config.seed,
    )

    # Save vocabulary for reproducibility
    vocab_path = Path(config.output_dir) / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump({"code_to_idx": code_to_idx, "idx_to_code": idx_to_code}, f)
    print(f"Vocabulary saved to {vocab_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4b: Optionally replace ICD descriptions with GPT-enriched versions
    # ─────────────────────────────────────────────────────────────────────────
    if args.enrich:
        print("\n" + "="*65)
        print(f"STEP 4b: Enriching ICD descriptions with {args.openai_model}")
        print("="*65)
        enrich_cache = str(Path(config.output_dir) / "enriched_descriptions.json")
        descriptions = load_or_build_enriched(
            descriptions=descriptions,
            cache_path=enrich_cache,
            model=args.openai_model,
        )
        # descriptions is now a drop-in replacement — all downstream code
        # (build_code_embeddings, phecode fallback, OOV extension) unchanged

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Compute frozen code embeddings
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"STEP 5: Building code embeddings ({args.encoder})")
    print("="*65)
    # Each encoder type gets its own cache file to avoid collisions
    code_embeddings = build_code_embeddings(
        config=config,
        idx_to_code=idx_to_code,
        descriptions=descriptions,
        cache_dir=config.output_dir,
    )
    encoder_dim = code_embeddings.shape[1]
    print(f"  encoder_dim = {encoder_dim}  →  hidden_dim = {config.hidden_dim}")
    if encoder_dim != config.hidden_dim:
        print(f"  (input_projection Linear({encoder_dim}→{config.hidden_dim}) will be learned)")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5b: Compute PheCode embeddings (optional)
    # ─────────────────────────────────────────────────────────────────────────
    phecode_map        = None
    phecode_embeddings = None

    if args.phecode:
        print("\n" + "="*65)
        print(f"STEP 5b: Building PheCode embeddings (fusion={args.phecode_fusion})")
        print("="*65)
        phecode_map = load_phecode_map(args.phecode_map)

        # Cache key includes the SBERT model name so it's safe to change models
        phecode_cache = str(
            Path(config.output_dir) / f"phecode_embeddings_{config.sbert_model_name}.pt"
        )
        phecode_embeddings = build_phecode_embeddings(
            idx_to_code=idx_to_code,
            phecode_map=phecode_map,
            icd_descriptions=descriptions,
            sbert_model_name=config.sbert_model_name,
            cache_path=phecode_cache,
        )
        print(f"  PheCode embedding shape: {tuple(phecode_embeddings.shape)}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Build model
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 6: Building Sent-e-Med model")
    print("="*65)
    vocab_size = len(idx_to_code)
    model = SenteMed(
        config,
        vocab_size=vocab_size,
        sbert_embeddings=code_embeddings,
        phecode_embeddings=phecode_embeddings,   # None when --phecode not set
    )

    params = model.count_parameters()
    print(f"Trainable parameters      : {params['trainable_params']:,}")
    print(f"Frozen encoder buffer     : {params['sbert_buffer_elements']:,} elements")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Pretrain (or load pretrained weights)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 7: Pretraining")
    print("="*65)

    # Checkpoint name encodes variant so different runs don't overwrite each other
    checkpoint_name = f"pretrained_sent_e_med_{variant_tag}.pt"

    if args.skip_pretrain:
        pretrained_path = Path(config.output_dir) / checkpoint_name
        if pretrained_path.exists():
            model = load_pretrained(model, str(pretrained_path), device=device)
        else:
            print(
                f"WARNING: --skip_pretrain set but {pretrained_path} not found. "
                "Training from scratch."
            )
    else:
        # Pretraining uses ~70% of MIMIC (pretrain_ids split)
        pretrain_dataset = PretrainDataset(
            patient_visits=patient_visits,
            patient_ids=pretrain_ids,
            code_to_idx=code_to_idx,
            vocab_size=vocab_size,
            max_seq_len=config.max_seq_len,
            max_visits=config.max_visits,
            mlm_probability=config.mlm_probability,
            mlm_mask_token_prob=config.mlm_mask_token_prob,
            mlm_random_token_prob=config.mlm_random_token_prob,
            use_nvp=config.use_nvp,
            masking_strategy=config.masking_strategy,
            visit_mask_probability=config.visit_mask_probability,
            seed=config.seed,
        )
        # Use a subset of val_ids as pretraining validation
        val_pretrain_dataset = PretrainDataset(
            patient_visits=patient_visits,
            patient_ids=val_ids[:min(5000, len(val_ids))],
            code_to_idx=code_to_idx,
            vocab_size=vocab_size,
            max_seq_len=config.max_seq_len,
            max_visits=config.max_visits,
            mlm_probability=config.mlm_probability,
            mlm_mask_token_prob=config.mlm_mask_token_prob,
            mlm_random_token_prob=config.mlm_random_token_prob,
            use_nvp=config.use_nvp,
            masking_strategy=config.masking_strategy,
            visit_mask_probability=config.visit_mask_probability,
            seed=config.seed + 1,
        )

        print(
            f"Pretraining on {len(pretrain_dataset):,} patients | "
            f"Validating on {len(val_pretrain_dataset):,}\n"
            f"Objective: MLM{'+ NVP' if config.use_nvp else ' only'}  "
            f"| Masking: {config.masking_strategy}-level"
        )
        model = pretrain(
            model, config, pretrain_dataset, val_pretrain_dataset, device,
            checkpoint_name=checkpoint_name,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8: Fine-tune on each condition
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 8: Fine-tuning")
    print("="*65)

    # Fine-tuning uses val+test patients (no overlap with pretrain patients)
    finetune_patient_ids = val_ids + test_ids
    all_results = {}

    for condition in args.conditions:
        # Create case/control samples
        samples = create_condition_dataset(
            patient_visits=patient_visits,
            patient_ids=finetune_patient_ids,
            condition=condition,
            seed=config.seed,
        )

        if len(samples) < 50:
            print(f"Insufficient samples for {condition} ({len(samples)}). Skipping.")
            continue

        # Check for OOV codes in fine-tuning data (from new vocabulary entries)
        ft_visits = {s["patient_id"]: s["visits"] for s in samples}
        code_to_idx, idx_to_code, new_codes = extend_vocabulary(
            code_to_idx, idx_to_code, ft_visits
        )
        if new_codes:
            # Compute encoder embeddings for OOV codes and extend the ICD table
            extended_icd = extend_embeddings(
                current_embeddings=model.sbert_code_embeddings.cpu(),
                new_codes=new_codes,
                descriptions=descriptions,
                encoder_type=config.encoder_type,
                sbert_model_name=config.sbert_model_name,
                bio_clinical_bert_model=config.bio_clinical_bert_model,
            )

            # Extend the PheCode table in parallel if phecode mode is active
            extended_phecode = None
            if args.phecode and phecode_map is not None:
                extended_phecode = extend_phecode_embeddings(
                    current_phecode_embeddings=model.phecode_code_embeddings.cpu(),
                    new_codes=new_codes,
                    phecode_map=phecode_map,
                    icd_descriptions=descriptions,
                    sbert_model_name=config.sbert_model_name,
                )

            update_model_embeddings(model, extended_icd, extended_phecode)

        # Stratified train/val/test split of fine-tuning samples
        train_samples, val_samples, test_samples = split_finetune_samples(
            samples, seed=config.seed
        )

        # Run fine-tuning
        results = finetune(
            model=model,
            config=config,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            code_to_idx=code_to_idx,
            condition=condition,
            device=device,
            n_runs=args.n_runs,
        )
        all_results[condition] = results

        # Save per-condition results — filename includes variant tag for comparison
        result_path = Path(config.output_dir) / f"{condition}_{variant_tag}_results.pkl"
        with open(result_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {result_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    print(f"{'Condition':<12} {'ROC-AUC':>12} {'PR-AUC':>12}")
    print("─"*40)
    for cond, res in all_results.items():
        print(
            f"{cond:<12} "
            f"{res['roc_auc_mean']:.4f}±{res['roc_auc_std']:.4f}  "
            f"{res['pr_auc_mean']:.4f}±{res['pr_auc_std']:.4f}"
        )


if __name__ == "__main__":
    main()
