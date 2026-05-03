"""
config.py
=========
Implementation details 
  hidden_dim=384, 4 layers, 4 heads,
  linear_dim=64, max_seq_len=128, lr=1e-5, AdamW)

Variants supported:
  encoder_type = "sbert"              — all-MiniLM-L6-v2, 384-dim
  encoder_type = "bio_clinical_bert"  — Bio_ClinicalBERT (768-dim, projected to hidden_dim)

  masking_strategy = "code"   — mask individual ICD codes
  masking_strategy = "visit"  — variant: mask entire visits at once

  use_phecode = True           — enable PheCode dual-embedding (extension)
  phecode_fusion = "concat"   — fusion method: "concat" (Linear) or "gate"
"""

from dataclasses import dataclass, field


@dataclass
class SenteMedConfig:
    # ── Encoder type ────────────────────────────────────────────────────────
    # "sbert"             :  all-MiniLM-L6-v2
    # "bio_clinical_bert" : variant — emilyalsentzer/Bio_ClinicalBERT (CLS token)
    encoder_type: str = "sbert"

    # ── SBERT (used when encoder_type="sbert") ───────────────────────────────
    # "all-MiniLM-L6-v2" produces 384-dim embeddings, matching hidden_dim.
    sbert_model_name: str = "all-MiniLM-L6-v2"

    # ── Bio_ClinicalBERT (used when encoder_type="bio_clinical_bert") ────────
    # Pre-trained on MIMIC-III clinical notes — strong clinical domain knowledge.
    # Outputs 768-dim CLS token embeddings, projected down to hidden_dim.
    bio_clinical_bert_model: str = "emilyalsentzer/Bio_ClinicalBERT"

    # ── Transformer (Section 7.2) ────────────────────────────────────────────
    # 4 hidden layers, 4 attention heads, and a hidden dimension of 384
    # For Bio_ClinicalBERT variant, keep hidden_dim=384 and project 768→384.
    hidden_dim: int = 384
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 1536        # standard 4 × hidden_dim
    dropout: float = 0.1

    # ── Masking strategy ─────────────────────────────────────────────────────
    # "code"  : randomly mask individual ICD codes (15%)
    # "visit" : variant — randomly mask entire visits (15% of visits);
    #           all codes within a masked visit are replaced with [MASK]
    masking_strategy: str = "code"

    # Fraction of VISITS to mask when masking_strategy="visit"
    visit_mask_probability: float = 0.15

    # ── Sequence limits ──────────────────────────────────────────────────────
    # "The maximum sequence length of 128 tokens (sentences) was used"
    max_seq_len: int = 128     # total flattened ICD codes per patient
    max_visits: int = 100      # maximum distinct visit positions (visit embeddings)

    # ── Classification head ──────────────────────────────────────────────────
    # "The dimension of the linear layer was set to 64"
    linear_dim: int = 64

    # ── MLM masking (standard BERT recipe) ──────────────────────────────────
    # "involves masking 15% of the medical codes"
    # "80% → [MASK], 10% → random code, 10% → unchanged"
    mlm_probability: float = 0.15
    mlm_mask_token_prob: float = 0.80
    mlm_random_token_prob: float = 0.10
    # remaining 0.10 → keep original

    # ── Pretraining objectives ───────────────────────────────────────────────
    # Set use_nvp=False for the MLM-only variant (Section 4.2.2).
    # "the variant that only used the MLM training objective performed
    #  slightly better overall" — so MLM-only is the recommended variant.
    use_nvp: bool = True       # include Next Visit Prediction objective

    # ── Pretraining hyperparameters ──────────────────────────────────────────
    # "AdamWeight decay optimizer was used with a learning rate of 1e-5"
    pretrain_lr: float = 1e-5
    pretrain_weight_decay: float = 0.01
    pretrain_batch_size: int = 32
    pretrain_epochs: int = 50
    pretrain_grad_clip: float = 1.0

    # ── Fine-tuning hyperparameters ──────────────────────────────────────────
    # "the finetuning took only around 20 minutes" on a V100
    finetune_lr: float = 1e-5
    finetune_weight_decay: float = 0.01
    finetune_batch_size: int = 32
    finetune_epochs: int = 20
    finetune_grad_clip: float = 1.0

    # ── Data ─────────────────────────────────────────────────────────────────
    # "data is partitioned into training, test, and validation sets in a
    #  ratio of 7:2:1" — pretraining split
    pretrain_split: float = 0.70
    val_split: float = 0.20
    test_split: float = 0.10
    min_visits: int = 2         # minimum visits per patient to include

    # ── Evaluation ───────────────────────────────────────────────────────────
    # "average performances across 5 different runs" (Table 4)
    n_eval_runs: int = 5

    # ── PheCode dual-embedding (optional extension) ──────────────────────────
    # When enabled, each ICD token is represented by the fusion of:
    #   (a) its fine-grained ICD description embedding (via encoder_type)
    #   (b) its coarse-grained PheCode phenotype description (always SBERT)
    # Two fusion methods:
    #   "concat" — concat([icd_H, phecode_H]) → Linear(2H → H)  [default]
    #   "gate"   — gate * icd_H + (1-gate) * phecode_H,
    #              gate = sigmoid(Linear(2H → H)) initialised near 0 (→ 0.5 mix)
    use_phecode: bool = False
    phecode_fusion: str = "concat"   # "concat" or "gate"
    phecode_map_path: str = ""       # path to phecodes_cm_rolled.csv

    # ── Paths ─────────────────────────────────────────────────────────────────
    mimic_dir: str = "data/mimic-iv"
    output_dir: str = "outputs"

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42
