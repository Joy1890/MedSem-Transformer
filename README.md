# MedSem-Transformer

**Semantically-Grounded EHR Pre-Training for Clinical Risk Prediction on MIMIC-IV**

MedSem-Transformer is a BERT-style pre-training framework for Electronic Health Records (EHR). It grounds ICD diagnosis code tokens in frozen natural-language embeddings (SBERT or BioClinicalBERT), then learns visit-level contextual representations via Masked Language Modelling (MLM) and Next Visit Prediction (NVP). The pre-trained model is fine-tuned for binary clinical risk prediction across three conditions: **SUD**, **OUD**, and **Diabetes**.

---

## Project Structure

```
LLM project code/
├── main.py                        # Main entry point (pre-train + fine-tune)
├── requirements.txt               # Python dependencies
├── arch_diagram.py                # Architecture diagram generator
│
├── sent_e_med/                    # Core model package
│   ├── model.py                   # MedSem-Transformer architecture
│   ├── config.py                  # All hyperparameters
│   ├── pretrain.py                # Pre-training loop (MLM + NVP)
│   ├── finetune.py                # Fine-tuning loop (binary classification)
│   ├── dataset.py                 # Pre-train & fine-tune datasets
│   ├── data_processing.py         # MIMIC-IV loading & patient sequences
│   ├── icd_utils.py               # ICD code → text description mapping
│   ├── sbert_utils.py             # SBERT / BioClinicalBERT embedding builder
│   ├── phecode_utils.py           # PheCode dual-embedding extension
│   └── enrich_icd.py              # ICD description enrichment utilities
│
├── baselines/
│   ├── run_traditional_ml.py      # BoC+LR, BoC+RF, SBERT+LR, SBERT+RF
│   └── run_bert.py                # BERT baseline
│
├── hosp/                          # MIMIC-IV data files (not included)
│   ├── admissions.csv
│   ├── diagnoses_icd.csv
│   └── d_icd_diagnoses.csv
│
└── outputs/                       # Generated checkpoints & embeddings
    ├── sbert_embeddings.pt
    ├── pretrained_sent_e_med.pt
    └── vocab.pkl
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch>=2.0.0`, `sentence-transformers>=2.2.0`, `scikit-learn>=1.2.0`, `pandas>=1.5.0`

---

## Data

Requires access to **MIMIC-IV** (PhysioNet). Place the following files under `hosp/`:

```
hosp/
    admissions.csv
    diagnoses_icd.csv
    d_icd_diagnoses.csv      ← ICD code → text description mapping (built into MIMIC-IV)
```

The `d_icd_diagnoses.csv` file provides the ICD code → clinical text description mapping used to build the frozen code embeddings. No external lookup is needed — the mapping is part of MIMIC-IV itself.

---

## Usage

### 1. Pre-train + Fine-tune (default: SBERT encoder, code-level masking)

```bash
python main.py --mimic_dir hosp/
```

### 2. BioClinicalBERT encoder

```bash
python main.py --mimic_dir hosp/ --encoder bio_clinical_bert
```

### 3. Visit-level masking (variant)

```bash
python main.py --mimic_dir hosp/ --masking visit
```

### 4. MLM-only (no Next Visit Prediction)

```bash
python main.py --mimic_dir hosp/ --mlm_only
```

### 5. Skip pre-training, load existing checkpoint

```bash
python main.py --mimic_dir hosp/ --skip_pretrain
```

### 6. Fine-tune specific conditions only

```bash
python main.py --mimic_dir hosp/ --skip_pretrain --conditions SUD Diabetes
```

### 7. Run traditional ML baselines

```bash
python baselines/run_traditional_ml.py --mimic_dir hosp/

# Skip SBERT features (faster)
python baselines/run_traditional_ml.py --mimic_dir hosp/ --no_sbert
```

---

## Model Architecture

```
ICD code → d_icd_diagnoses.csv → text description
                                        ↓
                            Frozen Text Encoder
                       (SBERT all-MiniLM-L6-v2 / BioClinicalBERT)
                                        ↓
                                   code_emb  (384-dim, frozen)
                                        +
                                  visit_emb  (384-dim, learnable)
                                        +
                                  [MASK] emb  (384-dim, learnable, at masked positions)
                                        ↓
                            Transformer Encoder
                         (4 layers, 4 heads, dim=384, FFN=1536)
                                        ↓
                          ┌─────────────────────────┐
                   Pre-training                 Fine-tuning
               MLM Head + NVP Head          Mean Pool → MLP → Risk Score
```

### Key design choices

| Component | Detail |
|-----------|--------|
| Encoder | SBERT all-MiniLM-L6-v2 (384-dim) or BioClinicalBERT (768→384 projected) |
| Code embeddings | Frozen — never updated during training |
| Visit embeddings | Learnable segment embeddings (like BERT segment tokens) |
| Masking | Code-level (15% of tokens) or Visit-level (15% of visits) |
| MLM scheme | 80% → [MASK], 10% → random code, 10% → unchanged |
| Pre-training objectives | MLM + NVP (optional) |
| Fine-tuning | All parameters updated except frozen code embeddings |
| PheCode fusion | Optional: concat or gated fusion of ICD + PheCode embeddings |

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden dim | 384 |
| Transformer layers | 4 |
| Attention heads | 4 |
| FFN dim | 1536 |
| Max sequence length | 128 tokens |
| Pre-train LR | 1e-5 (AdamW) |
| Fine-tune LR | 1e-5 (AdamW) |
| Pre-train epochs | 50 |
| Fine-tune epochs | 20 |
| Batch size | 32 |
| Evaluation runs | 5 (mean ± std) |

---

## Evaluation

Metrics: **ROC-AUC** and **PR-AUC**, averaged over 5 independent runs with different train/val/test splits.

Conditions evaluated: **SUD** (Substance Use Disorder), **OUD** (Opioid Use Disorder), **Diabetes**

Baselines compared:
- BoC + Logistic Regression
- BoC + Random Forest
- SBERT mean-pool + Logistic Regression
- SBERT mean-pool + Random Forest

---

## Outputs

Results and checkpoints are saved to the `outputs/` directory:

| File | Description |
|------|-------------|
| `sbert_embeddings.pt` | Frozen SBERT code embedding table |
| `pretrained_sent_e_med.pt` | Best pre-trained model checkpoint |
| `vocab.pkl` | ICD vocabulary mapping |
| `{condition}_results.pkl` | Fine-tuning results per condition |
| `{condition}_{baseline}_results.pkl` | Baseline results per condition |
