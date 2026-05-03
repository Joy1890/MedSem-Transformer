"""
sbert_utils.py
==============
Pre-compute and cache frozen text embeddings for all ICD codes.

Two encoders are supported (controlled by config.encoder_type):
  "sbert":
    sentence-transformers all-MiniLM-L6-v2 → 384-dim sentence embedding.
    Called via build_sbert_embeddings().

  "bio_clinical_bert" (variant):
    emilyalsentzer/Bio_ClinicalBERT — a BERT model fine-tuned on MIMIC-III
    clinical notes. Extracts the [CLS] token (768-dim), then projects it down
    to hidden_dim (384) via a learned linear layer stored in the model.
    Called via build_bioclinicalbert_embeddings().

In both cases the resulting embedding table is FROZEN during training.

For new OOV codes seen only at fine-tuning time:
  - Sent-e-Med's advantage: compute embeddings for their text descriptions.
    No retraining needed. extend_embeddings() handles this.

Public API:
  build_code_embeddings()          — unified dispatcher (use this in main.py)
  build_sbert_embeddings()         — SBERT path
  build_bioclinicalbert_embeddings() — Bio_ClinicalBERT path
  extend_embeddings()              — OOV extension
  update_model_embeddings()        — swap embedding buffer in a live model
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .icd_utils import get_description


# ─────────────────────────────────────────────────────────────────────────────
# Main embedding builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sbert_embeddings(
    idx_to_code: List[Tuple[str, int]],
    descriptions: Dict[Tuple[str, int], str],
    sbert_model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    cache_path: Optional[str] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Pre-compute SBERT sentence embeddings for every code in the vocabulary.

    Args:
        idx_to_code:       vocabulary list — (icd_code, icd_version) at each index
        descriptions:      (icd_code, icd_version) → long_title text
        sbert_model_name:  HuggingFace / sentence-transformers model name
        batch_size:        encoding mini-batch size
        cache_path:        if set, save tensor here and reload on future runs
        device:            'cuda' | 'cpu' | None (auto-detect)

    Returns:
        embeddings: FloatTensor of shape (vocab_size, embedding_dim)
                    — 384-dim for "all-MiniLM-L6-v2"
    """
    # ── Try cache ─────────────────────────────────────────────────────────────
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached SBERT embeddings from {cache_path} …")
        emb = torch.load(cache_path, map_location="cpu")
        print(f"  Shape: {tuple(emb.shape)}")
        return emb

    # ── Load SBERT model ──────────────────────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required.\n"
            "Install with:  pip install sentence-transformers"
        )

    print(f"Loading SBERT model '{sbert_model_name}' …")
    sbert = SentenceTransformer(sbert_model_name)
    sbert.eval()

    # ── Build description list in vocab order ─────────────────────────────────
    text_list: List[str] = []
    for code_pair in idx_to_code:
        text = get_description(code_pair[0], code_pair[1], descriptions)
        text_list.append(text)

    print(f"Encoding {len(text_list):,} ICD code descriptions …")
    embeddings_np: np.ndarray = sbert.encode(
        text_list,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # keep raw L2-unnormalized embeddings
    )

    embeddings = torch.tensor(embeddings_np, dtype=torch.float32)
    print(f"SBERT embeddings shape: {tuple(embeddings.shape)}")

    # ── Save cache ────────────────────────────────────────────────────────────
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_path)
        print(f"Saved embeddings to {cache_path}")

    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Bio_ClinicalBERT encoder (Variant 1)
# ─────────────────────────────────────────────────────────────────────────────

def build_bioclinicalbert_embeddings(
    idx_to_code: List[Tuple[str, int]],
    descriptions: Dict[Tuple[str, int], str],
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    batch_size: int = 64,
    cache_path: Optional[str] = None,
    max_length: int = 64,
) -> torch.Tensor:
    """
    Pre-compute Bio_ClinicalBERT [CLS] token embeddings for every ICD code.

    Bio_ClinicalBERT is a BERT-base model (12 layers, 768-dim hidden) fine-tuned
    on clinical notes from MIMIC-III. It has strong domain knowledge about
    clinical terminology, making it well-suited for encoding ICD descriptions.

    How it works:
        text → tokenize → Bio_ClinicalBERT → last hidden state → [CLS] token
        → 768-dim vector (raw, not normalized)

    The 768-dim output is stored as-is in the embedding table. The model then
    applies a projection layer (Linear 768→hidden_dim) that IS learned during
    pretraining — this is the only architectural difference vs the SBERT variant.

    Args:
        idx_to_code:  vocabulary list — (icd_code, icd_version) at each index
        descriptions: (icd_code, icd_version) → long_title text
        model_name:   HuggingFace path (default: emilyalsentzer/Bio_ClinicalBERT)
        batch_size:   encoding mini-batch size (smaller than SBERT due to BERT size)
        cache_path:   if set, save/load tensor from this path
        max_length:   max tokenizer length (ICD descriptions are short, 64 is fine)

    Returns:
        embeddings: FloatTensor of shape (vocab_size, 768)
                    NOTE: 768-dim, not 384. The model's input_projection handles
                    mapping 768 → hidden_dim before the transformer.
    """
    # ── Try cache ─────────────────────────────────────────────────────────────
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached Bio_ClinicalBERT embeddings from {cache_path} …")
        emb = torch.load(cache_path, map_location="cpu")
        print(f"  Shape: {tuple(emb.shape)}")
        return emb

    # ── Load model and tokenizer ──────────────────────────────────────────────
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers is required for Bio_ClinicalBERT.\n"
            "Install with:  pip install transformers"
        )

    print(f"Loading Bio_ClinicalBERT model '{model_name}' …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    bert_model.eval()

    enc_device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model = bert_model.to(enc_device)

    # ── Build description list ────────────────────────────────────────────────
    text_list: List[str] = [
        get_description(code, version, descriptions)
        for code, version in idx_to_code
    ]

    print(f"Encoding {len(text_list):,} ICD descriptions with Bio_ClinicalBERT …")

    all_embeddings: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(text_list), batch_size):
            batch_texts = text_list[start: start + batch_size]

            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(enc_device) for k, v in encoded.items()}

            # Forward pass — take [CLS] token (index 0) from last hidden state
            outputs = bert_model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (B, 768)

            all_embeddings.append(cls_embeddings.cpu().numpy())

            if (start // batch_size) % 10 == 0:
                print(f"  {start + len(batch_texts)}/{len(text_list)}")

    embeddings_np = np.vstack(all_embeddings)
    embeddings = torch.tensor(embeddings_np, dtype=torch.float32)
    print(f"Bio_ClinicalBERT embeddings shape: {tuple(embeddings.shape)}")

    # ── Save cache ────────────────────────────────────────────────────────────
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_path)
        print(f"Saved to {cache_path}")

    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Unified dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def build_code_embeddings(
    config,
    idx_to_code: List[Tuple[str, int]],
    descriptions: Dict[Tuple[str, int], str],
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Build frozen code embeddings for the configured encoder type.

    Reads config.encoder_type and calls the appropriate builder.
    Cache filenames are automatically distinguished so you can switch
    encoders without invalidating each other's cache.

    Returns:
        FloatTensor of shape (vocab_size, encoder_dim)
        where encoder_dim = 384 for sbert, 768 for bio_clinical_bert
    """
    if config.encoder_type == "sbert":
        cache_path = (
            str(Path(cache_dir) / "sbert_embeddings.pt") if cache_dir else None
        )
        return build_sbert_embeddings(
            idx_to_code=idx_to_code,
            descriptions=descriptions,
            sbert_model_name=config.sbert_model_name,
            cache_path=cache_path,
        )
    elif config.encoder_type == "bio_clinical_bert":
        cache_path = (
            str(Path(cache_dir) / "bio_clinical_bert_embeddings.pt")
            if cache_dir
            else None
        )
        return build_bioclinicalbert_embeddings(
            idx_to_code=idx_to_code,
            descriptions=descriptions,
            model_name=config.bio_clinical_bert_model,
            cache_path=cache_path,
        )
    else:
        raise ValueError(
            f"Unknown encoder_type '{config.encoder_type}'. "
            "Choose 'sbert' or 'bio_clinical_bert'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Extend embeddings for OOV codes (fine-tuning on new datasets)
# ─────────────────────────────────────────────────────────────────────────────

def extend_embeddings(
    current_embeddings: torch.Tensor,          # (vocab_size, encoder_dim)
    new_codes: List[Tuple[str, int]],           # OOV codes to add
    descriptions: Dict[Tuple[str, int], str],
    encoder_type: str = "sbert",
    sbert_model_name: str = "all-MiniLM-L6-v2",
    bio_clinical_bert_model: str = "emilyalsentzer/Bio_ClinicalBERT",
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Compute embeddings for new OOV codes and concatenate to the existing table.

    Works for both encoder types (sbert and bio_clinical_bert) — pass the same
    encoder_type that was used to build the original embedding table.

    Returns:
        extended embeddings: (vocab_size + len(new_codes), encoder_dim)
    """
    if not new_codes:
        return current_embeddings

    text_list = [get_description(c, v, descriptions) for c, v in new_codes]

    if encoder_type == "sbert":
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer(sbert_model_name)
        sbert.eval()
        print(f"Computing SBERT embeddings for {len(new_codes)} OOV codes …")
        new_emb_np = sbert.encode(
            text_list, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True
        )
        new_emb = torch.tensor(new_emb_np, dtype=torch.float32)

    elif encoder_type == "bio_clinical_bert":
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(bio_clinical_bert_model)
        bert_model = AutoModel.from_pretrained(bio_clinical_bert_model)
        bert_model.eval()
        enc_device = "cuda" if torch.cuda.is_available() else "cpu"
        bert_model = bert_model.to(enc_device)
        cls_list = []
        print(f"Computing Bio_ClinicalBERT embeddings for {len(new_codes)} OOV codes …")
        with torch.no_grad():
            for start in range(0, len(text_list), batch_size):
                batch = text_list[start: start + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=64, return_tensors="pt")
                enc = {k: v.to(enc_device) for k, v in enc.items()}
                out = bert_model(**enc)
                cls_list.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        new_emb = torch.tensor(np.vstack(cls_list), dtype=torch.float32)
    else:
        raise ValueError(f"Unknown encoder_type '{encoder_type}'")

    extended = torch.cat([current_embeddings, new_emb], dim=0)
    print(f"Extended embedding table: {tuple(extended.shape)}")
    return extended


# ─────────────────────────────────────────────────────────────────────────────
# Utility: update model embedding table in-place (after extending vocabulary)
# ─────────────────────────────────────────────────────────────────────────────

def update_model_embeddings(
    model,
    new_embeddings: torch.Tensor,
    new_phecode_embeddings: Optional[torch.Tensor] = None,
) -> None:
    """
    Replace the model's frozen embedding buffers with extended tables.

    Call this after extend_embeddings() (and optionally extend_phecode_embeddings())
    when fine-tuning on a new dataset that contains OOV codes not seen during
    pretraining.

    Args:
        model:                   SenteMed instance
        new_embeddings:          (new_vocab_size, encoder_dim) ICD embedding table
        new_phecode_embeddings:  (new_vocab_size, 384) PheCode table — only
                                 required when model.use_phecode is True
    """
    device = model.sbert_code_embeddings.device
    model.register_buffer("sbert_code_embeddings", new_embeddings.to(device))
    model.vocab_size = new_embeddings.shape[0]

    # Resize the MLM and NVP heads to match new vocab size
    old_mlm_out = model.mlm_head[-1]
    model.mlm_head[-1] = torch.nn.Linear(
        old_mlm_out.in_features, model.vocab_size, bias=True
    ).to(device)
    old_nvp_out = model.nvp_head[-1]
    model.nvp_head[-1] = torch.nn.Linear(
        old_nvp_out.in_features, model.vocab_size, bias=True
    ).to(device)

    # Also update the PheCode buffer if the model uses phecode fusion
    if model.use_phecode:
        if new_phecode_embeddings is None:
            raise ValueError(
                "model.use_phecode=True but new_phecode_embeddings was not provided. "
                "Pass the extended phecode table from extend_phecode_embeddings()."
            )
        model.register_buffer(
            "phecode_code_embeddings", new_phecode_embeddings.to(device)
        )
        print(
            f"Model embedding tables updated → vocab_size={model.vocab_size:,} "
            f"(ICD + PheCode buffers)"
        )
    else:
        print(f"Model embedding table updated → vocab_size={model.vocab_size:,}")
