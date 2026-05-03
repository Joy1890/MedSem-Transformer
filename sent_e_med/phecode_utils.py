"""
phecode_utils.py
================
PheCode dual-embedding support for Sent-e-Med.

PheCode groups thousands of ICD codes into ~1,800 clinically meaningful
phenotypes.  Each code gets a coarse-grained text label ("Alcohol-related
disorders") alongside its fine-grained ICD description ("Alcohol dependence
with unspecified withdrawal").  Both descriptions are encoded by SBERT and
fused in the model, giving every token two levels of clinical granularity.

Mapping file expected: phecodes_cm_rolled.csv
Columns: vocabulary_id | code | code_description | phecode | phecode_description

Public API:
  load_phecode_map()            — build (norm_icd_code, version) → phecode_description
  get_phecode_description()     — single-code lookup with ICD fallback
  build_phecode_embeddings()    — encode all vocab codes via SBERT, with caching
  extend_phecode_embeddings()   — extend table for OOV codes at fine-tuning time

Reference: https://phewascatalog.org/phecodes
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load ICD → PheCode description mapping
# ─────────────────────────────────────────────────────────────────────────────

def load_phecode_map(phecode_map_path: str) -> Dict[Tuple[str, int], str]:
    """
    Load ICD→PheCode description mapping from phecodes_cm_rolled.csv.

    Only ICD9CM and ICD10CM rows are used; SNOMEDCT_US rows are skipped
    (MIMIC-IV diagnoses_icd.csv uses ICD only).

    Codes in the CSV have dots (e.g., "A00.0", "304.00").  They are normalized
    here — dots removed, whitespace stripped, uppercased — to match the format
    MIMIC-IV uses (e.g., "A000", "30400").

    Args:
        phecode_map_path: path to phecodes_cm_rolled.csv

    Returns:
        dict: (normalized_icd_code, icd_version) → phecode_description
        e.g.: ("F1013", 10) → "Alcohol-related disorders"
              ("30400",  9) → "Opioid dependence"
    """
    vocab_to_version = {"ICD10CM": 10, "ICD9CM": 9}
    mapping: Dict[Tuple[str, int], str] = {}

    with open(phecode_map_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vocab_id = row["vocabulary_id"]
            if vocab_id not in vocab_to_version:
                continue  # skip SNOMEDCT_US

            version = vocab_to_version[vocab_id]
            # Normalize to MIMIC format: remove dots, strip, uppercase
            code = row["code"].replace(".", "").strip().upper()
            desc = row["phecode_description"].strip()

            if code and desc:
                key = (code, version)
                if key not in mapping:   # first occurrence wins
                    mapping[key] = desc

    print(
        f"Loaded {len(mapping):,} ICD→PheCode mappings "
        f"from {phecode_map_path}"
    )
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# 2. Single-code lookup helper
# ─────────────────────────────────────────────────────────────────────────────

def get_phecode_description(
    code: str,
    version: int,
    phecode_map: Dict[Tuple[str, int], str],
    icd_description_fallback: str = "",
) -> str:
    """
    Look up the PheCode description for one ICD code.

    Falls back to:
      1. The other ICD version (9 ↔ 10) if the primary lookup misses.
      2. The raw ICD description (passed as icd_description_fallback).
      3. "ICD code <code>" as a last resort.

    This graceful fallback is important: ~30% of MIMIC ICD codes have no
    PheCode match; using the ICD description keeps those embeddings meaningful.
    """
    desc = phecode_map.get((code, version))
    if desc:
        return desc

    other_version = 9 if version == 10 else 10
    desc = phecode_map.get((code, other_version))
    if desc:
        return desc

    return icd_description_fallback if icd_description_fallback else f"ICD code {code}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build PheCode embedding table
# ─────────────────────────────────────────────────────────────────────────────

def build_phecode_embeddings(
    idx_to_code: List[Tuple[str, int]],
    phecode_map: Dict[Tuple[str, int], str],
    icd_descriptions: Dict[Tuple[str, int], str],
    sbert_model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    cache_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Build SBERT embeddings for PheCode descriptions for every vocab code.

    For each ICD code in the vocabulary:
      1. Look up its PheCode description (coarse-grained phenotype label).
      2. Fall back to the ICD description if no PheCode match exists.
      3. Encode the description via SBERT → 384-dim vector.

    PheCode embeddings always use SBERT (all-MiniLM-L6-v2, 384-dim) regardless
    of the ICD encoder configured in config.encoder_type.  This keeps the
    phecode branch at exactly hidden_dim (384) so no extra projection is needed
    in the fusion layer.

    The table is cached as a .pt file after the first run so that subsequent
    runs skip re-encoding (encoding ~15K codes takes ~30 seconds on CPU).

    Args:
        idx_to_code:       vocab list, (icd_code, icd_version) at each index
        phecode_map:       from load_phecode_map()
        icd_descriptions:  from load_icd_descriptions() — used as fallback text
        sbert_model_name:  should match the ICD SBERT model for consistency
        batch_size:        SBERT encoding batch size
        cache_path:        if provided, save/reload tensor at this path

    Returns:
        FloatTensor of shape (vocab_size, 384)
    """
    # ── Try cache ─────────────────────────────────────────────────────────────
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached PheCode embeddings from {cache_path} …")
        emb = torch.load(cache_path, map_location="cpu")
        print(f"  Shape: {tuple(emb.shape)}")
        return emb

    # ── Load SBERT ────────────────────────────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required.\n"
            "Install with:  pip install sentence-transformers"
        )

    from .icd_utils import get_description

    print(f"Building PheCode embeddings using SBERT '{sbert_model_name}' …")
    sbert = SentenceTransformer(sbert_model_name)
    sbert.eval()

    # ── Build description list in vocab order ─────────────────────────────────
    matched = 0
    text_list: List[str] = []

    for code, version in idx_to_code:
        icd_desc = get_description(code, version, icd_descriptions)
        ph_desc  = get_phecode_description(code, version, phecode_map, icd_desc)
        text_list.append(ph_desc)
        if phecode_map.get((code, version)) or phecode_map.get(
            (code, 9 if version == 10 else 10)
        ):
            matched += 1

    coverage = 100.0 * matched / max(len(idx_to_code), 1)
    print(
        f"  PheCode coverage: {matched:,}/{len(idx_to_code):,} codes "
        f"({coverage:.1f}%)  |  "
        f"{len(idx_to_code) - matched:,} using ICD description fallback"
    )

    # ── Encode ────────────────────────────────────────────────────────────────
    print(f"  Encoding {len(text_list):,} descriptions …")
    embeddings_np: np.ndarray = sbert.encode(
        text_list,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = torch.tensor(embeddings_np, dtype=torch.float32)
    print(f"PheCode embeddings shape: {tuple(embeddings.shape)}")

    # ── Save cache ────────────────────────────────────────────────────────────
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_path)
        print(f"Saved PheCode embeddings to {cache_path}")

    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# 4. Extend table for OOV codes (fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────

def extend_phecode_embeddings(
    current_phecode_embeddings: torch.Tensor,
    new_codes: List[Tuple[str, int]],
    phecode_map: Dict[Tuple[str, int], str],
    icd_descriptions: Dict[Tuple[str, int], str],
    sbert_model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Extend the PheCode embedding table with embeddings for OOV codes.

    Called alongside extend_embeddings() in sbert_utils.py when fine-tuning
    data introduces ICD codes that were not in the pretraining vocabulary.

    Returns:
        extended: FloatTensor of shape (old_vocab + len(new_codes), 384)
    """
    if not new_codes:
        return current_phecode_embeddings

    from sentence_transformers import SentenceTransformer
    from .icd_utils import get_description

    sbert = SentenceTransformer(sbert_model_name)
    sbert.eval()

    text_list = [
        get_phecode_description(
            code, version, phecode_map,
            get_description(code, version, icd_descriptions),
        )
        for code, version in new_codes
    ]

    print(f"Computing PheCode embeddings for {len(new_codes)} OOV codes …")
    new_emb_np = sbert.encode(
        text_list,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    new_emb  = torch.tensor(new_emb_np, dtype=torch.float32)
    extended = torch.cat([current_phecode_embeddings, new_emb], dim=0)
    print(f"Extended PheCode embedding table: {tuple(extended.shape)}")
    return extended
