"""
dataset.py
==========
PyTorch Dataset classes for pretraining and fine-tuning.

PretrainDataset
    Each sample = one patient's visit sequence.
    Applies MLM masking and prepares Next Visit Prediction labels.

FinetuneDataset
    Each sample = patient's truncated visit sequence + binary label (0/1).
    No masking; used for risk prediction classification.

Shared sequence encoding logic:
    A patient's visits are flattened into a 1D token sequence:
        [code_1_v1, code_2_v1, ..., code_1_v2, code_2_v2, ..., code_1_vT, ...]

    Each token carries two IDs:
        code_id   — index into the ICD vocabulary (→ SBERT embedding in model)
        visit_id  — which visit this code belongs to (→ visit embedding in model)

    The sequence is padded/truncated to max_seq_len.
    attention_mask: 1 = real token, 0 = padding.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def encode_visits(
    visits: List[dict],
    code_to_idx: Dict[Tuple[str, int], int],
    max_seq_len: int,
    max_visits: int,
) -> Tuple[List[int], List[int]]:
    """
    Flatten patient visits into parallel (code_ids, visit_ids) lists.

    Codes are added in priority order within each visit (seq_num order
    preserved from data_processing.py). Truncated at max_seq_len.

    Unknown codes (not in vocabulary) are silently skipped.

    Returns:
        code_ids:  list of int — ICD code vocabulary indices
        visit_ids: list of int — visit position indices (0-indexed)
    """
    code_ids: List[int] = []
    visit_ids: List[int] = []

    for visit_idx, visit in enumerate(visits[:max_visits]):
        if len(code_ids) >= max_seq_len:
            break
        for code_pair in visit["codes"]:
            if len(code_ids) >= max_seq_len:
                break
            idx = code_to_idx.get(code_pair)
            if idx is not None:
                code_ids.append(idx)
                visit_ids.append(visit_idx)

    return code_ids, visit_ids


def pad_sequence(
    code_ids: List[int],
    visit_ids: List[int],
    max_seq_len: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Pad (code_ids, visit_ids) to max_seq_len.

    Returns:
        padded_code_ids, padded_visit_ids, attention_mask
        (attention_mask: 1=real, 0=padding)
    """
    seq_len = len(code_ids)
    pad_len = max_seq_len - seq_len

    padded_code_ids  = code_ids  + [0] * pad_len
    padded_visit_ids = visit_ids + [0] * pad_len
    attention_mask   = [1] * seq_len + [0] * pad_len

    return padded_code_ids, padded_visit_ids, attention_mask


# ─────────────────────────────────────────────────────────────────────────────
# Pretraining Dataset
# ─────────────────────────────────────────────────────────────────────────────

# Sentinel value used to signal [MASK] tokens in code_ids.
# The model checks for this value and uses its learnable mask_embedding instead
# of the SBERT code embedding.
MASK_TOKEN_ID = -1   # set to -1; model clamps to 0 for safe indexing, uses flag


class PretrainDataset(Dataset):
    """
    Dataset for pretraining with:
      - Masked Language Modeling (MLM)
      - Next Visit Prediction (NVP)   [optional — disable with use_nvp=False]

    Two masking strategies (controlled by masking_strategy):

      "code"  (original paper, Section 4.2.2):
        15% of INDIVIDUAL CODES are selected across the entire sequence.
        80% → [MASK], 10% → random code, 10% → unchanged.

      "visit" (variant):
        15% of VISITS are selected. ALL codes within a selected visit are masked.
        Same 80/10/10 substitution rule is applied per code within masked visits.
        This forces the model to reconstruct entire clinical encounters from context,
        which may better capture visit-level patterns.

    NVP (when use_nvp=True):
      Input  = all visits EXCEPT the last one.
      Labels = multi-hot vector of codes in the LAST visit.
      Loss   = binary cross-entropy (multi-label).
    """

    def __init__(
        self,
        patient_visits: Dict[int, List[dict]],
        patient_ids: List[int],
        code_to_idx: Dict[Tuple[str, int], int],
        vocab_size: int,
        max_seq_len: int = 128,
        max_visits: int = 100,
        mlm_probability: float = 0.15,
        mlm_mask_token_prob: float = 0.80,
        mlm_random_token_prob: float = 0.10,
        use_nvp: bool = True,
        masking_strategy: str = "code",    # "code" or "visit"
        visit_mask_probability: float = 0.15,
        seed: int = 42,
    ):
        self.patient_visits        = patient_visits
        self.patient_ids           = [p for p in patient_ids if p in patient_visits]
        self.code_to_idx           = code_to_idx
        self.vocab_size            = vocab_size
        self.max_seq_len           = max_seq_len
        self.max_visits            = max_visits
        self.mlm_probability       = mlm_probability
        self.mlm_mask_token_prob   = mlm_mask_token_prob
        self.mlm_rand_token_prob   = mlm_random_token_prob
        self.use_nvp               = use_nvp
        self.masking_strategy      = masking_strategy
        self.visit_mask_probability = visit_mask_probability
        self.rng                   = np.random.RandomState(seed)

        if masking_strategy not in ("code", "visit"):
            raise ValueError(
                f"masking_strategy must be 'code' or 'visit', got '{masking_strategy}'"
            )

    def __len__(self) -> int:
        return len(self.patient_ids)

    # ── Masking: shared substitution logic ──────────────────────────────────

    def _substitute(
        self,
        code_ids: List[int],
        masked_code_ids: List[int],
        is_masked: List[bool],
        mlm_labels: List[int],
        idx: int,
    ) -> None:
        """Apply 80/10/10 substitution at position idx (in-place)."""
        is_masked[idx]    = True
        mlm_labels[idx]   = code_ids[idx]
        r = self.rng.random()
        if r < self.mlm_mask_token_prob:
            masked_code_ids[idx] = MASK_TOKEN_ID
        elif r < self.mlm_mask_token_prob + self.mlm_rand_token_prob:
            masked_code_ids[idx] = int(self.rng.randint(0, self.vocab_size))
        # else: keep unchanged (10%)

    # ── Masking: code-level (original paper) ────────────────────────────────

    def _apply_mlm(
        self,
        code_ids: List[int],
    ) -> Tuple[List[int], List[bool], List[int]]:
        """
        Code-level masking (original paper, Section 4.2.2).

        Randomly selects 15% of individual codes and masks them.

        Returns:
            masked_code_ids: MASK_TOKEN_ID at masked positions
            is_masked:       True at positions selected for loss
            mlm_labels:      original code id at masked positions; -100 elsewhere
        """
        n        = len(code_ids)
        n_mask   = max(1, int(n * self.mlm_probability))
        selected = self.rng.choice(n, size=n_mask, replace=False)

        masked_code_ids = list(code_ids)
        is_masked       = [False] * n
        mlm_labels      = [-100]  * n

        for idx in selected:
            self._substitute(code_ids, masked_code_ids, is_masked, mlm_labels, idx)

        return masked_code_ids, is_masked, mlm_labels

    # ── Masking: visit-level (Variant 2) ────────────────────────────────────

    def _apply_visit_level_mlm(
        self,
        code_ids:  List[int],
        visit_ids: List[int],
    ) -> Tuple[List[int], List[bool], List[int]]:
        """
        Visit-level masking (Variant 2).

        Randomly selects 15% of VISITS. ALL codes in each selected visit are
        masked (same 80/10/10 per-code substitution rule still applies).

        This is a coarser masking granularity — the model must reconstruct an
        entire clinical encounter from surrounding visits, which can encourage
        stronger visit-level contextual representations.

        Args:
            code_ids:  flat list of code indices (all visits concatenated)
            visit_ids: parallel list — which visit index each code belongs to

        Returns:
            masked_code_ids, is_masked, mlm_labels  (same format as code-level)
        """
        # Find all unique visit indices present in this sequence
        unique_visits = sorted(set(visit_ids))
        n_visits      = len(unique_visits)
        n_mask_visits = max(1, int(n_visits * self.visit_mask_probability))
        selected_visits = set(
            self.rng.choice(unique_visits, size=n_mask_visits, replace=False).tolist()
        )

        masked_code_ids = list(code_ids)
        is_masked       = [False] * len(code_ids)
        mlm_labels      = [-100]  * len(code_ids)

        for pos, visit_idx in enumerate(visit_ids):
            if visit_idx in selected_visits:
                self._substitute(
                    code_ids, masked_code_ids, is_masked, mlm_labels, pos
                )

        return masked_code_ids, is_masked, mlm_labels

    # ── NVP labels ───────────────────────────────────────────────────────────

    def _make_nvp_label(self, last_visit: dict) -> torch.Tensor:
        """Multi-hot vector over vocabulary for codes in last visit."""
        label = torch.zeros(self.vocab_size, dtype=torch.float32)
        for code_pair in last_visit["codes"]:
            idx = self.code_to_idx.get(code_pair)
            if idx is not None:
                label[idx] = 1.0
        return label

    # ── __getitem__ ──────────────────────────────────────────────────────────

    def __getitem__(self, index: int) -> dict:
        pid = self.patient_ids[index]
        visits = self.patient_visits[pid]

        # For NVP: encode all visits except last; predict last visit's codes
        if self.use_nvp and len(visits) >= 2:
            input_visits = visits[:-1]
            nvp_label = self._make_nvp_label(visits[-1])
        else:
            input_visits = visits
            nvp_label = torch.zeros(self.vocab_size, dtype=torch.float32)

        # Flatten to code/visit ID sequences
        code_ids, visit_ids = encode_visits(
            input_visits, self.code_to_idx, self.max_seq_len, self.max_visits
        )

        # Edge case: patient has no valid codes in vocabulary
        if not code_ids:
            code_ids, visit_ids = [0], [0]

        # Apply masking (code-level or visit-level)
        if self.masking_strategy == "visit":
            masked_code_ids, is_masked, mlm_labels = self._apply_visit_level_mlm(
                code_ids, visit_ids
            )
        else:  # "code" — original paper
            masked_code_ids, is_masked, mlm_labels = self._apply_mlm(code_ids)

        # Pad to max_seq_len
        masked_code_ids, visit_ids, attention_mask = pad_sequence(
            masked_code_ids, visit_ids, self.max_seq_len
        )
        is_masked  = is_masked  + [False] * (self.max_seq_len - len(is_masked))
        mlm_labels = mlm_labels + [-100]  * (self.max_seq_len - len(mlm_labels))

        return {
            # Input tokens (MASK_TOKEN_ID where masked; model handles sentinel)
            "code_ids":       torch.tensor(masked_code_ids, dtype=torch.long),
            # Which visit each token belongs to (for visit embeddings)
            "visit_ids":      torch.tensor(visit_ids,       dtype=torch.long),
            # 1 = real token, 0 = padding
            "attention_mask": torch.tensor(attention_mask,  dtype=torch.long),
            # True at positions selected for MLM loss
            "is_masked":      torch.tensor(is_masked,       dtype=torch.bool),
            # Original code ids at masked positions; -100 elsewhere (ignored)
            "mlm_labels":     torch.tensor(mlm_labels,      dtype=torch.long),
            # Multi-hot: codes present in next visit (all zeros if use_nvp=False)
            "nvp_labels":     nvp_label,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FinetuneDataset(Dataset):
    """
    Dataset for binary risk classification fine-tuning.

    Each sample is a patient record truncated at the target visit
    (first visit with condition for cases; last visit for controls).
    Label: 1 = will develop condition, 0 = will not.

    No MLM masking is applied here.
    """

    def __init__(
        self,
        samples: List[dict],
        code_to_idx: Dict[Tuple[str, int], int],
        max_seq_len: int = 128,
        max_visits: int = 100,
    ):
        self.samples     = samples
        self.code_to_idx = code_to_idx
        self.max_seq_len = max_seq_len
        self.max_visits  = max_visits

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        visits = sample["visits"]
        label  = sample["label"]

        code_ids, visit_ids = encode_visits(
            visits, self.code_to_idx, self.max_seq_len, self.max_visits
        )

        if not code_ids:
            code_ids, visit_ids = [0], [0]

        code_ids, visit_ids, attention_mask = pad_sequence(
            code_ids, visit_ids, self.max_seq_len
        )

        return {
            "code_ids":       torch.tensor(code_ids,       dtype=torch.long),
            "visit_ids":      torch.tensor(visit_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label":          torch.tensor(label,          dtype=torch.float32),
        }
