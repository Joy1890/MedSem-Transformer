"""
data_processing.py
==================
MIMIC-IV data loading and preprocessing
Pipeline:
  1. load_mimic_data()           — load admissions + diagnoses tables
  2. build_patient_sequences()   — chronological visit sequences per patient
  3. build_vocabulary()          — ICD code vocab (code, version) → int index
  4. split_patients()            — 7:2:1 pretrain/val/test split
  5. create_condition_dataset()  — case/control groups (Section 3.3 of paper)

Patient sequence format (used throughout the codebase):
    {
        subject_id (int): [
            {
                "hadm_id":   int,
                "admittime": pd.Timestamp,
                "codes":     [(icd_code: str, icd_version: int), ...]
                             # ordered by seq_num (priority within visit)
            },
            ...  # visits in chronological order
        ]
    }
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .icd_utils import make_condition_matcher


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load MIMIC-IV raw tables
# ─────────────────────────────────────────────────────────────────────────────

def load_mimic_data(mimic_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MIMIC-IV admissions and diagnoses_icd tables.

    Supports both flat layout and MIMIC-IV hosp/ subdirectory layout.

    Returns:
        admissions  — columns: subject_id, hadm_id, admittime, dischtime
        diagnoses   — columns: subject_id, hadm_id, seq_num, icd_code, icd_version
    """
    base = Path(mimic_dir) # mimic_dir the path where store the hosp folder 
    hosp = base / "hosp"

    def _find(filename: str) -> Path:
        for p in [hosp / filename, base / filename]:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"{filename} not found in {mimic_dir}. "
            "Expected under hosp/ or directly in mimic_dir."
        )

    print("Loading admissions.csv …")
    adm_path = _find("admissions.csv")
    admissions = pd.read_csv(
        adm_path,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        parse_dates=["admittime", "dischtime"],
    )
    admissions = admissions.dropna(subset=["admittime"])
    print(f"  {len(admissions):,} admissions, {admissions['subject_id'].nunique():,} patients")

    print("Loading diagnoses_icd.csv …")
    diag_path = _find("diagnoses_icd.csv")
    diagnoses = pd.read_csv(
        diag_path,
        dtype={"icd_code": str, "icd_version": int},
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    )
    diagnoses["icd_code"] = diagnoses["icd_code"].str.strip()
    diagnoses = diagnoses.dropna(subset=["icd_code"])
    print(f"  {len(diagnoses):,} diagnosis records")

    return admissions, diagnoses


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build patient visit sequences
# ─────────────────────────────────────────────────────────────────────────────

def build_patient_sequences(
    admissions: pd.DataFrame,
    diagnoses: pd.DataFrame,
    min_visits: int = 2,
) -> Dict[int, List[dict]]:
    """
    Build chronological visit sequences for each patient.

    Each visit contains ICD codes ordered by seq_num (priority order).
    This mirrors the "serialization" concept from Med-BERT: the most
    important diagnoses appear first.

    Args:
        admissions: from load_mimic_data()
        diagnoses:  from load_mimic_data()
        min_visits: minimum number of visits to include a patient

    Returns:
        patient_visits: subject_id → list of visit dicts (chronological)
    """
    print("Building patient visit sequences …")

    # ① Merge admit time into diagnoses
    diag_merged = diagnoses.merge(
        admissions[["subject_id", "hadm_id", "admittime"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )

    # ② Sort by (patient, admit_time, seq_num) so codes within each visit
    #    are ordered by clinical priority (seq_num=1 is primary diagnosis)
    diag_merged = diag_merged.sort_values(
        ["subject_id", "admittime", "seq_num"]
    )

    # ③ Collect codes per (subject_id, hadm_id), hospital admission id
    visit_codes: Dict[Tuple[int, int], List[Tuple[str, int]]] = defaultdict(list)
    for row in diag_merged.itertuples(index=False):
        key = (row.subject_id, row.hadm_id)
        visit_codes[key].append((row.icd_code, int(row.icd_version)))

    # ④ Build chronological visit list per patient
    adm_sorted = admissions.sort_values(["subject_id", "admittime"])

    patient_visits: Dict[int, List[dict]] = defaultdict(list)
    for row in adm_sorted.itertuples(index=False):
        codes = visit_codes.get((row.subject_id, row.hadm_id), [])
        if not codes:
            continue  # skip visits with no diagnoses
        patient_visits[int(row.subject_id)].append({
            "hadm_id":   int(row.hadm_id),
            "admittime": row.admittime,
            "codes":     codes,
        })

    # ⑤ Filter by minimum visits
    filtered = {
        pid: visits
        for pid, visits in patient_visits.items()
        if len(visits) >= min_visits
    }

    n_codes = sum(
        len(v["codes"])
        for visits in filtered.values()
        for v in visits
    )
    avg_visits = np.mean([len(v) for v in filtered.values()])
    avg_codes = n_codes / max(len(filtered), 1)

    print(
        f"  {len(filtered):,} patients with ≥{min_visits} visits "
        f"| avg {avg_visits:.1f} visits/patient "
        f"| avg {avg_codes:.1f} codes/patient"
    )
    return dict(filtered)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def build_vocabulary(
    patient_visits: Dict[int, List[dict]],
) -> Tuple[Dict[Tuple[str, int], int], List[Tuple[str, int]]]:
    """
    Build a vocabulary of all (icd_code, icd_version) pairs.

    Returns:
        code_to_idx: (code, version) → integer index
        idx_to_code: list of (code, version) ordered by index
    """
    all_codes: Set[Tuple[str, int]] = set()
    for visits in patient_visits.values():
        for visit in visits:
            for pair in visit["codes"]:
                all_codes.add(pair)

    # Sort for deterministic ordering
    idx_to_code: List[Tuple[str, int]] = sorted(all_codes, key=lambda x: (x[1], x[0]))
    code_to_idx: Dict[Tuple[str, int], int] = {
        code: idx for idx, code in enumerate(idx_to_code)
    }

    print(f"Vocabulary size: {len(idx_to_code):,} unique ICD codes")
    return code_to_idx, idx_to_code


def extend_vocabulary(
    code_to_idx: Dict[Tuple[str, int], int],
    idx_to_code: List[Tuple[str, int]],
    new_patient_visits: Dict[int, List[dict]],
) -> Tuple[Dict[Tuple[str, int], int], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Extend existing vocabulary with codes from a new dataset (e.g., fine-tuning).

    This enables Sent-e-Med's key advantage: handling previously unseen
    medical concepts by computing SBERT embeddings for new codes.

    Returns:
        updated code_to_idx, updated idx_to_code, list of NEW codes added
    """
    new_codes = []
    for visits in new_patient_visits.values():
        for visit in visits:
            for pair in visit["codes"]:
                if pair not in code_to_idx:
                    new_codes.append(pair)

    new_codes = sorted(set(new_codes), key=lambda x: (x[1], x[0]))

    for pair in new_codes:
        idx = len(idx_to_code)
        idx_to_code.append(pair)
        code_to_idx[pair] = idx

    if new_codes:
        print(f"Extended vocabulary with {len(new_codes):,} new codes "
              f"→ total {len(idx_to_code):,}")
    return code_to_idx, idx_to_code, new_codes


# ─────────────────────────────────────────────────────────────────────────────
# 4. Patient split
# ─────────────────────────────────────────────────────────────────────────────

def split_patients(
    patient_ids: List[int],
    pretrain_split: float = 0.70,
    val_split: float = 0.20,
    test_split: float = 0.10,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Randomly split patient IDs into pretrain / val / test.

    Paper uses 7:2:1. Patients are exclusive across splits to prevent
    data leakage (Section 3.4).
    """
    assert abs(pretrain_split + val_split + test_split - 1.0) < 1e-6

    rng = np.random.RandomState(seed)
    ids = np.array(patient_ids, dtype=int)
    rng.shuffle(ids)

    n = len(ids)
    n_pretrain = int(n * pretrain_split)
    n_val = int(n * val_split)

    pretrain_ids = ids[:n_pretrain].tolist()
    val_ids = ids[n_pretrain: n_pretrain + n_val].tolist()
    test_ids = ids[n_pretrain + n_val:].tolist()

    print(
        f"Patient split — pretrain: {len(pretrain_ids):,} "
        f"| val: {len(val_ids):,} "
        f"| test: {len(test_ids):,}"
    )
    return pretrain_ids, val_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# 5. Condition-specific case/control dataset (Section 3.3)
# ─────────────────────────────────────────────────────────────────────────────

def create_condition_dataset(
    patient_visits: Dict[int, List[dict]],
    patient_ids: List[int],
    condition: str,
    seed: int = 42,
) -> List[dict]:
    """
    Create case/control patient groups for a given condition (Section 3.3).

    Case group (label=1):
        Patient has ≥1 visit with the target condition.
        The condition must NOT appear in the first visit (we need prior history).
        Target visit = first visit where condition is documented.
        Input = all visits BEFORE the condition visit (prospective prediction).
        The condition visit itself is EXCLUDED to prevent data leakage.

    Control group (label=0):
        Patient never has the target condition in any visit.
        Target visit = last visit.
        All visits are kept.

    Returns:
        list of sample dicts:
            {
                "patient_id": int,
                "visits":     [visit_dict, ...],   # truncated at target
                "label":      int (0 or 1),
            }
    """
    matcher = make_condition_matcher(condition)
    samples: List[dict] = []
    skipped = 0

    for pid in patient_ids:
        visits = patient_visits.get(pid)
        if not visits or len(visits) < 2:
            continue

        # Find the first visit where condition appears
        condition_visit_idx: Optional[int] = None
        for i, visit in enumerate(visits):
            for code, version in visit["codes"]:
                if matcher(code, version):
                    condition_visit_idx = i
                    break
            if condition_visit_idx is not None:
                break

        if condition_visit_idx is not None:
            # ── Case group ───────────────────────────────────────────────────
            if condition_visit_idx == 0:
                # Condition in FIRST visit → no prior history → discard
                # (matches Figure 2 flowchart: "1st visit has condition?" → Case[1]
                #  but the paper discards patients where visit_idx=0 so model can
                #  see at least one prior visit)
                skipped += 1
                continue
            # IMPORTANT: exclude the condition visit itself.
            # We only feed visits BEFORE the diagnosis to the model
            # (prospective risk prediction — predict the future from the past).
            # Including the condition visit would be data leakage because the
            # model could trivially detect condition codes and predict label=1.
            target_visits = visits[:condition_visit_idx]
            samples.append({
                "patient_id": pid,
                "visits":     target_visits,
                "label":      1,
            })
        else:
            # ── Control group ────────────────────────────────────────────────
            samples.append({
                "patient_id": pid,
                "visits":     visits,       # all visits; last is target
                "label":      0,
            })

    cases    = sum(1 for s in samples if s["label"] == 1)
    controls = sum(1 for s in samples if s["label"] == 0)
    print(
        f"{condition} dataset — "
        f"cases: {cases:,}, controls: {controls:,}, "
        f"skipped (condition at visit 0): {skipped:,}, "
        f"total: {len(samples):,}"
    )
    return samples


def split_finetune_samples(
    samples: List[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split fine-tuning samples into train / val / test.
    Stratified by label to keep class balance.
    """
    rng = np.random.RandomState(seed)

    cases    = [s for s in samples if s["label"] == 1]
    controls = [s for s in samples if s["label"] == 0]

    def _split(lst: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
        arr = list(lst)
        rng.shuffle(arr)
        n = len(arr)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        return arr[:n_train], arr[n_train: n_train + n_val], arr[n_train + n_val:]

    case_tr,  case_val,  case_te  = _split(cases)
    ctrl_tr,  ctrl_val,  ctrl_te  = _split(controls)

    train   = case_tr  + ctrl_tr
    val     = case_val + ctrl_val
    test    = case_te  + ctrl_te

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    print(
        f"  Fine-tune split — train: {len(train):,} "
        f"| val: {len(val):,} "
        f"| test: {len(test):,}"
    )
    return train, val, test
