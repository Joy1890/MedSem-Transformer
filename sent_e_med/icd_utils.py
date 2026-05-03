"""
icd_utils.py
============
ICD code utilities:
  - Load text descriptions from MIMIC-IV d_icd_diagnoses.csv
  - Normalize ICD codes (remove dots, strip whitespace)
  - Condition code matchers for SUD, OUD, Diabetes (Table 3 of paper)

MIMIC-IV note:
  Codes in diagnoses_icd.csv are stored WITHOUT dots.
  e.g., ICD-10 "F10.13" is stored as "F1013"
        ICD-9  "304.00"  is stored as "30400"
  This module handles that format throughout.
"""

from pathlib import Path
from typing import Dict, Tuple, Callable
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Description loading
# ─────────────────────────────────────────────────────────────────────────────

def load_icd_descriptions(mimic_dir: str) -> Dict[Tuple[str, int], str]:
    """
    Load ICD code → text description mapping from MIMIC-IV.

    MIMIC-IV directory structure:
        mimic_dir/hosp/d_icd_diagnoses.csv   (preferred)
        mimic_dir/d_icd_diagnoses.csv         (fallback)

    Returns:
        dict: (icd_code_no_dots, icd_version) → long_title
        e.g., ("F1013", 10) → "Alcohol abuse with withdrawal, unspecified"
    """
    mimic_path = Path(mimic_dir)
    candidates = [
        mimic_path / "hosp" / "d_icd_diagnoses.csv",
        mimic_path / "d_icd_diagnoses.csv",
    ]
    diag_path = next((p for p in candidates if p.exists()), None)
    if diag_path is None:
        raise FileNotFoundError(
            f"d_icd_diagnoses.csv not found in {mimic_dir}. "
            "Expected at: hosp/d_icd_diagnoses.csv"
        )

    df = pd.read_csv(diag_path, dtype={"icd_code": str, "icd_version": int})
    df["icd_code"] = df["icd_code"].str.strip()
    df["long_title"] = df["long_title"].fillna("Unknown diagnosis")

    mapping: Dict[Tuple[str, int], str] = {}
    for _, row in df.iterrows():
        key = (row["icd_code"], int(row["icd_version"]))
        mapping[key] = row["long_title"]

    print(f"Loaded {len(mapping):,} ICD descriptions from {diag_path}")
    return mapping


def get_description(
    code: str,
    version: int,
    descriptions: Dict[Tuple[str, int], str],
) -> str:
    """
    Retrieve the text description for a given ICD code.
    Falls back to 'ICD code <code>' if not found.
    """
    desc = descriptions.get((code, version))
    if desc:
        return desc
    # Try with version flipped (9 ↔ 10) as a last resort
    other_version = 9 if version == 10 else 10
    desc = descriptions.get((code, other_version))
    return desc if desc else f"ICD code {code}"


# ─────────────────────────────────────────────────────────────────────────────
# ICD code normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_icd(code: str) -> str:
    """Remove dots, strip whitespace, uppercase — for consistent comparison."""
    return code.replace(".", "").strip().upper()


# ─────────────────────────────────────────────────────────────────────────────
# Condition matchers (Table 3 of paper)
# ─────────────────────────────────────────────────────────────────────────────
# All codes are in MIMIC-IV format (no dots).
# We match by checking string prefixes.

def _make_oud_matcher() -> Callable[[str, int], bool]:
    """
    OUD — Opioid Use Disorder
    ICD-10: F11.*
    ICD-9 : 304.00–304.03, 304.70–304.73, 305.50–305.53
    """
    icd9_prefixes = {
        "30400", "30401", "30402", "30403",   # 304.00–304.03
        "30470", "30471", "30472", "30473",   # 304.70–304.73
        "30550", "30551", "30552", "30553",   # 305.50–305.53
    }

    def matcher(code: str, version: int) -> bool:
        c = normalize_icd(code)
        if version == 10:
            return c.startswith("F11")
        else:
            return c[:5] in icd9_prefixes

    return matcher


def _make_sud_matcher() -> Callable[[str, int], bool]:
    """
    SUD — Substance Use Disorder
    ICD-10: F10–F19
    ICD-9 : 291, 292, 303.xx, 304.xx, 305.5x, 648.3x
    """
    icd10_prefixes = tuple(f"F{i:02d}" for i in range(10, 20))  # F10–F19

    def matcher(code: str, version: int) -> bool:
        c = normalize_icd(code)
        if version == 10:
            return c.startswith(icd10_prefixes)
        else:
            return (
                c.startswith("291")
                or c.startswith("292")
                or (c.startswith("303") and len(c) >= 5)   # 303.xx
                or (c.startswith("304") and len(c) >= 5)   # 304.xx
                or c.startswith("3055")                     # 305.5x
                or c.startswith("6483")                     # 648.3x
            )

    return matcher


def _make_diabetes_matcher() -> Callable[[str, int], bool]:
    """
    Diabetes
    ICD-10: E08–E13
    ICD-9 : 250.xx
    """
    icd10_prefixes = tuple(f"E{i:02d}" for i in range(8, 14))  # E08–E13

    def matcher(code: str, version: int) -> bool:
        c = normalize_icd(code)
        if version == 10:
            return c.startswith(icd10_prefixes)
        else:
            return c.startswith("250")

    return matcher


# Public API: call make_condition_matcher(condition) to get a matcher function
_MATCHERS = {
    "OUD": _make_oud_matcher,
    "SUD": _make_sud_matcher,
    "Diabetes": _make_diabetes_matcher,
}

SUPPORTED_CONDITIONS = list(_MATCHERS.keys())


def make_condition_matcher(condition: str) -> Callable[[str, int], bool]:
    """
    Return a function  matcher(icd_code: str, icd_version: int) -> bool
    that checks whether a given code matches the target condition.

    Args:
        condition: one of "OUD", "SUD", "Diabetes"
    """
    if condition not in _MATCHERS:
        raise ValueError(
            f"Unknown condition '{condition}'. Choose from {SUPPORTED_CONDITIONS}."
        )
    return _MATCHERS[condition]()
