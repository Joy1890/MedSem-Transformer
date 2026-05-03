"""
enrich_icd.py
=============
GPT-based enrichment of ICD code text descriptions.

The original MIMIC-IV descriptions from d_icd_diagnoses.csv are short titles
(e.g. "Alcohol dependence with unspecified withdrawal").  This module uses
GPT-4o-mini to generate richer 100-120 word clinical summaries that include
comorbidities, risk factors, and patient profiles — information that improves
the semantic structure of SBERT embeddings for clinical prediction tasks.

The enriched descriptions are a drop-in replacement for the original dict:
    {(icd_code, icd_version): text_string}

Caching:
    Enriched texts are saved to a JSON file (enriched_descriptions.json) on
    first run.  All subsequent runs load from cache with zero API calls.
    Cost is therefore a one-time ~$1-2 expense for ~17K MIMIC ICD codes.

Public API:
    load_or_build_enriched()   — main entry point used by main.py
    build_enriched_descriptions() — calls GPT for uncached codes, saves JSON
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a clinical knowledge encoder. "
    "Write concise, factual clinical summaries of ICD diagnosis codes "
    "that will be encoded into semantic vectors for medical machine learning."
)

_USER_TEMPLATE = """\
Write a clinical summary of the following ICD diagnosis in 100-120 words of \
coherent prose (not bullet points). Include in this order:
1. What the condition is and its defining clinical features
2. Typical symptoms and how it presents
3. Common comorbid conditions that frequently co-occur with this diagnosis
4. Typical patient population and key risk factors

ICD code: {code}
Standard description: {description}

Focus on information that distinguishes this condition from similar ones \
and captures its clinical relationships to other conditions."""


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_or_build_enriched(
    descriptions: Dict[Tuple[str, int], str],
    cache_path: str,
    model: str = "gpt-4o-mini",
    batch_size: int = 50,
    requests_per_minute: int = 500,
) -> Dict[Tuple[str, int], str]:
    """
    Return an enriched descriptions dict, loading from cache if available.

    If the cache file exists and covers all codes in `descriptions`, returns
    immediately with no API calls.  Otherwise calls GPT for uncached codes,
    merges results, and saves the updated cache.

    This means partial caches are safe — you can interrupt and resume without
    re-processing completed codes.

    Args:
        descriptions:         original MIMIC ICD descriptions dict
        cache_path:           path to save/load enriched_descriptions.json
        model:                OpenAI model name (default: gpt-4o-mini)
        batch_size:           codes processed per progress print
        requests_per_minute:  rate limit (gpt-4o-mini default tier: 500 RPM)

    Returns:
        enriched dict: same structure as input descriptions,
                       {(icd_code, icd_version): enriched_text}
    """
    cache_file = Path(cache_path)

    # ── Load existing cache ───────────────────────────────────────────────────
    cached: Dict[str, str] = {}
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        print(f"Loaded {len(cached):,} cached enriched descriptions from {cache_path}")

    # ── Find uncached codes ───────────────────────────────────────────────────
    # Cache keys are "{code}_{version}" strings (JSON keys must be strings)
    to_enrich: List[Tuple[str, int]] = [
        (code, version)
        for code, version in descriptions.keys()
        if f"{code}_{version}" not in cached
    ]

    if not to_enrich:
        print("All codes already cached — skipping GPT enrichment.")
    else:
        print(
            f"Enriching {len(to_enrich):,} uncached codes "
            f"using {model} …"
        )
        _estimate_cost(len(to_enrich), model)
        new_enriched = build_enriched_descriptions(
            to_enrich=to_enrich,
            descriptions=descriptions,
            model=model,
            batch_size=batch_size,
            requests_per_minute=requests_per_minute,
        )
        cached.update(new_enriched)

        # Save updated cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cached, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(cached):,} enriched descriptions to {cache_path}")

    # ── Reconstruct dict with original (code, version) tuple keys ────────────
    enriched: Dict[Tuple[str, int], str] = {}
    fallback = 0
    for code, version in descriptions.keys():
        key = f"{code}_{version}"
        if key in cached:
            enriched[(code, version)] = cached[key]
        else:
            # Fallback to original if GPT call failed for this code
            enriched[(code, version)] = descriptions[(code, version)]
            fallback += 1

    if fallback:
        print(f"  {fallback} codes fell back to original description (GPT call failed)")

    print(f"Enriched descriptions ready: {len(enriched):,} codes")
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# GPT enrichment
# ─────────────────────────────────────────────────────────────────────────────

def build_enriched_descriptions(
    to_enrich: List[Tuple[str, int]],
    descriptions: Dict[Tuple[str, int], str],
    model: str = "gpt-4o-mini",
    batch_size: int = 50,
    requests_per_minute: int = 500,
) -> Dict[str, str]:
    """
    Call GPT for each uncached ICD code and return {cache_key: enriched_text}.

    Rate limiting: inserts a small sleep between batches to stay within the
    requests-per-minute limit of the OpenAI API tier.

    Args:
        to_enrich:            list of (code, version) pairs to process
        descriptions:         original descriptions for prompt construction
        model:                OpenAI model name
        batch_size:           codes per progress checkpoint
        requests_per_minute:  used to compute inter-batch sleep

    Returns:
        dict: {"F1013_10": "enriched text ...", ...}
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package is required for ICD enrichment.\n"
            "Install with:  pip install openai"
        )

    client = OpenAI()   # reads OPENAI_API_KEY from environment
    results: Dict[str, str] = {}
    failed  = 0

    # Seconds to sleep between individual requests to respect RPM limit
    # Add a small safety margin (× 1.1)
    sleep_per_request = (60.0 / requests_per_minute) * 1.1

    for i, (code, version) in enumerate(to_enrich):
        original = descriptions.get((code, version), f"ICD code {code}")
        prompt   = _USER_TEMPLATE.format(code=code, description=original)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,      # deterministic — cache stays valid on re-run
                max_tokens=200,     # 100-120 words ≈ 130-150 tokens; 200 is safe ceiling
            )
            enriched_text = response.choices[0].message.content.strip()
            results[f"{code}_{version}"] = enriched_text

        except Exception as e:
            # On any API error, skip and fall back to original at reconstruction
            print(f"  WARNING: GPT call failed for {code} (v{version}): {e}")
            failed += 1

        # Progress checkpoint
        if (i + 1) % batch_size == 0:
            print(f"  {i + 1}/{len(to_enrich)} codes enriched …")

        # Rate limiting
        time.sleep(sleep_per_request)

    print(
        f"GPT enrichment complete: {len(results):,} succeeded, {failed} failed"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Cost estimator (informational only)
# ─────────────────────────────────────────────────────────────────────────────

# Approximate pricing per 1M tokens (update if OpenAI changes rates)
_PRICING = {
    "gpt-4o-mini": {"input": 0.15,  "output": 0.60},
    "gpt-4o":      {"input": 2.50,  "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
}

_AVG_INPUT_TOKENS  = 140   # prompt template + code + original description
_AVG_OUTPUT_TOKENS = 145   # ~110 words of enriched text


def _estimate_cost(n_codes: int, model: str) -> None:
    """Print an estimated API cost before making calls."""
    pricing = _PRICING.get(model)
    if pricing is None:
        print(f"  (No pricing info for model '{model}' — cost unknown)")
        return

    input_cost  = (n_codes * _AVG_INPUT_TOKENS  / 1_000_000) * pricing["input"]
    output_cost = (n_codes * _AVG_OUTPUT_TOKENS / 1_000_000) * pricing["output"]
    total       = input_cost + output_cost

    print(
        f"  Estimated cost for {n_codes:,} codes ({model}): "
        f"${input_cost:.2f} input + ${output_cost:.2f} output = ${total:.2f} total"
    )
    print(f"  This is a ONE-TIME cost — results will be cached to JSON.")
