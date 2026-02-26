"""Catalog-text bootstrap utilities for Mission OS families.

These helpers map items into the family taxonomy for cold-start bootstrapping.
They are not intended for learned-moment priors.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

import numpy as np

from mosaic.moments_taxonomy import MOMENT_FAMILIES, DEFAULT_FAMILY

_WORD_RE = re.compile(r"[^a-z0-9]+")

DEFAULT_FAMILY_WEIGHT = 0.4

KEYWORDS_BY_FAMILY: Dict[str, List[str]] = {
    "Breakfast/Brunch": [
        "breakfast", "brunch", "pancake", "waffle", "cereal", "oat", "bagel",
        "coffee", "tea", "orange juice", "bacon", "sausage", "eggs",
    ],
    "Weeknight Dinner": [
        "dinner", "pasta", "sauce", "rice", "chicken", "beef", "fish",
        "vegetable", "stir fry", "quick meal", "family meal",
    ],
    "Snacks/Party": [
        "snack", "chips", "salsa", "dip", "cracker", "cookie", "candy",
        "soda", "sparkling", "party", "treat",
    ],
    "Stock-up/Household": [
        "household", "paper", "towel", "toilet", "laundry", "clean",
        "bulk", "pantry", "pack", "stock",
    ],
    "Health/Wellness": [
        "health", "wellness", "vitamin", "organic", "keto", "gluten",
        "protein", "salad", "greens", "supplement",
    ],
    "Comfort/Sick Day": [
        "soup", "broth", "tea", "honey", "comfort", "cozy", "relief",
        "ginger", "cough", "chamomile",
    ],
    "Baking/Projects": [
        "baking", "flour", "sugar", "mix", "frosting", "yeast",
        "icing", "vanilla", "chocolate chips",
    ],
    "Quick Grab/Convenience": [
        "milk", "bread", "banana", "eggs", "staple", "restock",
        "everyday", "grab", "quick", "convenience", "single serve",
        "on the go",
    ],
    "Everyday Staples": [
        "staple", "everyday", "basic", "essentials", "pantry", "restock",
    ],
}

FIELD_WEIGHTS = {
    "product_type": 2.0,
    "tags": 1.5,
    "title": 1.0,
    "handle": 0.8,
    "vendor": 0.5,
    "collections": 1.2,
}


def _normalize_text(value: str) -> str:
    return _WORD_RE.sub(" ", value.lower()).strip()


def _score_text(text: str, keywords: Iterable[str], weight: float) -> float:
    if not text:
        return 0.0
    score = 0.0
    for kw in keywords:
        if kw in text:
            score += weight
    return score


def default_family_map(k: int, families_len: int) -> List[int]:
    if k <= 0 or families_len <= 0:
        return []
    if k <= families_len:
        return list(range(k))
    return [i % families_len for i in range(k)]


def compute_item_affinity(
    item: Dict[str, Optional[str]],
    *,
    families: Optional[List[str]] = None,
    keywords_by_family: Optional[Dict[str, List[str]]] = None,
) -> np.ndarray:
    families = families or MOMENT_FAMILIES
    keywords_by_family = keywords_by_family or KEYWORDS_BY_FAMILY
    scores = np.zeros(len(families), dtype=np.float32)

    if not families:
        return scores

    raw_fields = {
        "product_type": item.get("product_type") or "",
        "tags": item.get("tags") or "",
        "title": item.get("title") or "",
        "handle": item.get("handle") or "",
        "vendor": item.get("vendor") or "",
        "collections": item.get("collections") or "",
    }

    # Normalize all fields to lower-case keyword blobs
    normalized: Dict[str, str] = {}
    for key, value in raw_fields.items():
        if isinstance(value, list):
            joined = " ".join(str(v) for v in value)
            normalized[key] = _normalize_text(joined)
        else:
            normalized[key] = _normalize_text(str(value))

    for idx, family in enumerate(families):
        keywords = [kw.lower() for kw in keywords_by_family.get(family, [])]
        for field_name, text in normalized.items():
            weight = FIELD_WEIGHTS.get(field_name, 1.0)
            scores[idx] += _score_text(text, keywords, weight)

    # Smooth or fallback if empty
    total = float(scores.sum())
    if total <= 0.0:
        if DEFAULT_FAMILY in families:
            default_idx = families.index(DEFAULT_FAMILY)
            if len(families) == 1:
                scores[0] = 1.0
            else:
                residual = (1.0 - DEFAULT_FAMILY_WEIGHT) / (len(families) - 1)
                scores.fill(residual)
                scores[default_idx] = DEFAULT_FAMILY_WEIGHT
        else:
            scores = np.ones(len(families), dtype=np.float32)
        total = float(scores.sum())

    scores /= total
    return scores


def build_moment2vec_from_items(
    items: List[Dict[str, Optional[str]]],
    *,
    num_items: int,
    k: int,
    family_map: Optional[List[int]] = None,
) -> np.ndarray:
    if num_items <= 0 or k <= 0:
        return np.zeros((0, max(k, 0)), dtype=np.float32)

    families = MOMENT_FAMILIES
    family_map = family_map or default_family_map(k, len(families))
    moment2vec = np.zeros((num_items, k), dtype=np.float32)

    for item in items:
        model_id = item.get("model_id")
        if model_id is None:
            continue
        try:
            idx = int(model_id)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= num_items:
            continue
        fam_scores = compute_item_affinity(item, families=families)
        row = np.zeros(k, dtype=np.float32)
        for moment_idx, fam_idx in enumerate(family_map):
            if 0 <= fam_idx < len(fam_scores):
                row[moment_idx] += fam_scores[fam_idx]
        total = float(row.sum())
        if total <= 0.0:
            row[:] = 1.0 / max(1, k)
        else:
            row /= total
        moment2vec[idx] = row

    # Fill any untouched rows with a uniform prior
    missing = np.where(moment2vec.sum(axis=1) == 0.0)[0]
    if len(missing) > 0:
        moment2vec[missing] = 1.0 / max(1, k)

    return moment2vec
