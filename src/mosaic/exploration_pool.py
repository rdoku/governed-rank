"""Exploration pool (Pool C) for candidate recall.

Adds:
- Cold-start items (low impressions)
- Under-explored moments (especially when activation is low-confidence)

This ensures diversity and cold-start coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class ExplorationConfig:
    """Configuration for exploration pool."""

    # Exploration rate (fraction of H+C pools)
    explore_rate: float = 0.05

    # Cold-start thresholds
    cold_start_impressions: int = 100
    cold_start_fraction: float = 0.5  # Fraction of explore pool for cold-start

    # Under-explored moment sampling
    moment_explore_fraction: float = 0.5  # Fraction for under-explored moments
    moment_uncertainty_threshold: float = 0.35

    # Confidence tier overrides
    low_confidence_boost: float = 1.5  # Increase exploration when confidence is low


@dataclass
class ExplorationResult:
    """Result of exploration pool generation."""

    item_ids: List[int]
    provenance: Dict[int, str]  # item_id -> "cold_start" | "moment_explore"
    debug_info: Dict


def sample_cold_start_items(
    available_items: Set[int],
    impression_counts: Dict[int, int],
    n_items: int,
    threshold: int = 100,
    rng: Optional[np.random.Generator] = None,
    return_debug: bool = False,
) -> List[int] | Tuple[List[int], Dict]:
    """Sample cold-start items (low impression count).

    Args:
        available_items: Set of available item IDs
        impression_counts: item_id -> impression count
        n_items: Number of items to sample
        threshold: Impression count threshold for cold-start

    Returns:
        List of sampled cold-start item IDs
    """
    if not available_items or n_items <= 0:
        if return_debug:
            return [], {"method": "inverse_impressions", "candidates": 0}
        return []

    # Find cold-start items
    cold_start = [
        item for item in available_items
        if impression_counts.get(item, 0) < threshold
    ]

    if not cold_start:
        if return_debug:
            return [], {"method": "inverse_impressions", "candidates": 0}
        return []

    rng = rng or np.random.default_rng()

    # Sample with preference for lowest impression counts
    weights = np.array(
        [1.0 / (impression_counts.get(item, 0) + 1.0) for item in cold_start],
        dtype=np.float64,
    )
    weights /= weights.sum()

    n_sample = min(n_items, len(cold_start))
    indices = rng.choice(len(cold_start), size=n_sample, replace=False, p=weights)
    sampled = [cold_start[i] for i in indices]

    if return_debug:
        sampled_probs = {
            int(cold_start[i]): round(float(weights[i]), 6) for i in indices
        }
        return sampled, {
            "method": "inverse_impressions",
            "candidates": len(cold_start),
            "threshold": threshold,
            "sampled_probabilities": sampled_probs,
        }

    return sampled


def sample_under_explored_moments(
    available_items: Set[int],
    moment2vec: np.ndarray,
    activation: np.ndarray,
    n_items: int,
    exclude_items: Set[int] = None,
    rng: Optional[np.random.Generator] = None,
    confidence_tier: str = "medium",
    uncertainty_threshold: float = 0.35,
    return_debug: bool = False,
) -> List[int] | Tuple[List[int], Dict]:
    """Sample items from under-explored moments.

    Under-explored = moments with low activation but still present in catalog.
    This helps discover items from moments we're not confident about.

    Args:
        available_items: Set of available item IDs
        moment2vec: (N, K) affinity matrix
        activation: (K,) moment activation distribution
        n_items: Number of items to sample
        exclude_items: Items to exclude

    Returns:
        List of sampled item IDs
    """
    if not available_items or n_items <= 0:
        if return_debug:
            return [], {"method": "score_proportional", "candidates": 0}
        return []

    exclude_items = exclude_items or set()
    eligible = available_items - exclude_items

    if not eligible:
        if return_debug:
            return [], {"method": "score_proportional", "candidates": 0}
        return []

    rng = rng or np.random.default_rng()
    K = moment2vec.shape[1]

    # Find under-explored moments (low activation)
    sorted_moments = np.argsort(activation)  # Ascending order
    under_explored = sorted_moments[:max(1, K // 3)]  # Bottom third

    p_safe = np.clip(activation, 1e-10, 1.0)
    entropy = -np.sum(p_safe * np.log(p_safe))
    peakedness = 1.0 - (entropy / np.log(K)) if K > 1 else 1.0
    if peakedness > uncertainty_threshold and confidence_tier == "high":
        if return_debug:
            return [], {
                "method": "score_proportional",
                "candidates": len(eligible),
                "skipped": True,
                "reason": "high_confidence_activation",
                "peakedness": round(float(peakedness), 4),
            }
        return []

    # Sample items that are strong in under-explored moments
    scores = np.zeros(moment2vec.shape[0])
    for m in under_explored:
        # Weight by inverse of activation (lower activation = more exploration value)
        weight = 1.0 / (activation[m] + 0.01)
        scores += weight * moment2vec[:, m]

    # Filter to eligible items
    eligible_list = list(eligible)
    eligible_scores = scores[eligible_list]

    if eligible_scores.size == 0:
        if return_debug:
            return [], {"method": "score_proportional", "candidates": 0}
        return []

    # Sample from top-M with probability proportional to score
    top_m = min(len(eligible_list), max(50, n_items * 5))
    top_indices = np.argsort(-eligible_scores)[:top_m]
    top_items = [eligible_list[i] for i in top_indices]
    top_scores = eligible_scores[top_indices]
    top_scores = np.clip(top_scores, 0.0, None)
    if top_scores.sum() == 0:
        if return_debug:
            return [], {"method": "score_proportional", "candidates": len(top_items)}
        return []

    probs = top_scores / top_scores.sum()
    n_sample = min(n_items, len(top_items))
    sampled = rng.choice(len(top_items), size=n_sample, replace=False, p=probs)

    sampled_items = [top_items[i] for i in sampled]

    if return_debug:
        sampled_probs = {
            int(top_items[i]): round(float(probs[i]), 6) for i in sampled
        }
        return sampled_items, {
            "method": "score_proportional",
            "candidates": len(eligible_list),
            "top_m": top_m,
            "sampled_probabilities": sampled_probs,
        }

    return sampled_items


def generate_exploration_pool(
    history_pool: List[int],
    context_pool: List[int],
    available_items: Set[int],
    moment2vec: np.ndarray,
    activation: np.ndarray,
    confidence_tier: str = "medium",
    impression_counts: Optional[Dict[int, int]] = None,
    rng: Optional[np.random.Generator] = None,
    config: ExplorationConfig = None,
) -> ExplorationResult:
    """Generate exploration pool (Pool C).

    Pool C = explore_rate * (|H_pool| + |C_pool|)

    Split between:
    - Cold-start items (low impressions)
    - Under-explored moments (low activation)

    Args:
        history_pool: Items from history pool
        context_pool: Items from context pool
        available_items: All available item IDs
        moment2vec: (N, K) affinity matrix
        activation: (K,) moment activation distribution
        confidence_tier: "high", "medium", or "low"
        impression_counts: item_id -> impression count
        config: Exploration config

    Returns:
        ExplorationResult with items and provenance
    """
    config = config or ExplorationConfig()
    impression_counts = impression_counts or {}
    rng = rng or np.random.default_rng()

    # Compute pool size
    base_size = len(history_pool) + len(context_pool)
    explore_rate = config.explore_rate

    # Boost exploration when confidence is low
    if confidence_tier == "low":
        explore_rate *= config.low_confidence_boost
    elif confidence_tier == "medium":
        explore_rate *= 1.2

    pool_size = max(1, int(explore_rate * base_size))

    # Items to exclude (already in H or C pools)
    exclude = set(history_pool) | set(context_pool)
    available_for_explore = available_items - exclude

    # Split between cold-start and moment exploration
    n_cold_start = max(1, int(pool_size * config.cold_start_fraction))
    n_moment_explore = pool_size - n_cold_start

    # Sample cold-start items
    cold_start_items, cold_debug = sample_cold_start_items(
        available_for_explore,
        impression_counts,
        n_cold_start,
        config.cold_start_impressions,
        rng,
        return_debug=True,
    )

    # Sample under-explored moment items
    exclude_after_cold = exclude | set(cold_start_items)
    moment_items, moment_debug = sample_under_explored_moments(
        available_for_explore,
        moment2vec,
        activation,
        n_moment_explore,
        exclude_after_cold,
        rng=rng,
        confidence_tier=confidence_tier,
        uncertainty_threshold=config.moment_uncertainty_threshold,
        return_debug=True,
    )

    # Build result
    items = cold_start_items + moment_items
    provenance = {}

    for item in cold_start_items:
        provenance[item] = "cold_start"
    for item in moment_items:
        provenance[item] = "moment_explore"

    debug_info = {
        "pool_size": len(items),
        "explore_rate": round(explore_rate, 4),
        "n_cold_start": len(cold_start_items),
        "n_moment_explore": len(moment_items),
        "confidence_tier": confidence_tier,
        "under_explored_moments": [
            int(m) for m in np.argsort(activation)[:max(1, len(activation) // 3)]
        ],
        "cold_start_sampling": cold_debug,
        "moment_explore_sampling": moment_debug,
    }

    return ExplorationResult(
        item_ids=items,
        provenance=provenance,
        debug_info=debug_info,
    )


def merge_pools_with_provenance(
    history_pool: List[int],
    context_pool: List[int],
    exploration_pool: ExplorationResult,
    cart_items: List[int],
) -> Tuple[List[int], Dict[int, Set[str]]]:
    """Merge all pools with provenance tracking.

    Args:
        history_pool: Items from history pool
        context_pool: Items from context pool
        exploration_pool: Exploration pool result
        cart_items: Items already in cart (to exclude)

    Returns:
        Tuple of (merged_items, provenance) where:
        - merged_items: Deduplicated list of item IDs
        - provenance: item_id -> set of sources ("history", "context", "explore")
    """
    cart_set = set(cart_items)
    provenance: Dict[int, Set[str]] = {}

    # Add history pool
    for item in history_pool:
        if item not in cart_set:
            if item not in provenance:
                provenance[item] = set()
            provenance[item].add("history")

    # Add context pool
    for item in context_pool:
        if item not in cart_set:
            if item not in provenance:
                provenance[item] = set()
            provenance[item].add("context")

    # Add exploration pool
    for item in exploration_pool.item_ids:
        if item not in cart_set:
            if item not in provenance:
                provenance[item] = set()
            provenance[item].add("explore")

    # Merged list (order: history first, then context, then explore)
    seen = set()
    merged = []

    for item in history_pool:
        if item not in cart_set and item not in seen:
            merged.append(item)
            seen.add(item)

    for item in context_pool:
        if item not in cart_set and item not in seen:
            merged.append(item)
            seen.add(item)

    for item in exploration_pool.item_ids:
        if item not in cart_set and item not in seen:
            merged.append(item)
            seen.add(item)

    return merged, provenance


def provenance_to_receipt(provenance: Dict[int, Set[str]], top_n: int = 20) -> Dict:
    """Convert provenance to receipt-friendly summary."""
    n_history = sum(1 for sources in provenance.values() if "history" in sources)
    n_context = sum(1 for sources in provenance.values() if "context" in sources)
    n_explore = sum(1 for sources in provenance.values() if "explore" in sources)
    n_multi = sum(1 for sources in provenance.values() if len(sources) > 1)

    # Top items with their sources
    top_items = []
    for item, sources in list(provenance.items())[:top_n]:
        top_items.append({
            "item": item,
            "sources": list(sources),
        })

    return {
        "total_candidates": len(provenance),
        "from_history": n_history,
        "from_context": n_context,
        "from_explore": n_explore,
        "multi_source": n_multi,
        "top_items": top_items,
    }
