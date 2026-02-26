"""Legacy rank protection gate for moment scoring.

This is a position-based smoothstep gate (SRG-CC) intended as a fallback or
baseline when calibrated MOSAIC constraints are unavailable. MOSAIC's primary
protection mechanism should be gap-based constraints + isotonic projection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class RankProtectionConfig:
    """Configuration for rank protection gate."""

    # Ranks 1..decay_start get weight = 0 (fully protected)
    decay_start: int = 5

    # Ranks decay_start..decay_end get smoothstep weight 0→1
    decay_end: int = 12

    # Mission boost cap (never exceed this magnitude)
    mission_cap: float = 0.04

    # Confidence tier weights
    confidence_weights: dict = None

    def __post_init__(self):
        if self.confidence_weights is None:
            self.confidence_weights = {
                "high": 1.0,
                "medium": 0.6,
                "low": 0.0,  # Don't steer when guessing
            }


def smoothstep(x: float) -> float:
    """Smoothstep interpolation: S(x) = x^2 * (3 - 2x).

    Maps [0, 1] → [0, 1] with smooth derivatives at endpoints.
    """
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def compute_rank_weights(
    n_items: int,
    config: RankProtectionConfig = None,
) -> np.ndarray:
    """Compute rank protection weights for positions 1..n_items.

    Legacy helper. Prefer per-item weights from apply_rank_protection() in
    production paths to avoid confusion about rank indexing.

    Args:
        n_items: Number of items
        config: Rank protection config

    Returns:
        (n_items,) array of weights in [0, 1]
    """
    config = config or RankProtectionConfig()

    weights = np.zeros(n_items, dtype=np.float32)

    for rank in range(1, n_items + 1):  # 1-indexed rank
        idx = rank - 1  # 0-indexed array position

        if rank <= config.decay_start:
            # Fully protected
            weights[idx] = 0.0
        elif rank >= config.decay_end:
            # Fully exposed to steering
            weights[idx] = 1.0
        else:
            # Smoothstep transition
            t = (rank - config.decay_start) / (config.decay_end - config.decay_start)
            weights[idx] = smoothstep(t)

    return weights


def apply_rank_protection(
    base_ranks: np.ndarray,
    mission_boosts_raw: np.ndarray,
    confidence_tier: str = "medium",
    config: RankProtectionConfig = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply rank protection gate to mission boosts.

    Args:
        base_ranks: (N,) base ranks from model score (1-indexed, lower = better)
        mission_boosts_raw: (N,) raw mission boost values
        confidence_tier: "high", "medium", or "low"
        config: Rank protection config

    Returns:
        Tuple of (gated_boosts, rank_weights) where:
        - gated_boosts: mission boosts after rank protection and capping
        - rank_weights: the rank-based weights applied
    """
    config = config or RankProtectionConfig()
    N = len(mission_boosts_raw)

    # Get confidence weight
    conf_weight = config.confidence_weights.get(confidence_tier, 0.5)

    # Compute rank weights
    rank_weights = np.zeros(N, dtype=np.float32)
    for i, rank in enumerate(base_ranks):
        rank = int(rank)
        if rank <= config.decay_start:
            rank_weights[i] = 0.0
        elif rank >= config.decay_end:
            rank_weights[i] = 1.0
        else:
            t = (rank - config.decay_start) / (config.decay_end - config.decay_start)
            rank_weights[i] = smoothstep(t)

    # Apply gates
    gated = mission_boosts_raw * conf_weight * rank_weights

    # Apply cap
    gated = np.clip(gated, -config.mission_cap, config.mission_cap)

    return gated, rank_weights


def get_base_ranks(model_scores: np.ndarray) -> np.ndarray:
    """Convert model scores to ranks (1-indexed, higher score = lower rank).

    Args:
        model_scores: (N,) model scores

    Returns:
        (N,) ranks where rank 1 = highest score
    """
    # argsort gives indices that would sort ascending
    # We want descending (highest score = rank 1)
    order = np.argsort(-model_scores)

    ranks = np.zeros(len(model_scores), dtype=np.int32)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank

    return ranks


def compute_protected_mission_boost(
    item_idx: int,
    model_scores: np.ndarray,
    mission_alignment: float,
    satiation_factor: float,
    confidence_tier: str = "medium",
    mission_lambda: float = 0.08,
    config: RankProtectionConfig = None,
) -> Tuple[float, dict]:
    """Compute protected mission boost for a single item.

    Full formula:
        S_mission_raw = λ_m * align(i) * sat(i)
        rank_weight = smoothstep((rank - decay_start) / (decay_end - decay_start))
        conf_weight = weight[confidence_tier]
        S_mission = clamp(S_mission_raw * conf_weight * rank_weight, [-cap, +cap])

    Args:
        item_idx: Index of item in model_scores array
        model_scores: (N,) all model scores (to compute ranks)
        mission_alignment: align(i) = A[i] · p
        satiation_factor: sat(i) from satiation module
        confidence_tier: "high", "medium", or "low"
        mission_lambda: λ_m scaling factor
        config: Rank protection config

    Returns:
        Tuple of (S_mission, debug_info)
    """
    config = config or RankProtectionConfig()

    # Compute raw mission boost
    S_mission_raw = mission_lambda * mission_alignment * satiation_factor

    # Get rank for this item
    ranks = get_base_ranks(model_scores)
    rank = int(ranks[item_idx])

    # Compute rank weight
    if rank <= config.decay_start:
        rank_weight = 0.0
    elif rank >= config.decay_end:
        rank_weight = 1.0
    else:
        t = (rank - config.decay_start) / (config.decay_end - config.decay_start)
        rank_weight = smoothstep(t)

    # Get confidence weight
    conf_weight = config.confidence_weights.get(confidence_tier, 0.5)

    # Apply gates
    S_mission = S_mission_raw * conf_weight * rank_weight

    # Apply cap
    S_mission = max(-config.mission_cap, min(config.mission_cap, S_mission))

    debug_info = {
        "S_mission_raw": round(S_mission_raw, 6),
        "base_rank": rank,
        "rank_weight": round(rank_weight, 4),
        "conf_weight": conf_weight,
        "confidence_tier": confidence_tier,
        "S_mission": round(S_mission, 6),
        "was_capped": abs(S_mission_raw * conf_weight * rank_weight) > config.mission_cap,
    }

    return S_mission, debug_info


def batch_compute_protected_mission_boost(
    model_scores: np.ndarray,
    mission_alignments: np.ndarray,
    satiation_factors: np.ndarray,
    confidence_tier: str = "medium",
    mission_lambda: float = 0.08,
    config: RankProtectionConfig = None,
) -> Tuple[np.ndarray, dict]:
    """Batch compute protected mission boosts for all items.

    Args:
        model_scores: (N,) model scores
        mission_alignments: (N,) align(i) = A[i] · p
        satiation_factors: (N,) sat(i) values
        confidence_tier: "high", "medium", or "low"
        mission_lambda: λ_m scaling factor
        config: Rank protection config

    Returns:
        Tuple of (S_mission array, summary debug info)
    """
    config = config or RankProtectionConfig()
    N = len(model_scores)

    # Compute raw mission boosts
    S_mission_raw = mission_lambda * mission_alignments * satiation_factors

    # Get ranks
    ranks = get_base_ranks(model_scores)

    # Compute rank weights
    rank_weights = np.zeros(N, dtype=np.float32)
    for i, rank in enumerate(ranks):
        if rank <= config.decay_start:
            rank_weights[i] = 0.0
        elif rank >= config.decay_end:
            rank_weights[i] = 1.0
        else:
            t = (rank - config.decay_start) / (config.decay_end - config.decay_start)
            rank_weights[i] = smoothstep(t)

    # Get confidence weight
    conf_weight = config.confidence_weights.get(confidence_tier, 0.5)

    # Apply gates
    S_mission = S_mission_raw * conf_weight * rank_weights

    # Apply cap
    S_mission = np.clip(S_mission, -config.mission_cap, config.mission_cap)

    # Summary stats
    n_protected = int(np.sum(ranks <= config.decay_start))
    n_partial = int(np.sum((ranks > config.decay_start) & (ranks < config.decay_end)))
    n_exposed = int(np.sum(ranks >= config.decay_end))

    debug_info = {
        "confidence_tier": confidence_tier,
        "conf_weight": conf_weight,
        "decay_start": config.decay_start,
        "decay_end": config.decay_end,
        "n_protected": n_protected,
        "n_partial": n_partial,
        "n_exposed": n_exposed,
        "max_boost": float(np.max(S_mission)),
        "min_boost": float(np.min(S_mission)),
        "mean_boost": float(np.mean(S_mission)),
    }

    return S_mission, debug_info
