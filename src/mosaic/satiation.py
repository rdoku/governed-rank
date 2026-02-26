"""Satiation curves for moment-based scoring.

Implements diminishing returns: after the cart already has items from a moment,
don't keep recommending more from that moment unless truly high value.

Key insight: "Once a mission is 'full', marginal utility drops."
This gives diversity without hacky penalties.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def compute_moment_fill(
    cart_affinities: List[np.ndarray],
    K: Optional[int] = None,
    normalize: str = "mean",
) -> np.ndarray:
    """Compute how "filled" each moment is based on cart contents.

    Args:
        cart_affinities: List of (K,) affinity vectors for items in cart
        normalize: How to normalize - "mean" (bounded), "sqrt" (scales with cart), "sum" (raw)

    Returns:
        (K,) fill vector, bounded [0, 1] for "mean" normalization
    """
    if not cart_affinities:
        if K is None:
            K = 0
        return np.zeros((K,), dtype=np.float32)

    stacked = np.stack(cart_affinities, axis=0)  # (cart_size, K)

    if normalize == "mean":
        # Bounded [0, 1] - best for stability
        fill = np.mean(stacked, axis=0)
    elif normalize == "sqrt":
        # Scales with sqrt(cart_size) - moderate growth
        fill = np.sum(stacked, axis=0) / np.sqrt(len(cart_affinities) + 1)
    else:  # "sum"
        # Raw sum - can explode with large carts
        fill = np.sum(stacked, axis=0)

    return fill.astype(np.float32)


def compute_satiation(
    fill: np.ndarray,
    activation: np.ndarray,
    rate: float = 0.6,
    top_m: int = 2,
    floor: float = 0.25,
) -> np.ndarray:
    """Compute satiation factors for each moment.

    Only satiates the top-M active moments to preserve shopping intent.
    This prevents "diversifying away" from what the shopper actually wants.

    Uses a floor to never fully zero out a moment - sometimes it IS correct
    to recommend the 5th brunch item (e.g., "butter" after bread/eggs).

    Args:
        fill: (K,) how filled each moment is
        activation: (K,) current moment activation (p vector)
        rate: Satiation rate - higher = faster diminishing returns
        top_m: Only apply satiation to top-M active moments
        floor: Minimum satiation factor (never go below this)

    Returns:
        (K,) satiation multipliers in [floor, 1]
    """
    K = len(activation)
    satiation = np.ones(K, dtype=np.float32)

    # Only satiate the most active moments
    active_idx = np.argsort(-activation)[:top_m]
    # sat = floor + (1 - floor) * exp(-rate * fill)
    # This ensures we never fully zero out a moment
    fill = np.clip(fill, 0.0, 1.0)
    raw_sat = np.exp(-rate * fill[active_idx])
    satiation[active_idx] = floor + (1.0 - floor) * raw_sat

    return satiation


def satiated_moment_score(
    item_affinity: np.ndarray,
    activation: np.ndarray,
    cart_affinities: Optional[List[np.ndarray]] = None,
    rate: float = 0.6,
    top_m: int = 2,
    floor: float = 0.25,
) -> float:
    """Compute moment alignment score with satiation applied.

    Note: Uses presence-only (not quantity). Each unique item in cart
    contributes once to fill, regardless of quantity.

    Args:
        item_affinity: (K,) affinity for candidate item (A[item])
        activation: (K,) moment activation (p)
        cart_affinities: List of (K,) affinities for items in cart (one per unique item)
        rate: Satiation rate
        top_m: Number of top moments to satiate
        floor: Minimum satiation factor

    Returns:
        Scalar score with satiation applied
    """
    if cart_affinities is None or len(cart_affinities) == 0:
        # No cart = no satiation (cold start bypass)
        return float(np.dot(item_affinity, activation))

    fill = compute_moment_fill(cart_affinities, K=len(activation), normalize="mean")
    satiation = compute_satiation(fill, activation, rate=rate, top_m=top_m, floor=floor)

    # score = sum(A[item] * p * satiation)
    return float(np.sum(item_affinity * activation * satiation))


def satiated_moment_scores_batch(
    item_affinities: np.ndarray,
    activation: np.ndarray,
    cart_affinities: Optional[List[np.ndarray]] = None,
    rate: float = 0.6,
    top_m: int = 2,
    floor: float = 0.25,
) -> np.ndarray:
    """Batch version: compute satiated scores for multiple candidates.

    Note: Uses presence-only (not quantity). Each unique item in cart
    contributes once to fill, regardless of quantity.

    Args:
        item_affinities: (N, K) affinity matrix for N candidate items
        activation: (K,) moment activation
        cart_affinities: List of (K,) affinities for items in cart (one per unique item)
        rate: Satiation rate
        top_m: Number of top moments to satiate
        floor: Minimum satiation factor

    Returns:
        (N,) scores with satiation applied
    """
    if cart_affinities is None or len(cart_affinities) == 0:
        # No satiation - standard dot product (cold start bypass)
        return item_affinities @ activation

    fill = compute_moment_fill(cart_affinities, K=len(activation), normalize="mean")
    satiation = compute_satiation(fill, activation, rate=rate, top_m=top_m, floor=floor)

    # Broadcast: (N, K) * (K,) * (K,) -> sum over K -> (N,)
    return np.sum(item_affinities * activation * satiation, axis=1)


def get_satiation_debug_info(
    activation: np.ndarray,
    cart_affinities: Optional[List[np.ndarray]] = None,
    rate: float = 0.6,
    top_m: int = 2,
    floor: float = 0.25,
) -> dict:
    """Get debug info for satiation state (for receipts/logging).

    Returns provable satiation info:
    - satiation_active_moments: which moments are being satiated
    - satiation_fill_topM: fill levels for active moments
    - satiation_factor_topM: resulting satiation factors

    This makes the "principled diversity" claim verifiable.
    """
    K = len(activation)
    active_idx = list(np.argsort(-activation)[:top_m])

    if cart_affinities is None or len(cart_affinities) == 0:
        return {
            "cart_size": 0,
            "satiation_active_moments": active_idx,
            "satiation_fill_topM": [0.0] * len(active_idx),
            "satiation_factor_topM": [1.0] * len(active_idx),
            "rate": rate,
            "floor": floor,
        }

    fill = compute_moment_fill(cart_affinities, K=len(activation), normalize="mean")
    satiation = compute_satiation(fill, activation, rate=rate, top_m=top_m, floor=floor)

    # Extract just the top-M values for compact receipts
    fill_topM = [round(float(fill[i]), 3) for i in active_idx]
    factor_topM = [round(float(satiation[i]), 3) for i in active_idx]

    return {
        "cart_size": len(cart_affinities),
        "satiation_active_moments": active_idx,
        "satiation_fill_topM": fill_topM,
        "satiation_factor_topM": factor_topM,
        "rate": rate,
        "floor": floor,
        "normalize": "mean",
    }
