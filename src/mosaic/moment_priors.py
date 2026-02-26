"""Learned-moment priors for Mission OS v0.

Includes population, cart, and history priors for learned moment spaces.
Catalog-family bootstrap lives in mosaic/core/family_bootstrap_priors.py.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# Population Prior P_m (Cold-Start Fallback)
# =============================================================================

def compute_population_prior(
    orders: List[List[int]],
    moment2vec: np.ndarray,
    min_basket_size: int = 2,
) -> np.ndarray:
    """Compute population-level moment prior from historical orders.

    This is the stable "population" prior used for cold start when we have
    no cart or history signal.

    Algorithm:
        For each order:
            order_profile = mean(A[item] for item in order)
        P_m = normalize(mean(order_profiles))

    Args:
        orders: List of orders, each order is a list of item_ids
        moment2vec: (N_items, K) affinity matrix A[item, moment]
        min_basket_size: Minimum items per order to include

    Returns:
        (K,) normalized population prior
    """
    K = moment2vec.shape[1]
    N = moment2vec.shape[0]

    if not orders:
        # Uniform fallback
        return np.ones(K, dtype=np.float32) / K

    profiles = []
    for order in orders:
        # Filter valid items
        valid_items = [i for i in order if 0 <= i < N]
        if len(valid_items) < min_basket_size:
            continue

        # Order profile = mean of item affinities
        order_affinities = moment2vec[valid_items]  # (n_items, K)
        order_profile = np.mean(order_affinities, axis=0)  # (K,)
        profiles.append(order_profile)

    if not profiles:
        return np.ones(K, dtype=np.float32) / K

    # Aggregate across orders
    stacked = np.stack(profiles, axis=0)  # (n_orders, K)
    P_m = np.mean(stacked, axis=0)  # (K,)

    # Normalize to sum to 1
    total = P_m.sum()
    if total > 0:
        P_m /= total
    else:
        P_m = np.ones(K, dtype=np.float32) / K

    return P_m.astype(np.float32)


def save_population_prior(
    P_m: np.ndarray,
    artifact_dir: str,
    moment_space_id: str,
) -> str:
    """Save population prior to disk.

    Args:
        P_m: (K,) population prior
        artifact_dir: Directory to save to
        moment_space_id: Version identifier

    Returns:
        Path to saved file
    """
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, f"population_prior_{moment_space_id}.npy")
    np.save(path, P_m)
    return path


def load_population_prior(
    artifact_dir: str,
    moment_space_id: str,
    K: int,
) -> np.ndarray:
    """Load population prior from disk, or return uniform if not found.

    Args:
        artifact_dir: Directory to load from
        moment_space_id: Version identifier
        K: Number of moments (for fallback)

    Returns:
        (K,) population prior
    """
    path = os.path.join(artifact_dir, f"population_prior_{moment_space_id}.npy")
    if os.path.exists(path):
        try:
            P_m = np.load(path).astype(np.float32)
            if len(P_m) == K:
                return P_m
        except Exception:
            pass
    # Uniform fallback
    return np.ones(K, dtype=np.float32) / K


# =============================================================================
# Cart Prior
# =============================================================================

def compute_cart_prior(
    cart_items: List[int],
    moment2vec: np.ndarray,
) -> np.ndarray:
    """Compute moment prior from current cart.

    Cart prior = normalize(mean(A[i] for i in cart))

    Args:
        cart_items: List of item_ids in cart
        moment2vec: (N_items, K) affinity matrix

    Returns:
        (K,) normalized cart prior
    """
    K = moment2vec.shape[1]
    N = moment2vec.shape[0]

    if not cart_items:
        # No cart = uniform
        return np.ones(K, dtype=np.float32) / K

    valid_items = [i for i in cart_items if 0 <= i < N]
    if not valid_items:
        return np.ones(K, dtype=np.float32) / K

    cart_affinities = moment2vec[valid_items]  # (n_items, K)
    p_cart = np.mean(cart_affinities, axis=0)  # (K,)

    # Normalize
    total = p_cart.sum()
    if total > 0:
        p_cart /= total
    else:
        p_cart = np.ones(K, dtype=np.float32) / K

    return p_cart.astype(np.float32)


# =============================================================================
# History Prior (Recency-Weighted)
# =============================================================================

def compute_history_prior(
    history_items: List[int],
    moment2vec: np.ndarray,
    decay: float = 0.9,
    max_history_items: Optional[int] = None,
) -> np.ndarray:
    """Compute moment prior from purchase history with recency weighting.

    More recent items get higher weight:
        weight[j] = decay^(H - j - 1)  where H = len(history), j = 0..H-1

    History prior = normalize(sum(weight[j] * A[history[j]]))

    Args:
        history_items: List of item_ids, most recent LAST
        moment2vec: (N_items, K) affinity matrix
        decay: Weight decay factor (0.9 = recent items ~10x weight of oldest)
        max_history_items: Optional cap on history length (most recent items kept)

    Returns:
        (K,) normalized history prior
    """
    K = moment2vec.shape[1]
    N = moment2vec.shape[0]

    if not history_items:
        return np.ones(K, dtype=np.float32) / K

    if max_history_items and len(history_items) > max_history_items:
        history_items = history_items[-max_history_items:]

    valid_items = [(i, j) for j, i in enumerate(history_items) if 0 <= i < N]
    if not valid_items:
        return np.ones(K, dtype=np.float32) / K

    H = len(history_items)
    p_hist = np.zeros(K, dtype=np.float32)

    for item_id, pos in valid_items:
        # Weight: more recent = higher weight
        weight = decay ** (H - pos - 1)
        p_hist += weight * moment2vec[item_id]

    # Normalize
    total = p_hist.sum()
    if total > 0:
        p_hist /= total
    else:
        p_hist = np.ones(K, dtype=np.float32) / K

    return p_hist.astype(np.float32)


# =============================================================================
# Combined Prior (Priority: Cart > History > Population)
# =============================================================================

def compute_combined_prior(
    cart_items: List[int],
    history_items: List[int],
    moment2vec: np.ndarray,
    population_prior: Optional[np.ndarray] = None,
    cart_weight: float = 0.6,
    history_weight: float = 0.3,
    population_weight: float = 0.1,
    max_history_items: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    """Compute combined prior with priority fallback.

    Priority order:
    1. If cart is non-empty: cart-weighted blend
    2. Else if history is non-empty: history-weighted blend
    3. Else: population prior

    Args:
        cart_items: Current cart items
        history_items: Recent history items
        moment2vec: (N_items, K) affinity matrix
        population_prior: (K,) fallback prior (optional)
        cart_weight: Weight for cart prior when blending
        history_weight: Weight for history prior when blending
        population_weight: Weight for population prior when blending
        max_history_items: Optional cap on history length (most recent items kept)

    Returns:
        Tuple of (prior, source) where source is "cart", "history", or "population"
    """
    K = moment2vec.shape[1]

    # Compute individual priors
    p_cart = compute_cart_prior(cart_items, moment2vec) if cart_items else None
    p_hist = compute_history_prior(
        history_items,
        moment2vec,
        max_history_items=max_history_items,
    ) if history_items else None
    p_pop = population_prior if population_prior is not None else np.ones(K, dtype=np.float32) / K

    # Determine source and blend
    if cart_items:
        # Have cart: blend cart + history + population
        p_combined = np.zeros(K, dtype=np.float32)
        weights_used = 0.0

        p_combined += cart_weight * p_cart
        weights_used += cart_weight

        if history_items and p_hist is not None:
            p_combined += history_weight * p_hist
            weights_used += history_weight

        p_combined += population_weight * p_pop
        weights_used += population_weight

        # Renormalize
        if weights_used > 0:
            p_combined /= weights_used

        # Final normalization
        total = p_combined.sum()
        if total > 0:
            p_combined /= total

        return p_combined.astype(np.float32), "cart"

    elif history_items:
        # No cart, have history: blend history + population
        blend_ratio = 0.7  # 70% history, 30% population
        p_combined = blend_ratio * p_hist + (1 - blend_ratio) * p_pop

        total = p_combined.sum()
        if total > 0:
            p_combined /= total

        return p_combined.astype(np.float32), "history"

    else:
        # Cold start: use population prior
        return p_pop.astype(np.float32), "population"
