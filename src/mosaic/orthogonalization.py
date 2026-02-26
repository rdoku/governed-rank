"""
MOSAIC Orthogonalization: Interference removal between base scores and steering utility.

This module implements Stage E of the MOSAIC algorithm - removing the component
of the steering utility that aligns with the base score direction.

Key insight: Naively adding steering utility u to base scores s can "pollute"
the accuracy signal. Orthogonalization ensures the steering signal is uncorrelated
with the base score direction on the candidate set.

Math:
    s~ = s - mean(s)  # center base scores
    u~ = u - mean(u)  # center steering utilities

    # Project out the component aligned with base scores
    u_perp = u~ - (s~.T @ u~) / (s~.T @ s~ + eps) * s~

    # Property: Cov(s~, u_perp) = 0 over the candidate set

References:
    - MOSAIC_paper.txt, Stage E
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class OrthogonalizationResult:
    """Result of orthogonalizing steering utility against base scores."""
    u_perp: Dict[int, float]  # Orthogonalized steering utility per item
    projection_coeff: float    # How much of u was aligned with s (for diagnostics)
    u_magnitude_before: float  # RMS of u before orthogonalization
    u_magnitude_after: float   # RMS of u_perp after orthogonalization
    corr_before: float         # Corr(s, u) before orthogonalization
    corr_after: float          # Corr(s, u_perp) after orthogonalization


def orthogonalize_against_base(
    base_scores: Dict[int, float],
    steering_utilities: Dict[int, float],
    eps: float = 1e-8,
    rescale: bool = False,
    target_rms: float = None,
) -> OrthogonalizationResult:
    """
    Remove the component of steering utility that aligns with base scores.

    This ensures the steering signal doesn't simply amplify/reduce what the
    base ranker already believes - it can only move items orthogonally to
    the base score direction.

    Args:
        base_scores: Dict mapping item_id -> base relevance score
        steering_utilities: Dict mapping item_id -> raw steering utility (mission + policy)
        eps: Small constant for numerical stability
        rescale: If True, rescale u_perp to maintain original RMS magnitude
        target_rms: If provided, rescale u_perp to this target RMS (overrides rescale)

    Returns:
        OrthogonalizationResult with orthogonalized utilities and diagnostics
    """
    # Get common items (should be all candidates)
    items = list(base_scores.keys())
    n = len(items)

    if n == 0:
        return OrthogonalizationResult(
            u_perp={},
            projection_coeff=0.0,
            u_magnitude_before=0.0,
            u_magnitude_after=0.0,
            corr_before=0.0,
            corr_after=0.0,
        )

    # Convert to arrays
    s = np.array([base_scores[i] for i in items])
    u = np.array([steering_utilities.get(i, 0.0) for i in items])

    # Center both vectors
    s_centered = s - np.mean(s)
    u_centered = u - np.mean(u)

    # Compute projection coefficient: how much of u is aligned with s
    s_norm_sq = np.dot(s_centered, s_centered) + eps
    proj_coeff = np.dot(s_centered, u_centered) / s_norm_sq

    # Remove the aligned component
    u_perp_arr = u_centered - proj_coeff * s_centered

    # Compute magnitudes for diagnostics
    u_rms_before = np.sqrt(np.mean(u_centered ** 2))
    u_rms_after = np.sqrt(np.mean(u_perp_arr ** 2))
    s_rms = np.sqrt(np.mean(s_centered ** 2))

    corr_before = 0.0
    corr_after = 0.0
    if s_rms > eps and u_rms_before > eps:
        corr_before = float(np.dot(s_centered, u_centered) / (n * s_rms * u_rms_before))
        corr_before = max(-1.0, min(1.0, corr_before))
    if s_rms > eps and u_rms_after > eps:
        corr_after = float(np.dot(s_centered, u_perp_arr) / (n * s_rms * u_rms_after))
        corr_after = max(-1.0, min(1.0, corr_after))

    # Optional rescaling
    if target_rms is not None and u_rms_after > eps:
        u_perp_arr = u_perp_arr * (target_rms / u_rms_after)
        u_rms_after = target_rms
    elif rescale and u_rms_after > eps and u_rms_before > eps:
        u_perp_arr = u_perp_arr * (u_rms_before / u_rms_after)
        u_rms_after = u_rms_before

    # Convert back to dict
    u_perp = {items[i]: float(u_perp_arr[i]) for i in range(n)}

    return OrthogonalizationResult(
        u_perp=u_perp,
        projection_coeff=float(proj_coeff),
        u_magnitude_before=float(u_rms_before),
        u_magnitude_after=float(u_rms_after),
        corr_before=corr_before,
        corr_after=corr_after,
    )


def compute_target_scores(
    base_scores: Dict[int, float],
    u_perp: Dict[int, float],
) -> Dict[int, float]:
    """
    Compute tentative target scores: t_i = s_i + u_perp_i

    These are the scores we'd use if we didn't have any constraints.
    The constraint projection stage will adjust these to preserve
    confident base decisions.

    Args:
        base_scores: Dict mapping item_id -> base relevance score
        u_perp: Dict mapping item_id -> orthogonalized steering utility

    Returns:
        Dict mapping item_id -> target score
    """
    return {
        item_id: base_scores[item_id] + u_perp.get(item_id, 0.0)
        for item_id in base_scores
    }
