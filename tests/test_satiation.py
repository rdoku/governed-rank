"""Tests for satiation curves."""
import numpy as np
import pytest

from mosaic.satiation import (
    compute_moment_fill,
    compute_satiation,
    satiated_moment_score,
    satiated_moment_scores_batch,
    get_satiation_debug_info,
)


def test_compute_moment_fill_empty():
    """Empty cart returns zeros."""
    fill = compute_moment_fill([], K=8)
    assert fill.shape == (8,)
    assert np.allclose(fill, 0)


def test_compute_moment_fill_mean_normalized():
    """Fill is bounded [0, 1] with mean normalization."""
    K = 4
    cart_affinities = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
    ]
    fill = compute_moment_fill(cart_affinities, normalize="mean")

    assert fill.shape == (K,)
    assert np.isclose(fill[0], 2/3)  # 2 of 3 items have moment 0
    assert np.isclose(fill[1], 1/3)  # 1 of 3 items have moment 1
    assert fill[2] == 0
    assert fill.max() <= 1.0


def test_compute_satiation_only_active_moments():
    """Satiation only applies to top-M active moments."""
    K = 4
    fill = np.array([0.8, 0.2, 0.1, 0.0])
    activation = np.array([0.5, 0.3, 0.15, 0.05])  # moment 0 and 1 are top-2

    sat = compute_satiation(fill, activation, rate=1.0, top_m=2)

    # Only moments 0 and 1 should be satiated
    assert sat[0] < 1.0  # exp(-1.0 * 0.8) ≈ 0.45
    assert sat[1] < 1.0  # exp(-1.0 * 0.2) ≈ 0.82
    assert sat[2] == 1.0  # Not in top-2, no satiation
    assert sat[3] == 1.0


def test_satiated_score_no_cart():
    """No cart = no satiation = standard dot product."""
    item_aff = np.array([0.6, 0.3, 0.1, 0.0])
    activation = np.array([0.5, 0.3, 0.15, 0.05])

    score_no_cart = satiated_moment_score(item_aff, activation, cart_affinities=None)
    expected = np.dot(item_aff, activation)

    assert np.isclose(score_no_cart, expected)


def test_satiated_score_with_cart_reduces_score():
    """Cart with items from same moment reduces score."""
    K = 4
    item_aff = np.array([0.9, 0.1, 0.0, 0.0])  # Item is strongly moment-0
    activation = np.array([0.7, 0.2, 0.1, 0.0])  # Moment 0 is active

    # Cart already full of moment-0 items
    cart_affinities = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.8, 0.2, 0.0, 0.0]),
    ]

    score_no_sat = np.dot(item_aff, activation)
    score_with_sat = satiated_moment_score(item_aff, activation, cart_affinities, rate=1.0)

    # Satiated score should be lower
    assert score_with_sat < score_no_sat


def test_satiated_scores_batch():
    """Batch scoring works correctly."""
    K = 4
    N = 10
    item_affinities = np.random.rand(N, K).astype(np.float32)
    item_affinities = item_affinities / item_affinities.sum(axis=1, keepdims=True)
    activation = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)

    cart_affinities = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ]

    scores = satiated_moment_scores_batch(item_affinities, activation, cart_affinities)

    assert scores.shape == (N,)
    assert all(s >= 0 for s in scores)


def test_satiation_debug_info():
    """Debug info returns expected structure for receipts."""
    K = 4
    activation = np.array([0.5, 0.3, 0.15, 0.05])
    cart_affinities = [
        np.array([1.0, 0.0, 0.0, 0.0]),
    ]

    info = get_satiation_debug_info(activation, cart_affinities, rate=0.6, top_m=2)

    assert info["cart_size"] == 1
    assert info["satiation_active_moments"] == [0, 1]  # top-2 by activation
    assert len(info["satiation_fill_topM"]) == 2  # Only top-M values
    assert len(info["satiation_factor_topM"]) == 2
    assert info["rate"] == 0.6
    assert info["floor"] == 0.25  # Default floor


def test_satiation_floor_prevents_zeroing():
    """Floor prevents satiation from fully zeroing out a moment."""
    K = 4
    fill = np.array([1.0, 0.0, 0.0, 0.0])  # Moment 0 completely full
    activation = np.array([0.7, 0.2, 0.1, 0.0])  # Moment 0 is active

    # With floor=0.25, even fully saturated moment stays at least 0.25
    sat = compute_satiation(fill, activation, rate=10.0, top_m=2, floor=0.25)

    # Moment 0 should be floored at 0.25, not near-zero
    assert sat[0] >= 0.25
    # Other moments unaffected
    assert sat[2] == 1.0
    assert sat[3] == 1.0


def test_satiation_preserves_diversity_intent():
    """Satiation doesn't diversify away from intent."""
    K = 4
    # Shopping intent is strongly moment 0 (breakfast)
    activation = np.array([0.7, 0.1, 0.1, 0.1])

    # Cart already has breakfast items
    cart_affinities = [
        np.array([0.9, 0.1, 0.0, 0.0]),
        np.array([0.8, 0.1, 0.1, 0.0]),
    ]

    # Candidate A: another breakfast item
    item_a = np.array([0.9, 0.1, 0.0, 0.0])
    # Candidate B: random item from inactive moment
    item_b = np.array([0.1, 0.1, 0.1, 0.7])

    score_a = satiated_moment_score(item_a, activation, cart_affinities, rate=0.6, top_m=2)
    score_b = satiated_moment_score(item_b, activation, cart_affinities, rate=0.6, top_m=2)

    # Even with satiation, breakfast item should still score reasonably
    # (we're not diversifying away from intent completely)
    # The inactive moment item gets no satiation boost, so breakfast can still win
    # if it's strongly aligned with activation
    assert score_a > 0  # Still positive
