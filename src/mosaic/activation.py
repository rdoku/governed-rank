"""Hybrid moment activation with rule posterior and model blending.

Implements the full activation pipeline:
1. Prior from items (cart/history/population)
2. Time multipliers
3. Evidence graph contribution
4. Rule posterior: p_rule ∝ p_prior · t · exp(β·e)
5. Model posterior (optional)
6. Interpretable blending gate α
7. Temperature softening
8. Confidence tiering
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np


@dataclass
class ActivationConfig:
    """Configuration for hybrid activation."""

    # Evidence graph weight
    beta: float = 1.0

    # Temperature softening
    T_max: float = 2.5
    gamma: float = 2.0

    # Blending gate thresholds
    ev_high: float = 0.6
    ev_low: float = 0.25
    agree_high: float = 0.7
    model_conf_high: float = 0.6
    history_strength_high: float = 0.5

    # Alpha values for different scenarios
    alpha_strong_rules: float = 0.35   # Strong evidence, high agreement
    alpha_weak_evidence: float = 0.75  # Weak evidence
    alpha_strong_model: float = 0.70   # Model confident, has history
    alpha_default: float = 0.50        # Default blend


@dataclass
class ActivationResult:
    """Result of moment activation."""

    # Final activation
    p: np.ndarray  # (K,) moment probabilities
    confidence_tier: str  # "high", "medium", "low"
    prior_source: str  # "cart", "history", or "population"

    # Diagnostics
    p_prior: np.ndarray
    p_rule: np.ndarray
    p_model: Optional[np.ndarray]
    p_mix: np.ndarray

    # Gate info
    alpha: float
    gate_reason: str
    evidence_volume: float
    agreement: float
    temperature: float

    # Feature info
    fired_features: FrozenSet[str]
    evidence_contributions: Dict[str, float]
    evidence_top_contributions: Dict[str, float]

    # Confidence signals
    margin: float
    peakedness: float


# =============================================================================
# Evidence Graph
# =============================================================================

def compute_evidence_contribution(
    fired_features: FrozenSet[str],
    evidence_graph: Dict[str, Dict[int, float]],
    K: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute evidence vector from fired features.

    Args:
        fired_features: Set of fired feature names
        evidence_graph: feature -> {moment_idx: weight}
        K: Number of moments

    Returns:
        Tuple of (evidence vector, per-feature contributions)
    """
    e = np.zeros(K, dtype=np.float32)
    contributions = {}

    for feature in fired_features:
        if feature not in evidence_graph:
            continue

        weights = evidence_graph[feature]
        feature_contrib = 0.0

        for m, w in weights.items():
            if 0 <= m < K:
                e[m] += w
                feature_contrib += abs(w)

        contributions[feature] = feature_contrib

    return e, contributions


# =============================================================================
# Rule Posterior
# =============================================================================

def compute_rule_posterior(
    p_prior: np.ndarray,
    time_mult: np.ndarray,
    evidence: np.ndarray,
    beta: float = 1.0,
    max_exp_arg: float = 20.0,
) -> np.ndarray:
    """Compute rule-based posterior.

    Formula: p_rule ∝ p_prior · t · exp(β·e)

    Uses numerically stable computation to prevent overflow/underflow.

    Args:
        p_prior: (K,) prior from items
        time_mult: (K,) time multipliers
        evidence: (K,) evidence vector from features
        beta: Evidence scaling factor
        max_exp_arg: Maximum absolute value for exp argument (prevents overflow)

    Returns:
        (K,) normalized rule posterior
    """
    K = len(p_prior)
    uniform = np.ones(K, dtype=np.float32) / K

    # Clamp evidence to prevent overflow: exp(20) ≈ 4.8e8, exp(-20) ≈ 2e-9
    # This is sufficient dynamic range for any practical application
    exp_arg = np.clip(beta * evidence, -max_exp_arg, max_exp_arg)

    # Compute unnormalized: prior * time * exp(clamped_evidence)
    unnorm = p_prior * time_mult * np.exp(exp_arg)

    # Handle edge cases: NaN, Inf, or all-zero
    if not np.all(np.isfinite(unnorm)):
        return uniform

    total = unnorm.sum()
    if total > 0 and np.isfinite(total):
        return (unnorm / total).astype(np.float32)
    else:
        return uniform


# =============================================================================
# Evidence Volume and Agreement
# =============================================================================

def compute_evidence_volume(
    cart_size: int,
    n_fired_features: int,
    max_cart: int = 5,
    max_features: int = 8,
) -> float:
    """Compute evidence volume (0-1).

    ev = sqrt(cart_signal * feature_signal)

    Args:
        cart_size: Number of items in cart
        n_fired_features: Number of non-time features fired
        max_cart: Cart size for full cart_signal
        max_features: Feature count for full feature_signal

    Returns:
        Evidence volume in [0, 1]
    """
    cart_signal = min(cart_size / max_cart, 1.0)
    feature_signal = min(n_fired_features / max_features, 1.0)
    return math.sqrt(cart_signal * feature_signal)


def compute_feature_agreement(
    fired_features: FrozenSet[str],
    evidence_graph: Dict[str, Dict[int, float]],
    top_moment: int,
) -> Tuple[float, bool]:
    """Compute weighted feature agreement.

    agree = sum|w[f, top_moment]| / sum|w[f, all moments]|

    Args:
        fired_features: Set of fired feature names
        evidence_graph: feature -> {moment_idx: weight}
        top_moment: Index of top moment

    Returns:
        Tuple of (agreement score in [0, 1], has_edges)
    """
    total_weight = 0.0
    top_moment_weight = 0.0
    has_edges = False

    for feature in fired_features:
        weights = evidence_graph.get(feature)
        if not weights:
            continue

        has_edges = True
        for m, w in weights.items():
            total_weight += abs(w)
            if m == top_moment:
                top_moment_weight += abs(w)

    if total_weight > 0:
        return top_moment_weight / total_weight, True

    return 0.5, has_edges


def compute_top_moment_contributions(
    fired_features: FrozenSet[str],
    evidence_graph: Dict[str, Dict[int, float]],
    top_moment: int,
) -> Dict[str, float]:
    """Compute per-feature contribution to the top moment."""
    contributions = {}
    for feature in fired_features:
        weights = evidence_graph.get(feature)
        if not weights:
            continue
        if top_moment in weights:
            contributions[feature] = abs(weights[top_moment])
    return contributions


# =============================================================================
# Blending Gate
# =============================================================================

def compute_blending_gate(
    evidence_volume: float,
    agreement: float,
    model_confidence: Optional[float],
    history_strength: float,
    has_model: bool,
    config: ActivationConfig = None,
) -> Tuple[float, str]:
    """Compute interpretable blending gate α.

    α determines blend: p_mix = (1-α) * p_rule + α * p_model

    Args:
        evidence_volume: ev score
        agreement: feature agreement score
        model_confidence: peakedness of p_model (if available)
        history_strength: how much history we have (0-1)
        has_model: whether model posterior is available
        config: Activation config

    Returns:
        Tuple of (alpha, gate_reason)
    """
    config = config or ActivationConfig()

    if not has_model:
        return 0.0, "no_model"

    # Strong evidence + high agreement -> mostly rules
    if evidence_volume >= config.ev_high and agreement >= config.agree_high:
        return config.alpha_strong_rules, "strong_evidence"

    # Weak evidence -> mostly model
    if evidence_volume <= config.ev_low:
        return config.alpha_weak_evidence, "weak_evidence"

    # Model confident + has history -> trust model more
    if (model_confidence is not None and
        model_confidence >= config.model_conf_high and
        history_strength >= config.history_strength_high):
        return config.alpha_strong_model, "model_confident"

    # Default
    return config.alpha_default, "default_blend"


# =============================================================================
# Temperature Softening
# =============================================================================

def compute_temperature(
    evidence_volume: float,
    T_max: float = 2.5,
    gamma: float = 2.0,
) -> float:
    """Compute evidence-controlled temperature.

    T = 1 + (T_max - 1) * (1 - ev)^gamma

    Higher T = flatter distribution (less confident).
    """
    return 1.0 + (T_max - 1.0) * ((1.0 - evidence_volume) ** gamma)


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature softening to distribution.

    p = normalize(p^(1/T))
    """
    if T <= 0:
        T = 1.0

    # Raise to power 1/T
    p_temp = np.power(p + 1e-10, 1.0 / T)

    # Normalize
    total = p_temp.sum()
    if total > 0:
        return p_temp / total
    else:
        return np.ones(len(p), dtype=np.float32) / len(p)


# =============================================================================
# Confidence Tiering
# =============================================================================

def compute_confidence_tier(
    p: np.ndarray,
    evidence_volume: float,
    agreement: Optional[float],
    margin_threshold: float = 0.25,
    peakedness_threshold: float = 0.35,
    ev_threshold: float = 0.4,
    agree_threshold: float = 0.6,
) -> Tuple[str, float, float]:
    """Compute confidence tier from multiple signals.

    Tier logic:
    - high: ≥3 of 4 signals pass
    - medium: 2 signals pass
    - low: else

    Args:
        p: (K,) activation distribution
        evidence_volume: ev score
        agreement: feature agreement score
        *_threshold: thresholds for each signal

    Returns:
        Tuple of (tier, margin, peakedness)
    """
    # Sort to get top-1 and top-2
    sorted_p = np.sort(p)[::-1]
    top1 = sorted_p[0] if len(sorted_p) > 0 else 0.0
    top2 = sorted_p[1] if len(sorted_p) > 1 else 0.0

    # Margin
    margin = top1 - top2

    # Peakedness: 1 - normalized entropy
    K = len(p)
    if K > 1:
        p_safe = np.clip(p, 1e-10, 1.0)
        entropy = -np.sum(p_safe * np.log(p_safe))
        max_entropy = np.log(K)
        peakedness = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        peakedness = 1.0

    signals = [
        margin >= margin_threshold,
        peakedness >= peakedness_threshold,
        evidence_volume >= ev_threshold,
    ]
    if agreement is not None:
        signals.append(agreement >= agree_threshold)

    signals_passed = sum(1 for s in signals if s)
    total_signals = len(signals)

    if total_signals >= 4:
        high_threshold = 3
        medium_threshold = 2
    elif total_signals == 3:
        high_threshold = 3
        medium_threshold = 2
    else:
        high_threshold = 2
        medium_threshold = 1

    if signals_passed >= high_threshold:
        tier = "high"
    elif signals_passed >= medium_threshold:
        tier = "medium"
    else:
        tier = "low"

    return tier, margin, peakedness


# =============================================================================
# Main Activation Function
# =============================================================================

def compute_hybrid_activation(
    cart_items: List[int],
    history_items: List[int],
    fired_features: FrozenSet[str],
    moment2vec: np.ndarray,
    timestamp: Optional[datetime] = None,
    time_mult: Optional[np.ndarray] = None,
    evidence_graph: Optional[Dict[str, Dict[int, float]]] = None,
    population_prior: Optional[np.ndarray] = None,
    p_model: Optional[np.ndarray] = None,
    config: Optional[ActivationConfig] = None,
) -> ActivationResult:
    """Compute full hybrid moment activation.

    Pipeline:
    1. Compute prior from cart/history/population
    2. Apply time multipliers
    3. Compute evidence from fired features
    4. Compute rule posterior
    5. Blend with model posterior (if available)
    6. Apply temperature softening
    7. Compute confidence tier

    Args:
        cart_items: Current cart item IDs
        history_items: Recent history item IDs
        fired_features: Features extracted from cart/context
        moment2vec: (N, K) affinity matrix
        timestamp: Current time (for time_mult lookup)
        time_mult: (K,) time multipliers (or full table to index)
        evidence_graph: feature -> {moment: weight}
        population_prior: (K,) fallback prior
        p_model: (K,) model posterior (if available)
        config: Activation config

    Returns:
        ActivationResult with full diagnostics
    """
    config = config or ActivationConfig()
    K = moment2vec.shape[1]
    N = moment2vec.shape[0]

    # 1. Compute prior from items
    from mosaic.moment_priors import compute_combined_prior
    p_prior, prior_source = compute_combined_prior(
        cart_items=cart_items,
        history_items=history_items,
        moment2vec=moment2vec,
        population_prior=population_prior,
    )

    # 2. Get time multipliers
    if time_mult is None:
        t = np.ones(K, dtype=np.float32)
    elif time_mult.ndim == 1:
        t = time_mult
    else:
        # It's a table - need to index with timestamp
        from mosaic.time_signatures import get_time_multiplier
        if timestamp is not None:
            t = get_time_multiplier(timestamp, time_mult)
        else:
            t = np.ones(K, dtype=np.float32)

    # 3. Compute evidence from features
    if evidence_graph:
        evidence, evidence_contributions = compute_evidence_contribution(
            fired_features, evidence_graph, K
        )
    else:
        evidence = np.zeros(K, dtype=np.float32)
        evidence_contributions = {}

    # 4. Compute rule posterior
    p_rule = compute_rule_posterior(p_prior, t, evidence, config.beta)

    # 5. Compute blending gate
    top_moment = int(np.argmax(p_rule))
    n_fired = len(fired_features)

    evidence_volume = compute_evidence_volume(len(cart_items), n_fired)

    agreement = 0.5
    agreement_applicable = False
    evidence_top_contributions = {}
    if evidence_graph:
        agreement, agreement_applicable = compute_feature_agreement(
            fired_features, evidence_graph, top_moment
        )
        evidence_top_contributions = compute_top_moment_contributions(
            fired_features, evidence_graph, top_moment
        )

    # Model confidence (peakedness)
    model_confidence = None
    if p_model is not None:
        p_safe = np.clip(p_model, 1e-10, 1.0)
        entropy = -np.sum(p_safe * np.log(p_safe))
        max_entropy = np.log(len(p_model)) if len(p_model) > 1 else 1.0
        model_confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

    history_strength = min(len(history_items) / 10, 1.0)

    alpha, gate_reason = compute_blending_gate(
        evidence_volume=evidence_volume,
        agreement=agreement,
        model_confidence=model_confidence,
        history_strength=history_strength,
        has_model=p_model is not None,
        config=config,
    )

    # 6. Blend rule and model posteriors
    if p_model is not None:
        p_mix = (1 - alpha) * p_rule + alpha * p_model
    else:
        p_mix = p_rule.copy()

    # Normalize
    p_mix /= p_mix.sum() + 1e-10

    # 7. Apply temperature softening
    temperature = compute_temperature(evidence_volume, config.T_max, config.gamma)
    p = apply_temperature(p_mix, temperature)

    # 8. Compute confidence tier
    tier, margin, peakedness = compute_confidence_tier(
        p, evidence_volume, agreement if agreement_applicable else None
    )

    return ActivationResult(
        p=p,
        confidence_tier=tier,
        prior_source=prior_source,
        p_prior=p_prior,
        p_rule=p_rule,
        p_model=p_model,
        p_mix=p_mix,
        alpha=alpha,
        gate_reason=gate_reason,
        evidence_volume=evidence_volume,
        agreement=agreement,
        temperature=temperature,
        fired_features=fired_features,
        evidence_contributions=evidence_contributions,
        evidence_top_contributions=evidence_top_contributions,
        margin=margin,
        peakedness=peakedness,
    )


def activation_to_receipt(result: ActivationResult) -> Dict:
    """Convert activation result to receipt-friendly dict."""
    return {
        "p": [round(float(x), 4) for x in result.p],
        "confidence_tier": result.confidence_tier,
        "top_moment": int(np.argmax(result.p)),
        "top_moment_prob": round(float(np.max(result.p)), 4),
        "alpha": round(result.alpha, 4),
        "gate_reason": result.gate_reason,
        "evidence_volume": round(result.evidence_volume, 4),
        "agreement": round(result.agreement, 4),
        "temperature": round(result.temperature, 4),
        "margin": round(result.margin, 4),
        "peakedness": round(result.peakedness, 4),
        "n_fired_features": len(result.fired_features),
        "top_evidence_features": sorted(
            result.evidence_contributions.items(),
            key=lambda x: -x[1]
        )[:5],
        "top_moment_evidence_features": sorted(
            result.evidence_top_contributions.items(),
            key=lambda x: -x[1]
        )[:5],
    }
