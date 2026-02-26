"""Evidence graph builder and management.

Learns sparse feature → moment weights using PMI (Pointwise Mutual Information):
    w[f, m] = log(P(m|f) / P(m))

Positive w boosts the moment, negative w suppresses it.
Store only edges where |w| >= threshold (sparse & interpretable).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


@dataclass
class EvidenceGraphConfig:
    """Configuration for evidence graph building."""

    # Minimum absolute weight to keep (sparsity)
    weight_threshold: float = 0.1
    max_edges_per_feature: int = 3

    # Minimum feature occurrences for reliable estimate
    min_feature_support: int = 50
    support_weight_threshold: float = 0.0

    # Smoothing for probability estimates (Laplace)
    smoothing_alpha: float = 1.0
    min_items_per_order: int = 1

    # Clamp weights to prevent extreme values
    max_weight: float = 3.0
    min_weight: float = -3.0

    # Time feature prefixes (excluded from evidence graph)
    time_feature_prefixes: Tuple[str, ...] = (
        "hour_",
        "dow_",
        "time_",
        "weekend_",
        "weekday_",
        "fri_",
        "sat_",
        "sun_",
        "mon_",
        "tue_",
        "wed_",
        "thu_",
        "month_",
        "season_",
        "holiday_",
    )


@dataclass
class EvidenceGraph:
    """Learned evidence graph."""

    # feature_name -> {moment_idx: weight}
    weights: Dict[str, Dict[int, float]]

    # Metadata
    n_features: int
    n_moments: int
    n_edges: int
    built_at: str
    config: EvidenceGraphConfig

    def get_weight(self, feature: str, moment: int) -> float:
        """Get weight for a feature-moment pair."""
        if feature not in self.weights:
            return 0.0
        return self.weights[feature].get(moment, 0.0)

    def get_feature_weights(self, feature: str) -> Dict[int, float]:
        """Get all moment weights for a feature."""
        return self.weights.get(feature, {})

    def get_moment_features(self, moment: int) -> Dict[str, float]:
        """Get all features that affect a moment."""
        result = {}
        for feature, moments in self.weights.items():
            if moment in moments:
                result[feature] = moments[moment]
        return result

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "weights": self.weights,
            "n_features": self.n_features,
            "n_moments": self.n_moments,
            "n_edges": self.n_edges,
            "built_at": self.built_at,
            "config": {
                "weight_threshold": self.config.weight_threshold,
                "max_edges_per_feature": self.config.max_edges_per_feature,
                "min_feature_support": self.config.min_feature_support,
                "support_weight_threshold": self.config.support_weight_threshold,
                "smoothing_alpha": self.config.smoothing_alpha,
                "min_items_per_order": self.config.min_items_per_order,
                "max_weight": self.config.max_weight,
                "min_weight": self.config.min_weight,
                "time_feature_prefixes": list(self.config.time_feature_prefixes),
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EvidenceGraph":
        """Load from dict."""
        config_data = data.get("config", {})
        config = EvidenceGraphConfig(
            weight_threshold=config_data.get("weight_threshold", 0.1),
            max_edges_per_feature=config_data.get("max_edges_per_feature", 3),
            min_feature_support=config_data.get("min_feature_support", 50),
            support_weight_threshold=config_data.get("support_weight_threshold", 0.0),
            smoothing_alpha=config_data.get("smoothing_alpha", 1.0),
            min_items_per_order=config_data.get("min_items_per_order", 1),
            max_weight=config_data.get("max_weight", 3.0),
            min_weight=config_data.get("min_weight", -3.0),
            time_feature_prefixes=tuple(config_data.get(
                "time_feature_prefixes",
                list(EvidenceGraphConfig().time_feature_prefixes),
            )),
        )
        # Convert string keys back to int for moment indices
        weights = {}
        for feature, moments in data.get("weights", {}).items():
            weights[feature] = {int(m): w for m, w in moments.items()}

        return cls(
            weights=weights,
            n_features=data.get("n_features", len(weights)),
            n_moments=data.get("n_moments", 0),
            n_edges=data.get("n_edges", 0),
            built_at=data.get("built_at", ""),
            config=config,
        )


def build_evidence_graph(
    orders: List[Tuple[List[int], FrozenSet[str]]],
    moment2vec: np.ndarray,
    config: Optional[EvidenceGraphConfig] = None,
) -> EvidenceGraph:
    """Build evidence graph from historical orders.

    For each order:
        1. Extract features (already provided)
        2. Compute order's moment profile from items
        3. Accumulate feature-moment co-occurrences

    Then compute PMI weights:
        w[f, m] = log(P(m|f) / P(m))

    Args:
        orders: List of (item_ids, fired_features) tuples
        moment2vec: (N_items, K) affinity matrix
        config: Evidence graph config

    Returns:
        EvidenceGraph with learned weights
    """
    config = config or EvidenceGraphConfig()
    K = moment2vec.shape[1]
    N = moment2vec.shape[0]

    # Accumulators
    feature_counts: Dict[str, float] = defaultdict(float)  # P(f) numerator
    moment_counts = np.zeros(K, dtype=np.float64)  # P(m) numerator
    feature_moment_counts: Dict[str, np.ndarray] = {}  # P(f, m) numerator
    total_orders = 0

    for item_ids, features in orders:
        # Filter valid items
        valid_items = [i for i in item_ids if 0 <= i < N]
        if len(valid_items) < config.min_items_per_order:
            continue

        # Compute order moment profile
        order_affinities = moment2vec[valid_items]
        order_profile = np.mean(order_affinities, axis=0)
        order_profile /= order_profile.sum() + 1e-10

        total_orders += 1

        # Accumulate moment counts (weighted by profile)
        moment_counts += order_profile

        # Accumulate feature counts and co-occurrences
        for feature in features:
            feature_counts[feature] += 1.0

            if feature not in feature_moment_counts:
                feature_moment_counts[feature] = np.zeros(K, dtype=np.float64)
            feature_moment_counts[feature] += order_profile

    if total_orders == 0:
        return EvidenceGraph(
            weights={},
            n_features=0,
            n_moments=K,
            n_edges=0,
            built_at=datetime.utcnow().isoformat(),
            config=config,
        )

    # Compute probabilities with smoothing
    alpha = config.smoothing_alpha

    # P(m) = (moment_counts + alpha) / (total_orders + K * alpha)
    P_m = (moment_counts + alpha) / (total_orders + K * alpha)

    # Compute PMI weights
    weights: Dict[str, Dict[int, float]] = {}
    n_edges = 0

    for feature, f_count in feature_counts.items():
        # Skip low-support features
        if f_count < config.min_feature_support:
            continue

        # P(m|f) = (feature_moment_counts[f] + alpha) / (f_count + K * alpha)
        fm_counts = feature_moment_counts.get(feature, np.zeros(K))
        P_m_given_f = (fm_counts + alpha) / (f_count + K * alpha)

        # PMI: w[f,m] = log(P(m|f) / P(m))
        ratio = (P_m_given_f + 1e-10) / (P_m + 1e-10)
        pmi = np.log(ratio)

        # Clamp to [min, max]
        pmi = np.clip(pmi, config.min_weight, config.max_weight)

        # Keep only significant edges
        feature_weights = {}
        candidates = []
        for m in range(K):
            if abs(pmi[m]) >= config.weight_threshold:
                if config.support_weight_threshold > 0:
                    support_weight = abs(pmi[m]) * math.sqrt(f_count)
                    if support_weight < config.support_weight_threshold:
                        continue
                candidates.append((m, float(pmi[m])))

        if config.max_edges_per_feature > 0 and len(candidates) > config.max_edges_per_feature:
            candidates.sort(key=lambda x: abs(x[1]), reverse=True)
            candidates = candidates[:config.max_edges_per_feature]

        for m, weight in candidates:
            feature_weights[m] = round(weight, 4)
            n_edges += 1

        if feature_weights:
            weights[feature] = feature_weights

    return EvidenceGraph(
        weights=weights,
        n_features=len(weights),
        n_moments=K,
        n_edges=n_edges,
        built_at=datetime.utcnow().isoformat(),
        config=config,
    )


def build_evidence_graph_from_raw_orders(
    orders: List[Tuple[datetime, List[int]]],
    moment2vec: np.ndarray,
    catalog: Optional[Dict[int, Dict]] = None,
    config: Optional[EvidenceGraphConfig] = None,
) -> EvidenceGraph:
    """Build evidence graph from raw orders (with feature extraction).

    This is a convenience wrapper that handles feature extraction.

    Args:
        orders: List of (timestamp, item_ids) tuples
        moment2vec: (N_items, K) affinity matrix
        catalog: item_id -> {name, aisle, dept}
        config: Evidence graph config

    Returns:
        EvidenceGraph with learned weights
    """
    from mosaic.feature_extractor import extract_features
    config = config or EvidenceGraphConfig()

    def is_time_feature(feature: str) -> bool:
        return any(feature.startswith(prefix) for prefix in config.time_feature_prefixes)

    # Extract features for each order
    orders_with_features = []
    for timestamp, item_ids in orders:
        # Build context from timestamp
        context = {
            "hour": timestamp.hour,
            "dow": timestamp.weekday(),
        }

        # Extract features
        result = extract_features(item_ids, context, catalog)

        # Filter to non-time features (time is handled by time_mult)
        non_time_features = frozenset(
            f for f in result.fired_features
            if not is_time_feature(f)
        )

        orders_with_features.append((item_ids, non_time_features))

    return build_evidence_graph(orders_with_features, moment2vec, config)


def save_evidence_graph(
    graph: EvidenceGraph,
    artifact_dir: str,
    moment_space_id: str,
) -> str:
    """Save evidence graph to disk.

    Args:
        graph: Evidence graph to save
        artifact_dir: Directory to save to
        moment_space_id: Version identifier

    Returns:
        Path to saved file
    """
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, f"evidence_graph_{moment_space_id}.json")

    with open(path, "w") as f:
        json.dump(graph.to_dict(), f, indent=2)

    return path


def load_evidence_graph(
    artifact_dir: str,
    moment_space_id: str,
) -> Optional[EvidenceGraph]:
    """Load evidence graph from disk.

    Args:
        artifact_dir: Directory to load from
        moment_space_id: Version identifier

    Returns:
        EvidenceGraph or None if not found
    """
    path = os.path.join(artifact_dir, f"evidence_graph_{moment_space_id}.json")

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)
        return EvidenceGraph.from_dict(data)
    except Exception:
        return None


# =============================================================================
# Default Evidence Graph (Bootstrap)
# =============================================================================

def create_default_evidence_graph(K: int) -> EvidenceGraph:
    """Create a minimal default evidence graph for bootstrapping.

    Uses hand-coded rules based on common shopping patterns.
    These will be overwritten by learned weights after data collection.

    Args:
        K: Number of moments

    Returns:
        Default EvidenceGraph
    """
    # Common patterns (moment indices are examples - adjust to your taxonomy)
    # Assume: 0=Party, 1=Breakfast, 2=StockUp, 3=Healthy, 4=Comfort, etc.

    default_rules = [
        # Party signals
        ("chips_present", {0: 0.5}),
        ("salsa_present", {0: 0.4}),
        ("chips_and_salsa", {0: 0.8}),
        ("beer_present", {0: 0.6}),
        ("snacks_and_drinks", {0: 0.5}),
        ("party_snacks", {0: 0.9}),

        # Breakfast signals
        ("breakfast_present", {1: 0.7}),
        ("eggs_present", {1: 0.5}),
        ("cereal_present", {1: 0.6}),
        ("coffee_present", {1: 0.4}),
        ("morning_routine", {1: 0.7}),
        ("breakfast_prep", {1: 0.6}),

        # Stock-up signals
        ("large_cart", {2: 0.4}),
        ("very_large_cart", {2: 0.7}),
        ("stock_up_cart", {2: 0.8}),
        ("household_present", {2: 0.5}),
        ("diverse_departments", {2: 0.4}),

        # Healthy signals
        ("produce_present", {3: 0.4}),
        ("yogurt_present", {3: 0.5}),

        # Quick/convenience signals
        ("single_item", {4: 0.3}),
        ("small_cart", {4: 0.2}),
    ]

    # Build weights dict, clamping to valid moment indices
    weights: Dict[str, Dict[int, float]] = {}
    n_edges = 0

    for feature, moment_weights in default_rules:
        valid_weights = {m: w for m, w in moment_weights.items() if m < K}
        if valid_weights:
            weights[feature] = valid_weights
            n_edges += len(valid_weights)

    return EvidenceGraph(
        weights=weights,
        n_features=len(weights),
        n_moments=K,
        n_edges=n_edges,
        built_at=datetime.utcnow().isoformat(),
        config=EvidenceGraphConfig(),
    )


# =============================================================================
# Utilities
# =============================================================================

def get_top_features_for_moment(
    graph: EvidenceGraph,
    moment: int,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """Get top features that boost a moment.

    Args:
        graph: Evidence graph
        moment: Moment index
        top_n: Number of features to return

    Returns:
        List of (feature, weight) tuples sorted by weight descending
    """
    features = graph.get_moment_features(moment)
    sorted_features = sorted(features.items(), key=lambda x: -x[1])
    return sorted_features[:top_n]


def get_feature_moment_summary(
    graph: EvidenceGraph,
    feature: str,
) -> Dict:
    """Get summary of a feature's effect on moments.

    Args:
        graph: Evidence graph
        feature: Feature name

    Returns:
        Dict with boosted/suppressed moments
    """
    weights = graph.get_feature_weights(feature)

    if not weights:
        return {"feature": feature, "found": False}

    boosted = [(m, w) for m, w in weights.items() if w > 0]
    suppressed = [(m, w) for m, w in weights.items() if w < 0]

    boosted.sort(key=lambda x: -x[1])
    suppressed.sort(key=lambda x: x[1])

    return {
        "feature": feature,
        "found": True,
        "boosted_moments": boosted[:5],
        "suppressed_moments": suppressed[:5],
        "total_edges": len(weights),
    }


def evidence_graph_summary(graph: EvidenceGraph) -> Dict:
    """Get summary statistics for an evidence graph."""
    all_weights = []
    for feature_weights in graph.weights.values():
        all_weights.extend(feature_weights.values())

    if not all_weights:
        return {
            "n_features": 0,
            "n_moments": graph.n_moments,
            "n_edges": 0,
        }

    weights_array = np.array(all_weights)

    return {
        "n_features": graph.n_features,
        "n_moments": graph.n_moments,
        "n_edges": graph.n_edges,
        "density": round(graph.n_edges / (graph.n_features * graph.n_moments + 1e-10), 4),
        "weight_stats": {
            "mean": round(float(np.mean(weights_array)), 4),
            "std": round(float(np.std(weights_array)), 4),
            "min": round(float(np.min(weights_array)), 4),
            "max": round(float(np.max(weights_array)), 4),
            "n_positive": int(np.sum(weights_array > 0)),
            "n_negative": int(np.sum(weights_array < 0)),
        },
        "built_at": graph.built_at,
    }
