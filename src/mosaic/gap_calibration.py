"""
MOSAIC Gap Calibration: Learn the mapping from score gaps to correctness probability.

This module implements the offline calibration that powers MOSAIC's confidence-based
protection. Instead of "freeze top-L" heuristics, MOSAIC protects only the base
ranker's confident comparisons using a learned calibration curve.

Key insight: The score gap between adjacent items in the base ranking tells us
how confident the base ranker is about that ordering. We can learn a calibrated
mapping: gap_to_conf(Δ) ≈ P(base ordering is correct | Δ)

IMPORTANT: The calibration quality depends critically on how pairs are sampled.
Use `extract_pairs_pos_neg` (positive vs negative items) for meaningful curves.
The old adjacent-pair method produces flat/uninformative curves.

References:
    - MOSAIC_paper.txt, Section 1.4
    - MOSAIC_paper.txt, Stage F
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from pathlib import Path
import json


@dataclass
class CalibrationConfig:
    """Configuration for gap-to-confidence calibration."""
    n_buckets: int = 30                    # Number of gap buckets
    min_samples_per_bucket: int = 50       # Minimum samples for reliable estimate
    smoothing_pseudocount: float = 1.0     # Laplace smoothing
    use_quantile_buckets: bool = True      # Use quantile edges (recommended)
    gap_range: Tuple[float, float] = (0.0, 1.0)  # Fallback if not using quantiles
    monotonic: bool = True                 # Enforce monotonicity via isotonic regression


@dataclass
class CalibrationResult:
    """Learned calibration mapping."""
    bucket_edges: np.ndarray               # Bucket boundaries
    bucket_confidences: np.ndarray         # P(correct) per bucket
    n_samples: int                         # Total training samples
    samples_per_bucket: np.ndarray         # Samples per bucket (for diagnostics)

    def gap_to_conf(self, gap: float) -> float:
        """Look up confidence for a given gap."""
        # Find bucket index
        bucket_idx = np.searchsorted(self.bucket_edges[1:], gap)
        bucket_idx = min(bucket_idx, len(self.bucket_confidences) - 1)
        return float(self.bucket_confidences[bucket_idx])

    def conf_to_gap(self, conf: float) -> float:
        """Inverse lookup: find gap threshold for a given confidence level."""
        # Find first bucket where confidence >= target
        for i, c in enumerate(self.bucket_confidences):
            if c >= conf:
                return float(self.bucket_edges[i])
        return float(self.bucket_edges[-1])

    def save(self, path: Path):
        """Save calibration to disk."""
        data = {
            "bucket_edges": self.bucket_edges.tolist(),
            "bucket_confidences": self.bucket_confidences.tolist(),
            "n_samples": self.n_samples,
            "samples_per_bucket": self.samples_per_bucket.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "CalibrationResult":
        """Load calibration from disk."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            bucket_edges=np.array(data["bucket_edges"]),
            bucket_confidences=np.array(data["bucket_confidences"]),
            n_samples=data["n_samples"],
            samples_per_bucket=np.array(data["samples_per_bucket"]),
        )

    def print_summary(self):
        """Print calibration curve summary."""
        print(f"Calibration: {self.n_samples} samples, {len(self.bucket_confidences)} buckets")
        print(f"{'Gap Range':<20} {'Confidence':>10} {'Samples':>10}")
        print("-" * 45)
        for i in range(len(self.bucket_confidences)):
            lo = self.bucket_edges[i]
            hi = self.bucket_edges[i + 1] if i + 1 < len(self.bucket_edges) else lo
            conf = self.bucket_confidences[i]
            samples = self.samples_per_bucket[i]
            print(f"{lo:>8.4f} - {hi:<8.4f} {conf:>10.3f} {samples:>10.0f}")


def learn_gap_calibration(
    pairs: List[Tuple[float, bool]],
    config: CalibrationConfig = None,
) -> CalibrationResult:
    """
    Learn the gap-to-confidence mapping from (gap, correct) pairs.

    Args:
        pairs: List of (gap, correct) tuples where:
               - gap = s_positive - s_negative (score difference)
               - correct = True if gap > 0 (positive scored higher)
        config: Calibration configuration

    Returns:
        CalibrationResult with learned mapping
    """
    if config is None:
        config = CalibrationConfig()

    if len(pairs) == 0:
        # Return default (linear) calibration
        edges = np.linspace(config.gap_range[0], config.gap_range[1], config.n_buckets + 1)
        confs = np.linspace(0.5, 1.0, config.n_buckets)
        return CalibrationResult(
            bucket_edges=edges,
            bucket_confidences=confs,
            n_samples=0,
            samples_per_bucket=np.zeros(config.n_buckets),
        )

    # Extract gaps and outcomes
    gaps = np.array([p[0] for p in pairs])
    correct = np.array([p[1] for p in pairs])

    # Create bucket edges - USE QUANTILES for balanced support
    if config.use_quantile_buckets:
        # Quantile edges ensure each bucket has roughly equal samples
        quantiles = np.linspace(0, 1, config.n_buckets + 1)
        edges = np.quantile(gaps, quantiles)
        edges[0] = min(edges[0], 0.0)  # Ensure 0 is included
        edges = np.unique(edges)  # Remove duplicates

        # If we lost buckets due to duplicates, recreate
        if len(edges) < config.n_buckets + 1:
            # Fall back to linear spacing over observed range
            edges = np.linspace(gaps.min(), gaps.max() * 1.01, config.n_buckets + 1)
    else:
        edges = np.linspace(config.gap_range[0], config.gap_range[1], config.n_buckets + 1)

    n_buckets = len(edges) - 1

    # Assign pairs to buckets
    bucket_indices = np.digitize(gaps, edges[1:])
    bucket_indices = np.clip(bucket_indices, 0, n_buckets - 1)

    # Compute empirical P(correct) per bucket
    bucket_correct = np.zeros(n_buckets)
    bucket_total = np.zeros(n_buckets)

    for i in range(len(pairs)):
        bucket_idx = bucket_indices[i]
        bucket_total[bucket_idx] += 1
        if correct[i]:
            bucket_correct[bucket_idx] += 1

    # Laplace smoothing
    alpha = config.smoothing_pseudocount
    bucket_confidences = (bucket_correct + alpha) / (bucket_total + 2 * alpha)

    # Enforce monotonicity via WEIGHTED isotonic regression
    if config.monotonic:
        bucket_confidences = _isotonic_regression_weighted(bucket_confidences, bucket_total)

    return CalibrationResult(
        bucket_edges=edges,
        bucket_confidences=bucket_confidences,
        n_samples=len(pairs),
        samples_per_bucket=bucket_total,
    )


def _isotonic_regression_weighted(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted Pool-Adjacent-Violators algorithm for isotonic regression.

    Unlike unweighted PAV, this gives more influence to buckets with more samples.

    Args:
        y: Array of values to make monotonic (bucket confidences)
        weights: Array of weights (bucket sample counts)

    Returns:
        Monotonically increasing array closest to y in weighted L2 sense
    """
    n = len(y)
    if n == 0:
        return y

    y = y.astype(float).copy()
    w = weights.astype(float).copy()
    w = np.maximum(w, 1e-12)  # Avoid division by zero

    # PAV with weighted means
    # blocks: list of (start_idx, end_idx, sum_weight, sum_weighted_y)
    blocks = [(i, i, w[i], w[i] * y[i]) for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        s1, e1, w1, wy1 = blocks[i]
        s2, e2, w2, wy2 = blocks[i + 1]

        m1 = wy1 / w1
        m2 = wy2 / w2

        if m1 > m2:
            # Violation: merge blocks
            blocks[i] = (s1, e2, w1 + w2, wy1 + wy2)
            blocks.pop(i + 1)
            # Go back to check previous block
            i = max(i - 1, 0)
        else:
            i += 1

    # Extract result
    result = np.zeros(n)
    for s, e, W, WY in blocks:
        block_mean = WY / W
        result[s:e + 1] = block_mean

    return result


def extract_pairs_pos_neg(
    baskets: List[List[int]],
    base_scorer: Callable[[List[int], List[int]], Dict[int, float]],
    n_items: int,
    n_baskets: int = 5000,
    neg_per_pos: int = 10,
    max_rank: int = 50,
    rng_seed: int = 42,
) -> List[Tuple[float, bool]]:
    """
    Extract (gap, correct) pairs using POSITIVE vs NEGATIVE sampling.

    This is the correct way to build calibration data. For each basket:
    - Holdout items are "positives" (items user actually wanted)
    - Non-basket items in top-K are "negatives"
    - For each positive, sample negatives and record gaps

    The label "correct" = (score_positive > score_negative), which is what
    we want to predict from gaps.

    Args:
        baskets: List of baskets (each basket is a list of item IDs)
        base_scorer: Function(cart, candidates) -> Dict[item, score]
        n_items: Total number of items
        n_baskets: Number of baskets to sample from
        neg_per_pos: Number of negatives to sample per positive
        max_rank: Focus on top-K candidates (where ranking matters)
        rng_seed: Random seed

    Returns:
        List of (gap, correct) pairs for calibration
    """
    pairs = []
    rng = np.random.default_rng(rng_seed)

    # Filter eligible baskets
    eligible = [b for b in baskets if len(b) >= 5]
    if len(eligible) > n_baskets:
        indices = rng.choice(len(eligible), n_baskets, replace=False)
        eligible = [eligible[i] for i in indices]

    for basket in eligible:
        basket = list(set(basket))  # Dedupe
        rng.shuffle(basket)

        # Split: holdout (positives) and cart (context)
        n_holdout = max(1, len(basket) // 3)
        positives = set(basket[:n_holdout])
        cart = basket[n_holdout:]

        if len(cart) < 2:
            continue

        # Get candidates (not in cart)
        candidates = [i for i in range(n_items) if i not in cart]

        # Score candidates
        scores = base_scorer(cart, candidates)

        # Get top-K by score
        ranked = sorted(candidates, key=lambda x: -scores.get(x, 0))[:max_rank]

        # Identify positives and negatives in top-K
        pos_in_topk = [i for i in ranked if i in positives]
        neg_in_topk = [i for i in ranked if i not in positives]

        if not pos_in_topk or not neg_in_topk:
            continue

        # For each positive, sample negatives
        for pos_item in pos_in_topk:
            s_pos = scores.get(pos_item, 0)

            n_sample = min(neg_per_pos, len(neg_in_topk))
            sampled_negs = rng.choice(neg_in_topk, size=n_sample, replace=False)

            for neg_item in sampled_negs:
                s_neg = scores.get(neg_item, 0)
                gap = s_pos - s_neg
                correct = gap > 0  # Did base ranker get it right?
                pairs.append((gap, correct))

    return pairs


def extract_pairs_adjacent_boundary(
    baskets: List[List[int]],
    base_scorer: Callable[[List[int], List[int]], Dict[int, float]],
    n_items: int,
    n_baskets: int = 5000,
    max_rank: int = 50,
    rng_seed: int = 42,
) -> List[Tuple[float, bool]]:
    """
    Extract (gap, correct) pairs from ADJACENT BOUNDARY comparisons.

    This is the recommended calibration method for MOSAIC. For each basket:
    - Create a cart context and holdout set (ground truth engagement)
    - Score candidates and create base ranking
    - For each adjacent pair (k, k+1) in the ranking:
      - Skip if both are in holdout (tie: both "engaged")
      - Skip if neither is in holdout (tie: both "not engaged")
      - correct = True if the engaged item is ranked higher

    This produces calibration curves that rise smoothly from ~0.5 to ~0.9+
    instead of jumping to 0.999 at tiny gaps.

    Args:
        baskets: List of baskets (each basket is a list of item IDs)
        base_scorer: Function(cart, candidates) -> Dict[item, score]
        n_items: Total number of items
        n_baskets: Number of baskets to sample from
        max_rank: Focus on top-K candidates (where ranking matters)
        rng_seed: Random seed

    Returns:
        List of (gap, correct) pairs for calibration
    """
    pairs = []
    rng = np.random.default_rng(rng_seed)

    # Filter eligible baskets
    eligible = [b for b in baskets if len(b) >= 5]
    if len(eligible) > n_baskets:
        indices = rng.choice(len(eligible), n_baskets, replace=False)
        eligible = [eligible[i] for i in indices]

    for basket in eligible:
        basket = list(set(basket))  # Dedupe
        rng.shuffle(basket)

        # Split: holdout (engagement labels) and cart (context)
        n_holdout = max(1, len(basket) // 3)
        holdout = set(basket[:n_holdout])
        cart = basket[n_holdout:]

        if len(cart) < 2:
            continue

        # Get candidates (not in cart)
        candidates = [i for i in range(n_items) if i not in cart]

        # Score candidates
        scores = base_scorer(cart, candidates)

        # Get top-K by score (this is our base ranking)
        ranked = sorted(candidates, key=lambda x: -scores.get(x, 0))[:max_rank]

        # For each adjacent pair, check if one is engaged and one is not
        for k in range(len(ranked) - 1):
            item_higher = ranked[k]      # Ranked higher by base
            item_lower = ranked[k + 1]   # Ranked lower by base

            higher_engaged = item_higher in holdout
            lower_engaged = item_lower in holdout

            # Skip ties (both engaged or both not engaged)
            if higher_engaged == lower_engaged:
                continue

            # Gap = score difference
            gap = scores.get(item_higher, 0) - scores.get(item_lower, 0)

            # Correct if engaged item is ranked higher
            correct = higher_engaged

            pairs.append((gap, correct))

    return pairs


def extract_pairs_from_logs(
    rankings: List[List[int]],
    base_scores: List[Dict[int, float]],
    outcomes: List[Dict[int, float]],
    sample_rate: float = 0.3,
) -> List[Tuple[float, bool]]:
    """
    Extract (gap, correct) pairs from historical ranking logs.

    DEPRECATED: Use extract_pairs_pos_neg instead. This adjacent-pair method
    produces flat calibration curves because most adjacent pairs have no
    engagement signal (both 0).

    Args:
        rankings: List of ranked item lists (in base order)
        base_scores: List of dicts mapping item_id -> base score
        outcomes: List of dicts mapping item_id -> engagement signal
        sample_rate: Fraction of adjacent pairs to sample

    Returns:
        List of (gap, correct) pairs for calibration
    """
    pairs = []
    rng = np.random.default_rng(42)

    for ranking, scores, outcome in zip(rankings, base_scores, outcomes):
        for i in range(len(ranking) - 1):
            if rng.random() > sample_rate:
                continue

            item_higher = ranking[i]
            item_lower = ranking[i + 1]

            gap = scores.get(item_higher, 0) - scores.get(item_lower, 0)

            eng_higher = outcome.get(item_higher, 0)
            eng_lower = outcome.get(item_lower, 0)

            # SKIP TIES - they add noise
            if eng_higher == eng_lower:
                continue

            correct = eng_higher > eng_lower
            pairs.append((gap, correct))

    return pairs


def get_protected_edges(
    base_order: List[int],
    base_scores: Dict[int, float],
    calibration: CalibrationResult,
    rho: float = 0.90,
) -> List[int]:
    """
    Identify which adjacent edges in the base ranking should be protected.

    An edge (k, k+1) is protected if gap_to_conf(gap_k) >= rho, meaning
    the base ranker is confident about this ordering.

    Args:
        base_order: Items sorted by base score (descending)
        base_scores: Dict mapping item_id -> base score
        calibration: Learned gap-to-confidence mapping
        rho: Confidence threshold (e.g., 0.90 = "protect if 90% likely correct")

    Returns:
        List of edge indices k where edge (k, k+1) is protected
    """
    protected = []

    for k in range(len(base_order) - 1):
        item_k = base_order[k]
        item_k1 = base_order[k + 1]

        gap = base_scores[item_k] - base_scores[item_k1]
        conf = calibration.gap_to_conf(gap)

        if conf >= rho:
            protected.append(k)

    return protected


def get_protected_edges_by_percentile(
    base_order: List[int],
    base_scores: Dict[int, float],
    protect_top_pct: float = 0.20,
    max_rank: int = 20,
) -> List[int]:
    """
    Fallback: protect edges by gap percentile when calibration isn't available.

    Protects the top X% of gaps in the head of the ranking.

    Args:
        base_order: Items sorted by base score (descending)
        base_scores: Dict mapping item_id -> base score
        protect_top_pct: Protect edges with gaps in top X percentile
        max_rank: Only consider edges up to this rank

    Returns:
        List of edge indices k where edge (k, k+1) is protected
    """
    # Compute gaps for head positions
    gaps = []
    for k in range(min(len(base_order) - 1, max_rank)):
        gap = base_scores[base_order[k]] - base_scores[base_order[k + 1]]
        gaps.append((k, gap))

    if not gaps:
        return []

    # Find threshold
    gap_values = [g[1] for g in gaps]
    threshold = np.percentile(gap_values, 100 * (1 - protect_top_pct))

    # Protect edges above threshold
    protected = [k for k, gap in gaps if gap >= threshold]

    return protected


def get_protected_edges_by_budget(
    base_order: List[int],
    base_scores: Dict[int, float],
    calibration: Optional[CalibrationResult] = None,
    budget_pct: float = 0.30,
    max_rank: int = 50,
    rank_bands: Optional[List[Tuple[int, int, float]]] = None,
    normalize_gaps: bool = False,
    sentinel_k: Optional[int] = None,
) -> List[int]:
    """
    PRIMARY MODE: Protect top B% most confident edges by budget.

    This avoids the cliff problem when calibration produces clustered confidences.
    Instead of "protect if conf >= ρ", we "protect top B% of edges by confidence".

    Args:
        base_order: Items sorted by base score (descending)
        base_scores: Dict mapping item_id -> base score
        calibration: Optional learned gap-to-confidence mapping
        budget_pct: Fraction of edges to protect (0.30 = top 30%)
        max_rank: Only consider edges up to this rank
        rank_bands: Optional list of (start, end, band_budget_pct) for rank-band budgets
                    e.g., [(0, 10, 0.5), (10, 30, 0.3), (30, 50, 0.2)]
                    If provided, budget_pct is ignored.
        normalize_gaps: If True, normalize gaps by local score dispersion

    Returns:
        List of edge indices k where edge (k, k+1) is protected
    """
    n_edges = min(len(base_order) - 1, max_rank)
    if n_edges == 0:
        return []

    # Compute gaps and optionally normalize
    gaps = []
    for k in range(n_edges):
        gap = base_scores[base_order[k]] - base_scores[base_order[k + 1]]
        gaps.append(gap)

    if normalize_gaps and len(gaps) > 1:
        # Normalize by local standard deviation (window of 5 around each edge)
        gaps_array = np.array(gaps)
        normalized_gaps = []
        for k in range(len(gaps)):
            window_start = max(0, k - 2)
            window_end = min(len(gaps), k + 3)
            local_std = np.std(gaps_array[window_start:window_end]) + 1e-8
            normalized_gaps.append(gaps[k] / local_std)
        gaps = normalized_gaps

    # Compute confidence scores for each edge
    if calibration is not None:
        confidences = [(k, calibration.gap_to_conf(gaps[k])) for k in range(n_edges)]
    else:
        # Without calibration, use gap magnitude as proxy for confidence
        confidences = [(k, gaps[k]) for k in range(n_edges)]

    # Apply rank-band budgets if specified
    if rank_bands is not None:
        protected = []
        for band_start, band_end, band_budget in rank_bands:
            band_edges = [(k, conf) for k, conf in confidences
                         if band_start <= k < band_end]
            if not band_edges:
                continue
            # Sort by confidence (descending) and take top band_budget%
            band_edges_sorted = sorted(band_edges, key=lambda x: -x[1])
            n_protect = max(1, int(len(band_edges) * band_budget))
            protected.extend([k for k, _ in band_edges_sorted[:n_protect]])
        protected = sorted(set(protected))
        if sentinel_k is not None:
            sentinel_idx = sentinel_k - 1
            if 0 <= sentinel_idx < n_edges:
                protected = sorted(set(protected + [sentinel_idx]))
        return protected

    # Default: protect top budget_pct% of all edges
    confidences_sorted = sorted(confidences, key=lambda x: -x[1])
    n_protect = max(1, int(n_edges * budget_pct))
    protected = [k for k, _ in confidences_sorted[:n_protect]]
    protected = sorted(protected)

    if sentinel_k is not None:
        sentinel_idx = sentinel_k - 1
        if 0 <= sentinel_idx < n_edges:
            protected = sorted(set(protected + [sentinel_idx]))

    return protected


def get_protected_edges_adaptive(
    base_order: List[int],
    base_scores: Dict[int, float],
    calibration: Optional[CalibrationResult] = None,
    rho: Optional[float] = None,
    budget_pct: float = 0.30,
    max_rank: int = 50,
    mode: str = "budget",
    sentinel_k: Optional[int] = None,
) -> Tuple[List[int], str]:
    """
    Adaptive edge protection that chooses the best strategy.

    Modes:
        - "budget": Protect top B% edges (recommended for Item-CF)
        - "threshold": Protect edges with conf >= rho (for well-calibrated rankers)
        - "auto": Use threshold if calibration has good spread, else budget

    Args:
        base_order: Items sorted by base score (descending)
        base_scores: Dict mapping item_id -> base score
        calibration: Optional learned gap-to-confidence mapping
        rho: Confidence threshold (for threshold mode)
        budget_pct: Fraction of edges to protect (for budget mode)
        max_rank: Only consider edges up to this rank
        mode: "budget", "threshold", or "auto"

    Returns:
        Tuple of (protected edge indices, mode used)
    """
    if mode == "threshold" and calibration is not None and rho is not None:
        protected = get_protected_edges(base_order, base_scores, calibration, rho)
        if sentinel_k is not None:
            sentinel_idx = sentinel_k - 1
            if 0 <= sentinel_idx < min(len(base_order) - 1, max_rank):
                protected = sorted(set(protected + [sentinel_idx]))
        return protected, "threshold"

    if mode == "auto" and calibration is not None:
        # Check if calibration has good spread (not a cliff)
        conf_range = calibration.bucket_confidences.max() - calibration.bucket_confidences.min()
        if conf_range > 0.3 and rho is not None:
            # Good spread, use threshold mode
            protected = get_protected_edges(base_order, base_scores, calibration, rho)
            return protected, "threshold"
        # Fall through to budget mode

    # Budget mode (default)
    protected = get_protected_edges_by_budget(
        base_order,
        base_scores,
        calibration,
        budget_pct,
        max_rank,
        sentinel_k=sentinel_k,
    )
    return protected, "budget"
