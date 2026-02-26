"""Counterfactual logging for offline evaluation and governance.

Logs alternative rankings to answer:
- "Did steering actually change ranking?"
- "How much did policy move items?"
- "Are we protecting accuracy?"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class CounterfactualConfig:
    """Configuration for counterfactual logging."""

    # How many top items to log
    top_n: int = 50

    # Alternative alpha values to test
    alpha_variants: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])

    # Whether to compute each variant
    log_pure_model: bool = True
    log_no_mission: bool = True
    log_no_policy: bool = True
    log_alpha_variants: bool = False


@dataclass
class CounterfactualResult:
    """Result of counterfactual analysis."""

    # Final ranking (what was served)
    final_ranking: List[int]
    final_scores: List[float]

    # Alternative rankings
    pure_model_ranking: Optional[List[int]] = None
    no_mission_ranking: Optional[List[int]] = None
    no_policy_ranking: Optional[List[int]] = None
    alpha_variant_rankings: Optional[Dict[float, List[int]]] = None

    # Metrics
    mission_impact: Optional[Dict] = None
    policy_impact: Optional[Dict] = None
    head_stability: Optional[Dict] = None
    projection_effects: Optional[Dict[str, Dict]] = None
    projection_used: bool = False


ScoreInput = Union[np.ndarray, Dict[int, float], List[float]]
ProjectFn = Callable[[ScoreInput], Union[ScoreInput, Tuple[ScoreInput, Dict]]]


def compute_ranking(scores: ScoreInput, top_n: int = 50) -> Tuple[List[int], List[float]]:
    """Compute top-N ranking from scores.

    Args:
        scores: (N,) scores for all items
        top_n: Number of top items to return

    Returns:
        Tuple of (item_indices, scores) for top-N items
    """
    if isinstance(scores, dict):
        items = sorted(scores.items(), key=lambda x: -x[1])
        n = min(top_n, len(items))
        top_items = items[:n]
        return [item for item, _ in top_items], [float(score) for _, score in top_items]

    scores_arr = np.asarray(scores, dtype=np.float32)
    n = min(top_n, len(scores_arr))
    top_indices = np.argsort(-scores_arr)[:n]
    top_scores = scores_arr[top_indices]
    return top_indices.tolist(), top_scores.astype(float).tolist()


def _combine_scores(base: ScoreInput, *components: ScoreInput) -> ScoreInput:
    if isinstance(base, dict):
        keys = set(base.keys())
        for comp in components:
            if isinstance(comp, dict):
                keys.update(comp.keys())
        combined = {}
        for key in keys:
            total = base.get(key, 0.0)
            for comp in components:
                total += comp.get(key, 0.0)
            combined[key] = total
        return combined

    total = np.asarray(base, dtype=np.float32).copy()
    for comp in components:
        total += np.asarray(comp, dtype=np.float32)
    return total


def _score_stats(scores: ScoreInput) -> Dict[str, float]:
    if isinstance(scores, dict):
        values = np.array(list(scores.values()), dtype=np.float32)
    else:
        values = np.asarray(scores, dtype=np.float32)

    if values.size == 0:
        return {"total_boost": 0.0, "mean_boost": 0.0, "max_boost": 0.0}

    return {
        "total_boost": float(np.sum(np.abs(values))),
        "mean_boost": float(np.mean(values)),
        "max_boost": float(np.max(values)),
    }


def _kendall_tau_union(
    baseline_ranking: List[int],
    comparison_ranking: List[int],
) -> float:
    n = min(len(baseline_ranking), len(comparison_ranking))
    baseline_top = baseline_ranking[:n]
    comparison_top = comparison_ranking[:n]

    items = list(set(baseline_top) | set(comparison_top))
    if len(items) < 2:
        return 1.0

    missing_rank = n  # N+1 in 1-indexed terms
    baseline_pos = {item: idx for idx, item in enumerate(baseline_top)}
    comparison_pos = {item: idx for idx, item in enumerate(comparison_top)}
    for item in items:
        baseline_pos.setdefault(item, missing_rank)
        comparison_pos.setdefault(item, missing_rank)

    concordant = 0
    discordant = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            item_i = items[i]
            item_j = items[j]
            bi = baseline_pos[item_i]
            bj = baseline_pos[item_j]
            ci = comparison_pos[item_i]
            cj = comparison_pos[item_j]
            if bi == bj or ci == cj:
                continue
            if (bi - bj) * (ci - cj) > 0:
                concordant += 1
            else:
                discordant += 1

    total_pairs = concordant + discordant
    return (concordant - discordant) / total_pairs if total_pairs > 0 else 1.0


def _apply_projection(
    scores: ScoreInput,
    project_fn: Optional[ProjectFn],
) -> Tuple[ScoreInput, Optional[Dict]]:
    if project_fn is None:
        return scores, None

    projected = project_fn(scores)
    diag = None
    if isinstance(projected, tuple) and len(projected) == 2:
        projected_scores, diag = projected
    else:
        projected_scores = projected

    effects = {}
    if isinstance(scores, dict):
        changed = 0
        for key, value in projected_scores.items():
            if abs(value - scores.get(key, 0.0)) > 1e-12:
                changed += 1
        effects["n_score_changes"] = changed
        effects["n_items"] = len(projected_scores)
    else:
        before = np.asarray(scores, dtype=np.float32)
        after = np.asarray(projected_scores, dtype=np.float32)
        if len(before) == len(after):
            effects["n_score_changes"] = int(np.sum(np.abs(after - before) > 1e-12))
            effects["n_items"] = int(len(after))

    if diag:
        if "pooled_blocks" in diag:
            effects["n_pooled_blocks"] = len(diag["pooled_blocks"])
        for key in ("n_pre_violations", "n_post_violations", "n_active_constraints"):
            if key in diag:
                effects[key] = int(diag[key])

    return projected_scores, effects or None


def compute_rank_changes(
    baseline_ranking: List[int],
    comparison_ranking: List[int],
) -> Dict:
    """Compute how ranking changed between two rankings.

    Args:
        baseline_ranking: Original ranking
        comparison_ranking: Alternative ranking

    Returns:
        Dict with rank change statistics
    """
    baseline_set = set(baseline_ranking)
    comparison_set = set(comparison_ranking)

    # Items in both
    common = baseline_set & comparison_set

    # Items only in baseline (dropped)
    dropped = baseline_set - comparison_set

    # Items only in comparison (added)
    added = comparison_set - baseline_set

    # Compute rank changes for common items
    baseline_pos = {item: i for i, item in enumerate(baseline_ranking)}
    comparison_pos = {item: i for i, item in enumerate(comparison_ranking)}

    rank_changes = []
    for item in common:
        change = comparison_pos[item] - baseline_pos[item]
        rank_changes.append({
            "item": item,
            "baseline_rank": baseline_pos[item] + 1,
            "comparison_rank": comparison_pos[item] + 1,
            "change": change,
        })

    # Sort by absolute change
    rank_changes.sort(key=lambda x: abs(x["change"]), reverse=True)

    # Compute Kendall tau-like metric on intersection
    concordant = 0
    discordant = 0
    for i, item_i in enumerate(baseline_ranking):
        if item_i not in comparison_set:
            continue
        for j, item_j in enumerate(baseline_ranking[i+1:], start=i+1):
            if item_j not in comparison_set:
                continue
            # Check if relative order is preserved
            ci = comparison_pos[item_i]
            cj = comparison_pos[item_j]
            if ci < cj:
                concordant += 1
            elif ci > cj:
                discordant += 1

    total_pairs = concordant + discordant
    intersection_correlation = (concordant - discordant) / total_pairs if total_pairs > 0 else 1.0

    # Fixed-set correlation on union of top-N with missing = N+1
    union_correlation = _kendall_tau_union(baseline_ranking, comparison_ranking)

    # Overlap of top-N sets (Jaccard)
    top_n = min(len(baseline_ranking), len(comparison_ranking))
    baseline_top = set(baseline_ranking[:top_n])
    comparison_top = set(comparison_ranking[:top_n])
    union = baseline_top | comparison_top
    rank_overlap = len(baseline_top & comparison_top) / len(union) if union else 1.0

    abs_changes = np.array([abs(r["change"]) for r in rank_changes], dtype=np.float32)
    if abs_changes.size:
        p50_abs_change = float(np.percentile(abs_changes, 50))
        p90_abs_change = float(np.percentile(abs_changes, 90))
    else:
        p50_abs_change = 0.0
        p90_abs_change = 0.0

    return {
        "n_common": len(common),
        "n_dropped": len(dropped),
        "n_added": len(added),
        "dropped_items": list(dropped)[:10],
        "added_items": list(added)[:10],
        "top_rank_changes": rank_changes[:10],
        "rank_correlation": round(union_correlation, 4),
        "rank_correlation_intersection": round(intersection_correlation, 4),
        "rank_overlap": round(rank_overlap, 4),
        "mean_abs_change": round(
            np.mean([abs(r["change"]) for r in rank_changes]) if rank_changes else 0, 2
        ),
        "p50_abs_change": round(p50_abs_change, 2),
        "p90_abs_change": round(p90_abs_change, 2),
    }


def compute_counterfactuals(
    base_scores: ScoreInput,
    mission_component: ScoreInput,
    policy_component: ScoreInput,
    served_scores: ScoreInput,
    project_fn: Optional[ProjectFn] = None,
    config: CounterfactualConfig = None,
) -> CounterfactualResult:
    """Compute counterfactual rankings.

    Args:
        base_scores: (N,) base model scores
        mission_component: (N,) mission contributions (e.g., u_mission_perp)
        policy_component: (N,) policy contributions
        served_scores: (N,) final scores actually served (after projection)
        project_fn: Optional projection hook for MOSAIC-faithful counterfactuals
        config: Counterfactual config

    Returns:
        CounterfactualResult with all rankings and metrics
    """
    config = config or CounterfactualConfig()

    # Final scores (what was served)
    final_ranking, final_score_list = compute_ranking(served_scores, config.top_n)

    result = CounterfactualResult(
        final_ranking=final_ranking,
        final_scores=final_score_list,
        projection_used=project_fn is not None,
    )

    # Pure model (no mission, no policy)
    if config.log_pure_model:
        pure_scores, effects = _apply_projection(base_scores, project_fn)
        pure_model_ranking, _ = compute_ranking(pure_scores, config.top_n)
        result.pure_model_ranking = pure_model_ranking
        if effects:
            result.projection_effects = result.projection_effects or {}
            result.projection_effects["pure_model"] = effects

    # No mission (model + policy only)
    if config.log_no_mission:
        no_mission_scores = _combine_scores(base_scores, policy_component)
        no_mission_scores, effects = _apply_projection(no_mission_scores, project_fn)
        no_mission_ranking, _ = compute_ranking(no_mission_scores, config.top_n)
        result.no_mission_ranking = no_mission_ranking
        if effects:
            result.projection_effects = result.projection_effects or {}
            result.projection_effects["no_mission"] = effects

    # No policy (model + mission only)
    if config.log_no_policy:
        no_policy_scores = _combine_scores(base_scores, mission_component)
        no_policy_scores, effects = _apply_projection(no_policy_scores, project_fn)
        no_policy_ranking, _ = compute_ranking(no_policy_scores, config.top_n)
        result.no_policy_ranking = no_policy_ranking
        if effects:
            result.projection_effects = result.projection_effects or {}
            result.projection_effects["no_policy"] = effects

    # Compute impact metrics
    if result.pure_model_ranking:
        result.mission_impact = compute_rank_changes(
            result.pure_model_ranking,
            result.no_policy_ranking or final_ranking,
        )
        result.mission_impact.update(_score_stats(mission_component))

    if result.no_policy_ranking:
        result.policy_impact = compute_rank_changes(
            result.no_policy_ranking,
            final_ranking,
        )
        result.policy_impact.update(_score_stats(policy_component))

    if result.pure_model_ranking:
        pure_top10 = set(result.pure_model_ranking[:10])
        final_top10 = set(final_ranking[:10])
        result.head_stability = {
            "displaced_from_top10": len(pure_top10 - final_top10),
            "top10_overlap": len(pure_top10 & final_top10),
        }

    return result


def counterfactual_to_receipt(result: CounterfactualResult) -> Dict:
    """Convert counterfactual result to receipt-friendly dict."""
    receipt = {
        "final_top10": result.final_ranking[:10],
    }

    if result.pure_model_ranking:
        receipt["pure_model_top10"] = result.pure_model_ranking[:10]

    if result.no_mission_ranking:
        receipt["no_mission_top10"] = result.no_mission_ranking[:10]

    if result.no_policy_ranking:
        receipt["no_policy_top10"] = result.no_policy_ranking[:10]

    if result.mission_impact:
        receipt["mission_impact"] = {
            "rank_correlation": result.mission_impact.get("rank_correlation"),
            "rank_overlap": result.mission_impact.get("rank_overlap"),
            "mean_abs_change": result.mission_impact.get("mean_abs_change"),
            "n_reordered": result.mission_impact.get("n_added", 0) + result.mission_impact.get("n_dropped", 0),
        }

    if result.policy_impact:
        receipt["policy_impact"] = {
            "rank_correlation": result.policy_impact.get("rank_correlation"),
            "rank_overlap": result.policy_impact.get("rank_overlap"),
            "mean_abs_change": result.policy_impact.get("mean_abs_change"),
            "n_reordered": result.policy_impact.get("n_added", 0) + result.policy_impact.get("n_dropped", 0),
        }

    if result.head_stability:
        receipt["head_stability"] = result.head_stability

    if result.projection_effects:
        receipt["projection_effects"] = result.projection_effects

    receipt["projection_mode"] = "projected" if result.projection_used else "pre_projection"

    return receipt


# =============================================================================
# Logging utilities
# =============================================================================

def log_counterfactual(
    result: CounterfactualResult,
    request_id: str,
    logger=None,
) -> None:
    """Log counterfactual result for offline analysis.

    Args:
        result: Counterfactual result
        request_id: Request identifier
        logger: Logger to use (default: print)
    """
    log_fn = logger.info if logger else print

    summary = counterfactual_to_receipt(result)
    summary["request_id"] = request_id

    log_fn(f"[counterfactual] {summary}")


def compute_steering_effectiveness(
    counterfactuals: List[CounterfactualResult],
) -> Dict:
    """Compute aggregate steering effectiveness from multiple counterfactuals.

    Args:
        counterfactuals: List of counterfactual results

    Returns:
        Dict with aggregate metrics
    """
    if not counterfactuals:
        return {}

    mission_correlations = []
    policy_correlations = []
    mission_changes = []
    policy_changes = []
    mission_overlaps = []
    policy_overlaps = []

    for cf in counterfactuals:
        if cf.mission_impact:
            corr = cf.mission_impact.get("rank_correlation")
            if corr is not None:
                mission_correlations.append(corr)
            change = cf.mission_impact.get("mean_abs_change")
            if change is not None:
                mission_changes.append(change)
            overlap = cf.mission_impact.get("rank_overlap")
            if overlap is not None:
                mission_overlaps.append(overlap)

        if cf.policy_impact:
            corr = cf.policy_impact.get("rank_correlation")
            if corr is not None:
                policy_correlations.append(corr)
            change = cf.policy_impact.get("mean_abs_change")
            if change is not None:
                policy_changes.append(change)
            overlap = cf.policy_impact.get("rank_overlap")
            if overlap is not None:
                policy_overlaps.append(overlap)

    return {
        "n_samples": len(counterfactuals),
        "mission": {
            "mean_correlation": round(np.mean(mission_correlations), 4) if mission_correlations else None,
            "mean_rank_change": round(np.mean(mission_changes), 2) if mission_changes else None,
            "mean_overlap": round(np.mean(mission_overlaps), 4) if mission_overlaps else None,
            "pct_with_changes": round(
                100 * sum(1 for c in mission_changes if c > 0) / len(mission_changes), 1
            ) if mission_changes else None,
        },
        "policy": {
            "mean_correlation": round(np.mean(policy_correlations), 4) if policy_correlations else None,
            "mean_rank_change": round(np.mean(policy_changes), 2) if policy_changes else None,
            "mean_overlap": round(np.mean(policy_overlaps), 4) if policy_overlaps else None,
            "pct_with_changes": round(
                100 * sum(1 for c in policy_changes if c > 0) / len(policy_changes), 1
            ) if policy_changes else None,
        },
    }
