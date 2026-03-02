"""
govern() — the simplest entry point for governed reranking.

No moments, no calibration, no config objects. Just:

    from mosaic import govern

    result = govern(
        base_scores={"doc1": 0.9, "doc2": 0.8, "doc3": 0.7},
        steering_scores={"doc1": -0.5, "doc2": 0.3, "doc3": 0.8},
        budget=0.3,
    )
    print(result.ranked_items)  # reranked order
    print(result.receipts)      # per-item audit trail
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Hashable, Union, Sequence

import numpy as np

from .orthogonalization import orthogonalize_against_base, compute_target_scores
from .gap_calibration import get_protected_edges_by_budget
from .isotonic_projection import isotonic_project_on_runs, compute_final_ranking


@dataclass
class GovernReceipt:
    """Audit trail for a single item."""
    item: Any
    base_score: float
    steering_score: float
    orthogonalized_steering: float
    final_score: float
    base_rank: int
    final_rank: int


@dataclass
class GovernResult:
    """Result of govern() reranking."""
    ranked_items: List[Any]
    scores: Dict[Any, float]
    receipts: List[GovernReceipt]
    n_protected_edges: int
    n_active_constraints: int
    projection_coeff: float


def govern(
    base_scores: Union[Dict[Hashable, float], np.ndarray, Sequence[float]],
    steering_scores: Union[Dict[Hashable, float], np.ndarray, Sequence[float]],
    budget: float = 0.30,
) -> GovernResult:
    """Rerank a scored list toward a policy objective without breaking accuracy.

    Three steps, fully automatic:
      1. Orthogonalize steering against base (remove interference)
      2. Protect the most confident base edges (budget controls how many)
      3. Isotonic projection (enforce constraints, preserve steering elsewhere)

    Args:
        base_scores: Item → relevance/quality score from your base ranker.
                     Accepts a dict (key → score), numpy array, or list.
        steering_scores: Item → policy signal (e.g. toxicity, fairness, margin).
                         Accepts a dict (key → score), numpy array, or list.
        budget: Fraction of edges to protect (0 = full reorder, 1 = no change).
                Default 0.30 means the top 30% most confident ordering decisions
                are locked.

    Returns:
        GovernResult with ranked_items, scores, and per-item receipts.
    """
    # Normalize array/list inputs to dicts with integer keys
    if not isinstance(base_scores, dict):
        base_scores = {i: float(v) for i, v in enumerate(base_scores)}
    if not isinstance(steering_scores, dict):
        steering_scores = {i: float(v) for i, v in enumerate(steering_scores)}

    items = sorted(base_scores.keys(), key=lambda k: base_scores[k], reverse=True)

    # Map arbitrary keys → int indices for the core pipeline
    key_to_idx = {k: i for i, k in enumerate(items)}
    idx_to_key = {i: k for k, i in key_to_idx.items()}

    base_int = {key_to_idx[k]: v for k, v in base_scores.items()}
    steer_int = {key_to_idx[k]: steering_scores.get(k, 0.0) for k in items}

    # Step 1: Orthogonalize
    ortho = orthogonalize_against_base(base_int, steer_int)

    # Step 2: Target scores + protected edges
    target = compute_target_scores(base_int, ortho.u_perp)
    base_order = list(range(len(items)))  # already sorted descending
    protected = get_protected_edges_by_budget(
        base_order, base_int, budget_pct=budget,
    )

    # Step 3: Constrained projection
    proj = isotonic_project_on_runs(base_order, target, protected)
    final_order = compute_final_ranking(proj.z, base_order)

    # Map back to original keys
    ranked_items = [idx_to_key[i] for i in final_order]
    scores = {idx_to_key[i]: v for i, v in proj.z.items()}

    receipts = []
    for rank, idx in enumerate(final_order):
        key = idx_to_key[idx]
        receipts.append(GovernReceipt(
            item=key,
            base_score=base_scores[key],
            steering_score=steering_scores.get(key, 0.0),
            orthogonalized_steering=ortho.u_perp[idx],
            final_score=proj.z[idx],
            base_rank=items.index(key),
            final_rank=rank,
        ))

    return GovernResult(
        ranked_items=ranked_items,
        scores=scores,
        receipts=receipts,
        n_protected_edges=proj.n_constraints,
        n_active_constraints=proj.n_active_constraints,
        projection_coeff=ortho.projection_coeff,
    )
