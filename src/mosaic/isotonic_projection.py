"""
MOSAIC Isotonic Projection: Fast constrained projection on protected runs.

This module implements Stage G of the MOSAIC algorithm - computing final scores
that are as close as possible to target scores while respecting protected
ordering constraints.

Key insight: Because constraints are only on adjacent pairs in base order,
the problem decomposes into "runs" of consecutive protected edges. Each run
can be solved independently with the Pool-Adjacent-Violators (PAV) algorithm.

Optimization problem:
    z = argmin_z  sum_i w_i (z_i - t_i)^2
    s.t.  z_{pi_k} >= z_{pi_{k+1}}  for all k in E_protected

Complexity:
    - Sorting candidates: O(N log N)
    - Projection: O(N)  [linear scan with PAV]

References:
    - MOSAIC_paper.txt, Stage G
    - Best, M. J., & Chakravarti, N. (1990). Active set algorithms for isotonic regression.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class ProjectionResult:
    """Result of constrained isotonic projection."""
    z: Dict[int, float]           # Final scores per item
    n_constraints: int            # Number of protected edges
    n_active_constraints: int     # Number of constraints that were binding
    pooled_blocks: List[List[int]]  # Blocks that were pooled (for receipts)
    pooled_block_positions: List[Tuple[int, int]]  # Base-order index ranges
    n_pre_violations: int         # Protected edges violated pre-projection


def isotonic_project_on_runs(
    base_order: List[int],
    target_scores: Dict[int, float],
    protected_edges: List[int],
    weights: Optional[Dict[int, float]] = None,
) -> ProjectionResult:
    """
    Compute final scores via constrained projection.

    Solves the weighted least-squares problem subject to monotonicity
    constraints only on protected edges.

    Args:
        base_order: Items sorted by base score (descending), i.e., pi
        target_scores: Dict mapping item_id -> target score (t_i = s_i + u_perp_i)
        protected_edges: List of edge indices k where (k, k+1) must satisfy z_k >= z_{k+1}
        weights: Optional dict mapping item_id -> weight (default: uniform)

    Returns:
        ProjectionResult with final scores and diagnostic info
    """
    n = len(base_order)

    if n == 0:
        return ProjectionResult(
            z={},
            n_constraints=0,
            n_active_constraints=0,
            pooled_blocks=[],
            pooled_block_positions=[],
            n_pre_violations=0,
        )

    # Convert to arrays in base order
    t = np.array([target_scores.get(base_order[i], 0.0) for i in range(n)])

    if weights is not None:
        w = np.array([weights.get(base_order[i], 1.0) for i in range(n)])
    else:
        w = np.ones(n)
    w = np.maximum(w, 1e-8)

    if not protected_edges:
        z_dict = {base_order[i]: float(t[i]) for i in range(n)}
        return ProjectionResult(
            z=z_dict,
            n_constraints=0,
            n_active_constraints=0,
            pooled_blocks=[],
            pooled_block_positions=[],
            n_pre_violations=0,
        )

    # Identify runs of consecutive protected edges
    protected_set = set(protected_edges)
    runs = _identify_runs(n, protected_set)

    # Process each run with weighted PAV
    z = t.copy()
    pooled_blocks = []
    pooled_block_positions = []
    n_active = 0
    n_pre_violations = 0

    for k in protected_edges:
        if 0 <= k < n - 1 and t[k] < t[k + 1]:
            n_pre_violations += 1

    for run_start, run_end in runs:
        # Extract run
        run_t = t[run_start:run_end + 1]
        run_w = w[run_start:run_end + 1]

        # Apply weighted PAV to this run
        run_z, run_blocks, run_active = _weighted_pav(run_t, run_w)

        # Store results
        z[run_start:run_end + 1] = run_z
        n_active += run_active

        # Translate block indices to global and item IDs
        for block in run_blocks:
            if len(block) > 1:
                global_block = [base_order[run_start + i] for i in block]
                pooled_blocks.append(global_block)
                pooled_block_positions.append(
                    (run_start + min(block), run_start + max(block))
                )

    # Convert back to dict
    z_dict = {base_order[i]: float(z[i]) for i in range(n)}

    return ProjectionResult(
        z=z_dict,
        n_constraints=len(protected_edges),
        n_active_constraints=n_active,
        pooled_blocks=pooled_blocks,
        pooled_block_positions=pooled_block_positions,
        n_pre_violations=n_pre_violations,
    )


def _identify_runs(n: int, protected_edges: set) -> List[Tuple[int, int]]:
    """
    Identify runs of consecutive protected edges.

    A "run" is a maximal sequence of items where all intermediate edges
    are protected. Unprotected edges break runs.

    Args:
        n: Total number of items
        protected_edges: Set of protected edge indices

    Returns:
        List of (start_idx, end_idx) tuples for each run
    """
    if n <= 1:
        return []

    runs = []
    run_start = None

    for k in range(n - 1):
        if k in protected_edges:
            if run_start is None:
                run_start = k
        else:
            if run_start is not None:
                # End of run (includes item at k)
                runs.append((run_start, k))
                run_start = None

    # Handle run that extends to the end
    if run_start is not None:
        runs.append((run_start, n - 1))

    return runs


def _weighted_pav(t: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, List[List[int]], int]:
    """
    Weighted Pool-Adjacent-Violators algorithm for monotonically decreasing regression.

    We want z_0 >= z_1 >= ... >= z_{n-1} (descending, like base order).

    Args:
        t: Target values
        w: Weights for each position

    Returns:
        Tuple of (z, blocks, n_active) where:
        - z: Projected values satisfying monotonicity
        - blocks: List of index lists that were pooled together
        - n_active: Number of constraints that were active (blocks of size > 1)
    """
    n = len(t)
    if n == 0:
        return np.array([]), [], 0

    # Initialize each element as its own block
    # Each block stores: (weighted_sum, total_weight, indices)
    blocks = [(t[i] * w[i], w[i], [i]) for i in range(n)]

    # Process left to right, merging violating adjacent blocks
    i = 0
    while i < len(blocks) - 1:
        # Compute block means
        mean_i = blocks[i][0] / blocks[i][1] if blocks[i][1] > 0 else 0
        mean_next = blocks[i + 1][0] / blocks[i + 1][1] if blocks[i + 1][1] > 0 else 0

        if mean_i < mean_next:
            # Violation: block i should be >= block i+1, but it's smaller
            # Merge blocks
            merged_sum = blocks[i][0] + blocks[i + 1][0]
            merged_weight = blocks[i][1] + blocks[i + 1][1]
            merged_indices = blocks[i][2] + blocks[i + 1][2]

            blocks[i] = (merged_sum, merged_weight, merged_indices)
            blocks.pop(i + 1)

            # Go back to check previous block
            if i > 0:
                i -= 1
        else:
            i += 1

    # Extract final values
    z = np.zeros(n)
    final_blocks = []
    n_active = 0

    for weighted_sum, total_weight, indices in blocks:
        block_mean = weighted_sum / total_weight if total_weight > 0 else 0
        for idx in indices:
            z[idx] = block_mean
        if len(indices) > 1:
            final_blocks.append(indices)
            n_active += len(indices) - 1  # Number of constraints that became active

    return z, final_blocks, n_active


def compute_final_ranking(
    z: Dict[int, float],
    base_order: List[int],
) -> List[int]:
    """
    Compute final ranking from projected scores with stable tie-breaking.

    Items are sorted by z (descending), with ties broken by base order position.

    Args:
        z: Final projected scores
        base_order: Original base ranking (for tie-breaking)

    Returns:
        Final ranked list of item IDs
    """
    base_pos = {item: i for i, item in enumerate(base_order)}

    # Sort by (-z, base_position) for descending z with stable tie-break
    return sorted(
        z.keys(),
        key=lambda item: (-z[item], base_pos.get(item, len(base_order)))
    )
