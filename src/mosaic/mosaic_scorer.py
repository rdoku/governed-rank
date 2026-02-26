"""
MOSAIC Scorer: Full pipeline orchestration for Margin-Orthogonal Mission Steering.

This is the main entry point for MOSAIC recommendations. It coordinates all stages:
    A) Moment activation (priors + time + evidence → p(m|ctx))
    B) Candidate recall (accuracy pool + moment pool + exploration)
    C) Base scoring (unified model or Item-CF)
    D) Control utility (mission alignment + satiation + policy)
    E) Orthogonalization (interference removal)
    F) Margin-protected constraints (calibrated confidence)
    G) Constrained projection (isotonic on protected runs)

Key invariant: MOSAIC does not allow mission/policy steering to flip pairwise
orderings that the base ranker is "confident" about. Confidence is determined
by a calibrated gap-to-confidence function, not hardcoded top-L cutoffs.

References:
    - MOSAIC_paper.txt
    - mosaic_paper_draft_2.txt (MOSAIC-LENS)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
import json
import numpy as np

from .orthogonalization import orthogonalize_against_base, OrthogonalizationResult
from .gap_calibration import CalibrationResult, get_protected_edges, get_protected_edges_by_budget
from .isotonic_projection import isotonic_project_on_runs, compute_final_ranking
from .satiation import compute_moment_fill, compute_satiation


@dataclass
class MOSAICConfig:
    """Configuration for MOSAIC algorithm."""
    # Activation
    evidence_beta: float = 1.0           # Weight for evidence graph contributions
    temperature_max: float = 2.0         # Max temperature for low-evidence softening

    # Steering
    lambda_m: float = 0.03               # Mission alignment weight
    satiation_floor: float = 0.25        # Minimum satiation multiplier
    satiation_rate: float = 0.6          # Satiation decay rate
    satiation_top_m: int = 2             # Only satiate top-M active moments

    # Protection
    rho: float = 0.90                    # Confidence threshold for protection
    use_calibration: bool = True         # Use learned calibration (vs fixed threshold)
    fallback_gap_threshold: float = 0.05 # Gap threshold if no calibration available
    protection_mode: str = "threshold"   # "threshold" (default) or "budget"
    budget_pct: float = 0.30             # Budget fraction for budget mode
    max_rank: int = 50                   # Head band for protection

    # Target stability (auto-budget)
    target_stability: Optional[float] = None
    target_top_k: int = 10
    auto_budget_steps: Tuple[float, ...] = (0.30, 0.50, 0.70, 1.00)
    auto_budget_max_passes: int = 2
    auto_budget_use_sentinel: bool = False
    sentinel_k: Optional[int] = None

    # Risk guardrails (VaR/CVaR on displacement)
    risk_var_quantile: Optional[float] = None
    risk_max_displaced: Optional[float] = None
    risk_cvar_quantile: Optional[float] = None
    risk_cvar_max: Optional[float] = None
    risk_target_top_k: int = 10
    risk_budget_table: Optional[List[Dict[str, Any]]] = None
    risk_lookup_path: Optional[str] = None
    risk_budget_default: Optional[float] = None

    # Orthogonalization
    rescale_after_ortho: bool = False    # Rescale u_perp to maintain magnitude

    # Diagnostics
    log_receipts: bool = True            # Generate detailed receipts


@dataclass
class MOSAICReceipt:
    """Detailed receipt for a single recommendation, enabling faithful explanations."""
    item_id: int

    # Activation
    activation_p: np.ndarray             # Mission distribution
    activation_confidence: str           # high/medium/low
    fired_features: List[str]            # Evidence features that fired
    feature_contributions: Dict[str, float]  # Feature -> contribution to activation

    # Scoring
    base_score: float                    # s_i
    mission_alignment: float             # A[i] · p
    satiation_factor: float              # sat_i
    policy_boost: float                  # pol_i
    raw_steering_utility: float          # u_i
    mission_utility: float               # mission component before orthogonalization
    policy_utility: float                # policy component (not orthogonalized)

    # Orthogonalization
    orthogonalized_utility: float        # u_perp_i
    target_score: float                  # t_i = s_i + u_perp_i

    # Projection
    final_score: float                   # z_i
    was_constrained: bool                # True if this item was in a pooled block
    base_rank: int                       # Position in base ranking
    final_rank: int                      # Position in final ranking

    # Reason derivation
    primary_reason: str                  # policy/moment/history/popularity/exploration


@dataclass
class MOSAICResult:
    """Result of MOSAIC ranking."""
    ranked_items: List[int]              # Final ranked item IDs
    scores: Dict[int, float]             # Final z scores
    receipts: Dict[int, MOSAICReceipt]   # Per-item receipts (if enabled)

    # Diagnostics
    n_candidates: int
    n_protected_edges: int
    n_active_constraints: int
    projection_coeff: float              # How much steering was aligned with base
    pooled_blocks: List[List[int]]       # Items that were pooled together
    n_missing_base_scores: int
    base_score_coverage: float
    target_stability: Optional[float] = None
    achieved_stability: Optional[float] = None
    budget_used: Optional[float] = None
    auto_budget_passes: int = 1
    risk_budget_used: Optional[float] = None


class MOSAICScorer:
    """
    MOSAIC: Margin-Orthogonal Mission Steering for Accuracy-Preserving, Controllable Recommendations.

    Usage:
        scorer = MOSAICScorer(
            moment_affinities=A,           # item -> moment affinities
            calibration=calibration,       # learned gap->confidence
            config=MOSAICConfig(),
        )

        result = scorer.rank(
            candidates=candidate_items,
            base_scores=base_model_scores,
            activation_p=mission_distribution,
            cart_items=current_cart,
            policies=active_policies,
        )
    """

    def __init__(
        self,
        moment_affinities: np.ndarray,     # Shape: (n_items, K)
        calibration: Optional[CalibrationResult] = None,
        config: MOSAICConfig = None,
    ):
        self.A = moment_affinities
        self.calibration = calibration
        self.config = config or MOSAICConfig()

    def rank(
        self,
        candidates: List[int],
        base_scores: Dict[int, float],
        activation_p: np.ndarray,
        activation_confidence: str = "medium",
        cart_items: List[int] = None,
        policies: List[Any] = None,
        fired_features: List[str] = None,
        feature_contributions: Dict[str, float] = None,
    ) -> MOSAICResult:
        """
        Execute the full MOSAIC ranking pipeline.

        Args:
            candidates: List of candidate item IDs
            base_scores: Dict mapping item_id -> base relevance score
            activation_p: Mission distribution p(m|ctx)
            activation_confidence: Confidence tier (high/medium/low)
            cart_items: Current cart items (for satiation)
            policies: Active steering policies
            fired_features: Evidence features that fired (for receipts)
            feature_contributions: Feature contributions (for receipts)

        Returns:
            MOSAICResult with final ranking and diagnostics
        """
        cart_items = cart_items or []
        policies = policies or []
        fired_features = fired_features or []
        feature_contributions = feature_contributions or {}

        # Stage C: Get base order
        missing_base_scores = [i for i in candidates if i not in base_scores]
        base_score_coverage = 1.0
        if candidates:
            base_score_coverage = 1.0 - (len(missing_base_scores) / len(candidates))

        base_order = sorted(
            candidates,
            key=lambda i: (-base_scores.get(i, float("-inf")), i),
        )
        base_rank = {item: rank + 1 for rank, item in enumerate(base_order)}

        # Stage D: Compute control utility
        mission_utilities = {}
        item_details = {}  # For receipts
        cart_affinities = []
        for cart_item in set(cart_items):
            if 0 <= cart_item < len(self.A):
                cart_affinities.append(self.A[cart_item])

        sat_per_moment = np.ones(len(activation_p), dtype=np.float32)
        if cart_affinities:
            moment_fill = compute_moment_fill(
                cart_affinities,
                K=len(activation_p),
                normalize="mean",
            )
            sat_per_moment = compute_satiation(
                moment_fill,
                activation_p,
                rate=self.config.satiation_rate,
                top_m=self.config.satiation_top_m,
                floor=self.config.satiation_floor,
            )

        for item in candidates:
            # Mission alignment: A[i] · p
            if item < len(self.A):
                alignment = float(np.dot(self.A[item], activation_p))
            else:
                alignment = 0.0

            # Satiation factor from top-M active moments
            if item < len(self.A):
                sat_factor = float(np.dot(self.A[item], sat_per_moment))
            else:
                sat_factor = 1.0

            # Policy boost (simplified - full version uses steering_guardrails.py)
            policy_boost = self._compute_policy_boost(item, policies)

            # Confidence scaling
            conf_scale = {"high": 1.0, "medium": 0.6, "low": 0.0}.get(activation_confidence, 0.5)

            # Raw steering utility
            u_mission = self.config.lambda_m * alignment * sat_factor * conf_scale
            u_policy = policy_boost
            u_i = u_mission + u_policy
            mission_utilities[item] = u_mission

            item_details[item] = {
                "alignment": alignment,
                "sat_factor": sat_factor,
                "policy_boost": policy_boost,
                "raw_utility": u_i,
                "mission_utility": u_mission,
                "policy_utility": u_policy,
            }

        # Stage E: Orthogonalize
        ortho_result = orthogonalize_against_base(
            base_scores=base_scores,
            steering_utilities=mission_utilities,
            rescale=self.config.rescale_after_ortho,
        )

        # Compute target scores
        target_scores = {
            item: base_scores.get(item, 0)
            + ortho_result.u_perp.get(item, 0)
            + item_details[item]["policy_utility"]
            for item in candidates
        }

        def apply_sentinel_cap(scores: Dict[int, float], sentinel_k: Optional[int]) -> Dict[int, float]:
            if sentinel_k is None:
                return scores
            sentinel_idx = sentinel_k - 1
            if sentinel_idx < 0 or sentinel_idx >= len(base_order):
                return scores
            sentinel_item = base_order[sentinel_idx]
            sentinel_score = scores.get(sentinel_item, 0.0)
            eps = 1e-6
            capped = scores.copy()
            for offset, item in enumerate(base_order[sentinel_idx + 1:], start=1):
                cap = sentinel_score - eps * offset
                if capped.get(item, 0.0) > cap:
                    capped[item] = cap
            return capped

        def compute_membership_stability(ranking: List[int], k: int) -> float:
            if k <= 0:
                return 0.0
            base_top = set(base_order[:k])
            final_top = set(ranking[:k])
            return len(base_top & final_top) / k

        def load_risk_table() -> Optional[List[Dict[str, Any]]]:
            if self.config.risk_budget_table:
                return self.config.risk_budget_table
            if self.config.risk_lookup_path:
                try:
                    with open(self.config.risk_lookup_path, "r") as f:
                        data = json.load(f)
                    return data.get("budgets")
                except Exception:
                    return None
            return None

        def select_budget_from_risk_table(table: List[Dict[str, Any]]) -> Optional[float]:
            if not table:
                return None
            var_q = self.config.risk_var_quantile
            max_disp = self.config.risk_max_displaced
            cvar_q = self.config.risk_cvar_quantile
            cvar_max = self.config.risk_cvar_max
            eligible = []
            for row in table:
                budget = row.get("budget_pct")
                if budget is None:
                    continue
                ok = True
                if var_q is not None and max_disp is not None:
                    row_q = row.get("quantile", var_q)
                    if abs(row_q - var_q) > 1e-6:
                        ok = False
                    if row.get("var") is None or row["var"] > max_disp:
                        ok = False
                if cvar_q is not None and cvar_max is not None:
                    row_q = row.get("quantile", cvar_q)
                    if abs(row_q - cvar_q) > 1e-6:
                        ok = False
                    if row.get("cvar") is None or row["cvar"] > cvar_max:
                        ok = False
                if ok:
                    eligible.append(float(budget))
            if not eligible:
                return None
            return min(eligible)

        def run_with_budget(budget_pct: float, sentinel_k: Optional[int]) -> Tuple[Any, List[int], List[int]]:
            if self.calibration and self.config.use_calibration:
                protected = get_protected_edges_by_budget(
                    base_order=base_order,
                    base_scores=base_scores,
                    calibration=self.calibration,
                    budget_pct=budget_pct,
                    max_rank=self.config.max_rank,
                    sentinel_k=sentinel_k,
                )
            else:
                protected = get_protected_edges_by_budget(
                    base_order=base_order,
                    base_scores=base_scores,
                    calibration=None,
                    budget_pct=budget_pct,
                    max_rank=self.config.max_rank,
                    sentinel_k=sentinel_k,
                )
            capped_scores = apply_sentinel_cap(target_scores, sentinel_k)
            proj = isotonic_project_on_runs(
                base_order=base_order,
                target_scores=capped_scores,
                protected_edges=protected,
            )
            ranking = compute_final_ranking(proj.z, base_order)
            return proj, ranking, protected

        target_stability = self.config.target_stability
        achieved_stability = None
        budget_used = None
        auto_passes = 1
        risk_budget_used = None

        risk_table = None
        if self.config.risk_var_quantile is not None or self.config.risk_cvar_quantile is not None:
            risk_table = load_risk_table()
            risk_budget_used = select_budget_from_risk_table(risk_table) if risk_table else None

        if target_stability is not None:
            budget_used = risk_budget_used if risk_budget_used is not None else self.config.budget_pct
            sentinel_k = self.config.sentinel_k if self.config.auto_budget_use_sentinel else None
            proj_result, final_ranking, protected_edges = run_with_budget(budget_used, sentinel_k)
            achieved_stability = compute_membership_stability(final_ranking, self.config.target_top_k)

            if achieved_stability < target_stability and self.config.auto_budget_max_passes > 1:
                steps = sorted(set(self.config.auto_budget_steps))
                next_budget = None
                for step in steps:
                    if step >= budget_used + 1e-9:
                        next_budget = step
                        break
                if next_budget is None:
                    next_budget = budget_used
                budget_used = next_budget
                auto_passes = 2
                proj_result, final_ranking, protected_edges = run_with_budget(budget_used, sentinel_k)
                achieved_stability = compute_membership_stability(final_ranking, self.config.target_top_k)
        else:
            # Stage F: Get protected edges (default behavior)
            if self.config.protection_mode == "budget":
                budget_used = risk_budget_used if risk_budget_used is not None else self.config.budget_pct
                proj_result, final_ranking, protected_edges = run_with_budget(
                    budget_used,
                    self.config.sentinel_k,
                )
            else:
                if self.calibration and self.config.use_calibration:
                    protected_edges = get_protected_edges(
                        base_order=base_order,
                        base_scores=base_scores,
                        calibration=self.calibration,
                        rho=self.config.rho,
                    )
                else:
                    # Fallback: protect edges with gap > threshold
                    protected_edges = []
                    for k in range(len(base_order) - 1):
                        gap = base_scores[base_order[k]] - base_scores[base_order[k + 1]]
                        if gap > self.config.fallback_gap_threshold:
                            protected_edges.append(k)

                # Stage G: Constrained projection
                capped_scores = apply_sentinel_cap(target_scores, self.config.sentinel_k)
                proj_result = isotonic_project_on_runs(
                    base_order=base_order,
                    target_scores=capped_scores,
                    protected_edges=protected_edges,
                )
                final_ranking = compute_final_ranking(proj_result.z, base_order)

        final_rank = {item: rank + 1 for rank, item in enumerate(final_ranking)}

        # Build receipts
        receipts = {}
        if self.config.log_receipts:
            constrained_items = set()
            for block in proj_result.pooled_blocks:
                constrained_items.update(block)

            for item in candidates:
                details = item_details[item]
                receipts[item] = MOSAICReceipt(
                    item_id=item,
                    activation_p=activation_p,
                    activation_confidence=activation_confidence,
                    fired_features=fired_features,
                    feature_contributions=feature_contributions,
                    base_score=base_scores.get(item, 0),
                    mission_alignment=details["alignment"],
                    satiation_factor=details["sat_factor"],
                    policy_boost=details["policy_boost"],
                    raw_steering_utility=details["raw_utility"],
                    mission_utility=details["mission_utility"],
                    policy_utility=details["policy_utility"],
                    orthogonalized_utility=ortho_result.u_perp.get(item, 0),
                    target_score=target_scores.get(item, 0),
                    final_score=proj_result.z.get(item, 0),
                    was_constrained=item in constrained_items,
                    base_rank=base_rank.get(item, -1),
                    final_rank=final_rank.get(item, -1),
                    primary_reason=self._derive_reason(
                        base_score=base_scores.get(item, 0),
                        target_score=target_scores.get(item, 0),
                        final_score=proj_result.z.get(item, 0),
                        details=details,
                    ),
                )

        return MOSAICResult(
            ranked_items=final_ranking,
            scores=proj_result.z,
            receipts=receipts,
            n_candidates=len(candidates),
            n_protected_edges=len(protected_edges),
            n_active_constraints=proj_result.n_active_constraints,
            projection_coeff=ortho_result.projection_coeff,
            pooled_blocks=proj_result.pooled_blocks,
            n_missing_base_scores=len(missing_base_scores),
            base_score_coverage=base_score_coverage,
            target_stability=target_stability,
            achieved_stability=achieved_stability,
            budget_used=budget_used,
            auto_budget_passes=auto_passes,
            risk_budget_used=risk_budget_used,
        )

    def _compute_policy_boost(self, item: int, policies: List[Any]) -> float:
        """Compute total policy boost for an item."""
        # Simplified - full version uses steering_guardrails.py
        total_boost = 0.0
        for policy in policies:
            if hasattr(policy, "get_boost"):
                total_boost += policy.get_boost(item)
        return total_boost

    def _derive_reason(
        self,
        base_score: float,
        target_score: float,
        final_score: float,
        details: Dict,
    ) -> str:
        """Derive primary reason for this recommendation."""
        delta_target = target_score - base_score
        delta_final = final_score - base_score

        if abs(delta_final) < 1e-6:
            return "relevance"

        if abs(details["policy_boost"]) > 0.01 and abs(delta_final) > 1e-6:
            return "policy"

        if abs(details["mission_utility"]) > 0.001 and abs(delta_target) > 1e-6:
            return "moment"

        # Default fallback
        return "relevance"
