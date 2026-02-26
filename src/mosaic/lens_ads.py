"""
LENS: Load- and Experience-Neutral Sponsored Serving.

LENS inserts sponsored content without breaking organic experience by using
MOSAIC's calibrated confidence boundaries to choose experience-safe slots.

Key insight: Don't place ads above highly confident organic boundaries.
LENS uses the same gap→confidence calibration as MOSAIC to identify "safe"
insertion points where the organic ranking is uncertain.

Algorithm:
    1. Score ads: EV + mission alignment
    2. Identify safe slots using MOSAIC confidence boundaries
    3. Assign ads to slots maximizing gain subject to constraints
    4. Merge into final feed preserving organic order

References:
    - mosaic_paper_draft_2.txt, Section 5
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .gap_calibration import CalibrationResult


@dataclass
class AdCandidate:
    """A candidate ad for insertion."""
    ad_id: str
    campaign_id: str
    bid: float                          # Cost per click/impression
    p_click: float                      # Predicted click probability
    quality_score: float                # User experience quality (0-1)
    throttle: float = 1.0               # Pacing throttle (0-1)
    category: Optional[str] = None      # For mission alignment
    moment_affinities: Optional[np.ndarray] = None  # Ad's affinity to moments


@dataclass
class AdPlacement:
    """Result of placing an ad."""
    ad_id: str
    slot_position: int                  # Position in final feed
    organic_boundary: int               # Which organic boundary this is after
    boundary_confidence: float          # MOSAIC confidence at this boundary
    expected_value: float               # EV of this placement
    mission_alignment: float            # Alignment with active mission
    utility: float                      # Total ad utility score


@dataclass
class LENSConfig:
    """Configuration for LENS ad insertion."""
    # Slot constraints
    max_ad_load: float = 0.10           # Max fraction of feed that can be ads
    min_spacing: int = 5                # Minimum items between ads
    allowed_slots: List[int] = None     # Allowed feed positions (1-indexed, e.g., [3, 8, 15])
    slot_indexing: str = "one"          # "one" for 1-indexed feed positions, "zero" for 0-indexed

    # Confidence threshold
    rho_insert: float = 0.80            # Only insert where confidence < this

    # Scoring weights
    lambda_ad: float = 0.1              # Weight for mission alignment in ad scoring
    boundary_cost_weight: float = 0.2   # Penalty weight for higher-confidence boundaries

    # Position effects
    position_decay: float = 0.95        # CTR decay per position


@dataclass
class LENSResult:
    """Result of LENS ad insertion."""
    final_feed: List[Tuple[str, Any]]   # List of (type, item) where type is "organic" or "sponsored"
    placements: List[AdPlacement]       # Details of each ad placement
    blocked_slots: List[int]            # Blocked slot positions (same indexing as config)
    total_ad_value: float               # Sum of ad utilities


class LENSInserter:
    """
    LENS: Load- and Experience-Neutral Sponsored Serving.

    Uses MOSAIC's confidence calibration to choose safe ad insertion slots.

    Usage:
        inserter = LENSInserter(calibration, config)

        result = inserter.insert_ads(
            organic_ranking=organic_items,
            organic_scores=item_scores,
            ad_candidates=ads,
            activation_p=mission_distribution,
            base_scores=base_model_scores,
        )
    """

    def __init__(
        self,
        calibration: Optional[CalibrationResult] = None,
        config: LENSConfig = None,
    ):
        self.calibration = calibration
        self.config = config or LENSConfig()

    def insert_ads(
        self,
        organic_ranking: List[int],
        organic_scores: Dict[int, float],
        ad_candidates: List[AdCandidate],
        activation_p: Optional[np.ndarray] = None,
        max_ads: int = 3,
        base_scores: Optional[Dict[int, float]] = None,
        boundary_confidences: Optional[Dict[int, float]] = None,
    ) -> LENSResult:
        """
        Insert sponsored content into organic ranking.

        Args:
            organic_ranking: Ranked list of organic item IDs (from MOSAIC)
            organic_scores: Dict mapping item_id -> final MOSAIC score (fallback if base_scores not provided)
            ad_candidates: List of candidate ads
            activation_p: Mission distribution for alignment scoring
            max_ads: Maximum number of ads to insert
            base_scores: Optional base ranker scores for boundary confidence
            boundary_confidences: Optional precomputed boundary confidence map

        Returns:
            LENSResult with final feed and placement details
        """
        if not ad_candidates or not organic_ranking:
            return LENSResult(
                final_feed=[("organic", item) for item in organic_ranking],
                placements=[],
                blocked_slots=[],
                total_ad_value=0.0,
            )

        # Compute boundary confidences (prefer precomputed or base scores)
        if boundary_confidences is not None:
            boundaries = boundary_confidences
        else:
            score_source = base_scores or organic_scores
            boundaries = self._compute_boundaries(organic_ranking, score_source)

        # Enforce max ad load
        max_ads_allowed = max_ads
        if self.config.max_ad_load is not None:
            max_load_ads = int(np.floor(self.config.max_ad_load * len(organic_ranking)))
            max_ads_allowed = min(max_ads_allowed, max_load_ads)
        if max_ads_allowed <= 0:
            return LENSResult(
                final_feed=[("organic", item) for item in organic_ranking],
                placements=[],
                blocked_slots=[],
                total_ad_value=0.0,
            )

        # Identify eligible slots
        eligible_slots, blocked_slots = self._get_eligible_slots(
            boundaries, len(organic_ranking)
        )

        # Score ads
        ad_scoring = self._score_ads(ad_candidates, activation_p)

        # Assign ads to slots
        assignments = self._assign_ads_to_slots(
            ad_candidates=ad_candidates,
            ad_utilities={k: v["utility"] for k, v in ad_scoring.items()},
            eligible_slots=eligible_slots,
            boundaries=boundaries,
            max_ads=max_ads_allowed,
        )

        # Build final feed
        final_feed, placements = self._build_final_feed(
            organic_ranking=organic_ranking,
            assignments=assignments,
            ad_scoring=ad_scoring,
            boundaries=boundaries,
        )

        total_value = sum(p.utility for p in placements)

        return LENSResult(
            final_feed=final_feed,
            placements=placements,
            blocked_slots=blocked_slots,
            total_ad_value=total_value,
        )

    def _compute_boundaries(
        self,
        organic_ranking: List[int],
        organic_scores: Dict[int, float],
    ) -> Dict[int, float]:
        """Compute confidence at each boundary from base or fallback scores."""
        boundaries = {}

        for k in range(len(organic_ranking) - 1):
            item_k = organic_ranking[k]
            item_k1 = organic_ranking[k + 1]

            gap = organic_scores.get(item_k, 0) - organic_scores.get(item_k1, 0)

            if self.calibration:
                conf = self.calibration.gap_to_conf(max(0, gap))
            else:
                # Fallback: linear mapping from gap to confidence
                conf = min(1.0, 0.5 + gap * 5)

            boundaries[k] = conf

        return boundaries

    def _get_eligible_slots(
        self,
        boundaries: Dict[int, float],
        n_organic: int,
    ) -> Tuple[List[int], List[int]]:
        """Get slots eligible for ad insertion (low confidence boundaries)."""
        eligible = []
        blocked = []

        allowed_report = self.config.allowed_slots
        if allowed_report is None:
            # Default: every 5th position starting at 3 (1-indexed)
            if self.config.slot_indexing != "zero":
                allowed_report = list(range(3, n_organic + 1, 5))
            else:
                allowed_report = list(range(2, n_organic, 5))

        if self.config.slot_indexing != "zero":
            allowed_internal = [slot - 1 for slot in allowed_report]
        else:
            allowed_internal = list(allowed_report)

        for slot_internal, slot_report in zip(allowed_internal, allowed_report):
            if slot_internal < 0 or slot_internal >= n_organic:
                continue

            # Check boundary confidence
            boundary_idx = slot_internal - 1  # Boundary before this position
            if boundary_idx < 0:
                continue

            conf = boundaries.get(boundary_idx, 1.0)

            if conf < self.config.rho_insert:
                eligible.append(slot_internal)
            else:
                blocked.append(slot_report)

        return eligible, blocked

    def _score_ads(
        self,
        ad_candidates: List[AdCandidate],
        activation_p: Optional[np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        """Compute utility score for each ad."""
        scores = {}

        for ad in ad_candidates:
            # Expected value
            ev = ad.throttle * ad.bid * ad.p_click * ad.quality_score

            # Mission alignment
            if activation_p is not None and ad.moment_affinities is not None:
                alignment = float(np.dot(ad.moment_affinities, activation_p))
            else:
                alignment = 0.0

            utility = ev + self.config.lambda_ad * alignment
            scores[ad.ad_id] = {
                "ev": ev,
                "alignment": alignment,
                "utility": utility,
            }

        return scores

    def _assign_ads_to_slots(
        self,
        ad_candidates: List[AdCandidate],
        ad_utilities: Dict[str, float],
        eligible_slots: List[int],
        boundaries: Dict[int, float],
        max_ads: int,
    ) -> List[Tuple[str, int]]:
        """Greedy assignment of ads to slots."""
        if not eligible_slots or not ad_candidates:
            return []

        # Build gain matrix: (gain, ad_id, slot)
        gains = []
        for ad in ad_candidates:
            for slot in eligible_slots:
                # Position effect
                pos_mult = self.config.position_decay ** slot

                # Boundary confidence as cost
                boundary_idx = slot - 1
                conf = boundaries.get(boundary_idx, 0.5)
                cost = conf * self.config.boundary_cost_weight

                gain = ad_utilities[ad.ad_id] * pos_mult - cost
                gains.append((gain, ad.ad_id, slot))

        # Sort by gain (descending)
        gains.sort(reverse=True)

        # Greedy assignment with constraints
        assignments = []
        used_ads = set()
        used_slots = set()

        for gain, ad_id, slot in gains:
            if len(assignments) >= max_ads:
                break

            if ad_id in used_ads:
                continue

            if slot in used_slots:
                continue

            # Check spacing constraint
            too_close = False
            for _, used_slot in assignments:
                if abs(slot - used_slot) < self.config.min_spacing:
                    too_close = True
                    break

            if too_close:
                continue

            assignments.append((ad_id, slot))
            used_ads.add(ad_id)
            used_slots.add(slot)

        return assignments

    def _build_final_feed(
        self,
        organic_ranking: List[int],
        assignments: List[Tuple[str, int]],
        ad_scoring: Dict[str, Dict[str, float]],
        boundaries: Dict[int, float],
    ) -> Tuple[List[Tuple[str, Any]], List[AdPlacement]]:
        """Build final feed with ads inserted."""
        # Sort assignments by slot position
        assignments_sorted = sorted(assignments, key=lambda x: x[1])

        final_feed = []
        placements = []

        ad_idx = 0
        organic_idx = 0

        for pos in range(len(organic_ranking) + len(assignments)):
            # Check if we should insert an ad at this position
            if ad_idx < len(assignments_sorted):
                ad_id, target_slot = assignments_sorted[ad_idx]
                actual_pos = target_slot + ad_idx  # Adjust for previously inserted ads

                if pos == actual_pos:
                    final_feed.append(("sponsored", ad_id))

                    boundary_idx = target_slot - 1
                    placements.append(AdPlacement(
                        ad_id=ad_id,
                        slot_position=pos,
                        organic_boundary=boundary_idx,
                        boundary_confidence=boundaries.get(boundary_idx, 0.5),
                        expected_value=ad_scoring.get(ad_id, {}).get("ev", 0.0),
                        mission_alignment=ad_scoring.get(ad_id, {}).get("alignment", 0.0),
                        utility=ad_scoring.get(ad_id, {}).get("utility", 0.0),
                    ))

                    ad_idx += 1
                    continue

            # Insert organic item
            if organic_idx < len(organic_ranking):
                final_feed.append(("organic", organic_ranking[organic_idx]))
                organic_idx += 1

        return final_feed, placements


def create_mock_ad(
    ad_id: str,
    bid: float = 1.0,
    p_click: float = 0.05,
    quality: float = 0.8,
    category: str = None,
) -> AdCandidate:
    """Helper to create a mock ad for testing."""
    return AdCandidate(
        ad_id=ad_id,
        campaign_id=f"campaign_{ad_id}",
        bid=bid,
        p_click=p_click,
        quality_score=quality,
        category=category,
    )
