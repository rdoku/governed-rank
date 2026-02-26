"""
MOSAIC Objective Discovery Engine

Automatically discovers user preferences from behavioral data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np

from .models import (
    DiscoveredObjective,
    DiscoveryReport,
    SegmentStats,
    ActionType,
)


@dataclass
class DiscoveryConfig:
    """Configuration for the Discovery Engine."""

    # Filtering thresholds
    min_lift: float = 1.2              # Minimum lift to report as opportunity
    max_lift: float = 0.7              # Maximum lift to report as oversupply
    min_catalog_rate: float = 0.01     # Ignore categories < 1% of catalog
    min_reads: int = 30                # Minimum reads for significance

    # Segmentation
    enable_time_segments: bool = True   # Morning/evening segmentation
    hour_cutoff: int = 12               # Hour to split morning/evening

    # Statistical options
    compute_confidence: bool = False    # Compute bootstrap CIs (slower)
    n_bootstrap: int = 200              # Bootstrap iterations
    alpha: float = 0.05                 # Significance level

    # Category extraction
    category_field: str = "category"    # Field name in catalog items


class DiscoveryEngine:
    """
    Automatic Objective Discovery Engine.

    Discovers user preferences by computing preference lift:
        Lift = P(category | user_reads) / P(category | catalog)

    Usage:
        engine = DiscoveryEngine()
        report = engine.discover(sessions, catalog)

        for opp in report.opportunities:
            print(f"{opp.category}: {opp.lift}x in {opp.segment}")
    """

    def __init__(self, config: Optional[DiscoveryConfig] = None):
        """
        Initialize the Discovery Engine.

        Args:
            config: Configuration options (uses defaults if None)
        """
        self.config = config or DiscoveryConfig()

    def discover(
        self,
        sessions: Dict[str, List[Dict]],
        catalog: Dict[str, Dict],
        dataset_name: str = "unknown",
    ) -> DiscoveryReport:
        """
        Run full discovery pipeline.

        Args:
            sessions: Dict mapping session_id -> list of events
                      Each event should have 'article_id' and optionally 'timestamp'
            catalog: Dict mapping article_id -> article metadata
                     Each article should have a 'category' field (configurable)
            dataset_name: Name for the report

        Returns:
            DiscoveryReport with all findings
        """
        # Step 1: Compute catalog distribution
        catalog_dist = self._compute_catalog_distribution(catalog)

        # Step 2: Segment sessions
        segments = self._segment_sessions(sessions, catalog)

        # Step 3: Compute lifts for each segment
        discoveries = []
        segment_stats = {}

        for segment_name, reads in segments.items():
            if len(reads) < self.config.min_reads:
                continue

            # Compute user distribution for this segment
            user_dist = self._compute_user_distribution(reads, catalog)

            # Compute lifts
            segment_discoveries = self._compute_lifts(
                catalog_dist, user_dist, segment_name, len(reads)
            )
            discoveries.extend(segment_discoveries)

            # Compute segment stats
            if segment_discoveries:
                top = max(segment_discoveries, key=lambda x: x.preference_lift)
                segment_stats[segment_name] = SegmentStats(
                    name=segment_name,
                    n_sessions=len(set(reads)),  # Approximate
                    n_reads=len(reads),
                    top_category=top.category,
                    top_category_lift=top.preference_lift,
                )

        # Sort by lift magnitude
        discoveries.sort(key=lambda x: abs(x.preference_lift - 1.0), reverse=True)

        # Build report
        return DiscoveryReport(
            dataset=dataset_name,
            timestamp=datetime.now().isoformat(),
            n_sessions=len(sessions),
            n_articles=len(catalog),
            n_categories=len(catalog_dist),
            segments=segment_stats,
            discoveries=discoveries,
        )

    def discover_for_segment(
        self,
        reads: List[str],
        catalog: Dict[str, Dict],
        segment_name: str = "custom",
    ) -> List[DiscoveredObjective]:
        """
        Run discovery for a specific segment/list of reads.

        Args:
            reads: List of article IDs that were read
            catalog: Full catalog
            segment_name: Name for this segment

        Returns:
            List of discoveries for this segment
        """
        catalog_dist = self._compute_catalog_distribution(catalog)
        user_dist = self._compute_user_distribution(reads, catalog)
        return self._compute_lifts(catalog_dist, user_dist, segment_name, len(reads))

    def compare_segments(
        self,
        sessions: Dict[str, List[Dict]],
        catalog: Dict[str, Dict],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare preference lifts across segments for each category.

        Returns:
            Dict[category -> Dict[segment -> lift]]
        """
        segments = self._segment_sessions(sessions, catalog)
        catalog_dist = self._compute_catalog_distribution(catalog)

        result = defaultdict(dict)

        for segment_name, reads in segments.items():
            if len(reads) < self.config.min_reads:
                continue

            user_dist = self._compute_user_distribution(reads, catalog)

            for category in catalog_dist:
                if catalog_dist[category] >= self.config.min_catalog_rate:
                    user_rate = user_dist.get(category, 0)
                    lift = user_rate / catalog_dist[category] if catalog_dist[category] > 0 else 1.0
                    result[category][segment_name] = round(lift, 2)

        return dict(result)

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _compute_catalog_distribution(
        self,
        catalog: Dict[str, Dict],
    ) -> Dict[str, float]:
        """Compute category distribution in the catalog."""
        category_counts = defaultdict(int)
        category_field = self.config.category_field

        for item in catalog.values():
            cat = item.get(category_field, "unknown")
            category_counts[cat] += 1

        total = len(catalog)
        return {cat: count / total for cat, count in category_counts.items()}

    def _compute_user_distribution(
        self,
        reads: List[str],
        catalog: Dict[str, Dict],
    ) -> Dict[str, float]:
        """Compute category distribution in user reads."""
        category_counts = defaultdict(int)
        category_field = self.config.category_field
        valid_reads = 0

        for article_id in reads:
            item = catalog.get(article_id)
            if item:
                cat = item.get(category_field, "unknown")
                category_counts[cat] += 1
                valid_reads += 1

        if valid_reads == 0:
            return {}

        return {cat: count / valid_reads for cat, count in category_counts.items()}

    def _segment_sessions(
        self,
        sessions: Dict[str, List[Dict]],
        catalog: Dict[str, Dict],
    ) -> Dict[str, List[str]]:
        """Segment sessions and extract reads per segment."""
        segments = {
            "all": [],
        }

        if self.config.enable_time_segments:
            segments["morning"] = []
            segments["evening"] = []

        for sid, events in sessions.items():
            if not events:
                continue

            # Extract article IDs
            articles = []
            hours = []
            for e in events:
                aid = e.get("article_id")
                if aid and aid in catalog:
                    articles.append(aid)
                ts = e.get("timestamp")
                if ts:
                    try:
                        hours.append(datetime.utcfromtimestamp(ts).hour)
                    except (ValueError, TypeError, OSError):
                        pass

            if not articles:
                continue

            # Add to "all" segment
            segments["all"].extend(articles)

            # Add to time segments
            if self.config.enable_time_segments and hours:
                median_hour = int(np.median(hours))
                if median_hour < self.config.hour_cutoff:
                    segments["morning"].extend(articles)
                else:
                    segments["evening"].extend(articles)

        return segments

    def _compute_lifts(
        self,
        catalog_dist: Dict[str, float],
        user_dist: Dict[str, float],
        segment_name: str,
        n_reads: int,
    ) -> List[DiscoveredObjective]:
        """Compute preference lifts for all categories."""
        discoveries = []

        for category, catalog_rate in catalog_dist.items():
            # Skip rare categories
            if catalog_rate < self.config.min_catalog_rate:
                continue

            user_rate = user_dist.get(category, 0)
            n_cat_reads = int(user_rate * n_reads)

            # Skip if not enough reads
            if n_cat_reads < self.config.min_reads:
                continue

            # Compute lift
            lift = user_rate / catalog_rate if catalog_rate > 0 else 1.0

            # Only report significant lifts
            if lift > self.config.min_lift or lift < self.config.max_lift:
                discovery = DiscoveredObjective(
                    category=category,
                    segment=segment_name,
                    preference_lift=lift,
                    catalog_rate=catalog_rate,
                    user_rate=user_rate,
                    n_reads=n_cat_reads,
                )
                discoveries.append(discovery)

        return discoveries

    def _compute_lift_with_ci(
        self,
        reads: List[str],
        catalog: Dict[str, Dict],
        category: str,
        catalog_rate: float,
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Compute lift with bootstrap confidence interval.

        Returns:
            (lift, p_value, (ci_low, ci_high))
        """
        category_field = self.config.category_field

        # Get reads in this category
        read_items = [catalog.get(r, {}) for r in reads if r in catalog]
        if not read_items:
            return 1.0, 1.0, (0.0, float('inf'))

        n = len(read_items)
        matches = sum(1 for item in read_items if item.get(category_field) == category)
        user_rate = matches / n
        lift = user_rate / catalog_rate if catalog_rate > 0 else 1.0

        if not self.config.compute_confidence:
            return lift, None, None

        # Bootstrap
        rng = np.random.default_rng(42)
        bootstrap_lifts = []

        for _ in range(self.config.n_bootstrap):
            sample_idx = rng.choice(n, size=n, replace=True)
            sample_matches = sum(
                1 for i in sample_idx
                if read_items[i].get(category_field) == category
            )
            sample_rate = sample_matches / n
            sample_lift = sample_rate / catalog_rate if catalog_rate > 0 else 1.0
            bootstrap_lifts.append(sample_lift)

        ci_low = float(np.percentile(bootstrap_lifts, 100 * self.config.alpha / 2))
        ci_high = float(np.percentile(bootstrap_lifts, 100 * (1 - self.config.alpha / 2)))

        # Simple p-value approximation
        p_value = sum(1 for bl in bootstrap_lifts if bl <= 1.0) / len(bootstrap_lifts)
        if lift < 1.0:
            p_value = 1 - p_value

        return lift, p_value, (ci_low, ci_high)


# Convenience function for quick discovery
def quick_discover(
    sessions: Dict[str, List[Dict]],
    catalog: Dict[str, Dict],
    dataset_name: str = "dataset",
) -> DiscoveryReport:
    """
    Quick discovery with default settings.

    Args:
        sessions: Session data
        catalog: Article catalog
        dataset_name: Name for the report

    Returns:
        DiscoveryReport
    """
    engine = DiscoveryEngine()
    return engine.discover(sessions, catalog, dataset_name)
