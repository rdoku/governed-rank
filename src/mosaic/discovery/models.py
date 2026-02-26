"""
Data models for the Discovery Engine.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json


class ActionType(Enum):
    """Recommended action based on preference lift."""
    PROMOTE = "PROMOTE"        # Lift > 1.5: Strong unmet demand
    BOOST = "BOOST"            # Lift 1.2-1.5: Notable preference
    NEUTRAL = "NEUTRAL"        # Lift 0.8-1.2: Supply matches demand
    REDUCE = "REDUCE"          # Lift 0.5-0.8: Slight oversupply
    CUT = "CUT"                # Lift < 0.5: Major oversupply


@dataclass
class SegmentStats:
    """Statistics for a user segment."""
    name: str
    n_sessions: int
    n_reads: int
    top_category: str
    top_category_lift: float


@dataclass
class DiscoveredObjective:
    """A discovered user preference that could become a business policy."""

    # Core identification
    category: str                  # The content category/attribute
    segment: str                   # User segment (morning, evening, all, etc.)

    # Preference metrics
    preference_lift: float         # P(cat | reads) / P(cat | catalog)
    catalog_rate: float            # P(cat | catalog) - base rate (0-1)
    user_rate: float               # P(cat | reads) - observed rate (0-1)
    n_reads: int                   # Number of reads in this category

    # Statistical confidence (optional)
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

    # Business interpretation
    action: ActionType = ActionType.NEUTRAL
    confidence: str = "MEDIUM"     # HIGH, MEDIUM, LOW
    business_insight: str = ""

    def __post_init__(self):
        """Set action and insight based on lift."""
        if self.preference_lift >= 1.5:
            self.action = ActionType.PROMOTE
            self.confidence = "HIGH"
            self.business_insight = f"Strong unmet demand: {self.category} is {self.preference_lift:.1f}× more preferred than supplied"
        elif self.preference_lift >= 1.2:
            self.action = ActionType.BOOST
            self.confidence = "MEDIUM"
            self.business_insight = f"Notable preference: {self.category} over-indexes by {self.preference_lift:.1f}×"
        elif self.preference_lift >= 0.8:
            self.action = ActionType.NEUTRAL
            self.confidence = "LOW"
            self.business_insight = f"Balanced: {self.category} supply matches demand"
        elif self.preference_lift >= 0.5:
            self.action = ActionType.REDUCE
            self.confidence = "MEDIUM"
            self.business_insight = f"Slight oversupply: {self.category} under-indexes at {self.preference_lift:.1f}×"
        else:
            self.action = ActionType.CUT
            self.confidence = "HIGH"
            self.business_insight = f"Major oversupply: {self.category} is {1/self.preference_lift:.1f}× over-supplied"

    @property
    def expected_engagement_boost(self) -> str:
        """Expected engagement improvement if this content is promoted."""
        if self.preference_lift > 1.0:
            boost = (self.preference_lift - 1.0) * 100
            return f"+{boost:.0f}%"
        else:
            drop = (1.0 - self.preference_lift) * 100
            return f"-{drop:.0f}%"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "category": self.category,
            "segment": self.segment,
            "preference_lift": round(self.preference_lift, 3),
            "catalog_rate": round(self.catalog_rate * 100, 2),  # As percentage
            "user_rate": round(self.user_rate * 100, 2),        # As percentage
            "n_reads": self.n_reads,
            "action": self.action.value,
            "confidence": self.confidence,
            "business_insight": self.business_insight,
            "expected_engagement_boost": self.expected_engagement_boost,
            "p_value": self.p_value,
            "confidence_interval": self.confidence_interval,
        }

    def to_api_response(self) -> Dict[str, Any]:
        """Format for API response."""
        return {
            "category": self.category,
            "segment": self.segment,
            "preference_lift": round(self.preference_lift, 2),
            "catalog_rate_pct": round(self.catalog_rate * 100, 1),
            "user_rate_pct": round(self.user_rate * 100, 1),
            "n_reads": self.n_reads,
            "action": self.action.value,
            "confidence": self.confidence,
            "insight": self.business_insight,
            "expected_boost": self.expected_engagement_boost,
        }


@dataclass
class DiscoveryReport:
    """Complete discovery report with all findings."""

    # Metadata
    dataset: str
    timestamp: str
    n_sessions: int
    n_articles: int
    n_categories: int

    # Segment breakdown
    segments: Dict[str, SegmentStats] = field(default_factory=dict)

    # All discoveries
    discoveries: List[DiscoveredObjective] = field(default_factory=list)

    # Filtered views
    @property
    def opportunities(self) -> List[DiscoveredObjective]:
        """High-value opportunities (lift > 1.2)."""
        return [d for d in self.discoveries if d.preference_lift > 1.2]

    @property
    def oversupply(self) -> List[DiscoveredObjective]:
        """Oversupply warnings (lift < 0.7)."""
        return [d for d in self.discoveries if d.preference_lift < 0.7]

    @property
    def by_segment(self) -> Dict[str, List[DiscoveredObjective]]:
        """Group discoveries by segment."""
        result = {}
        for d in self.discoveries:
            if d.segment not in result:
                result[d.segment] = []
            result[d.segment].append(d)
        return result

    def top_opportunities(self, n: int = 5) -> List[DiscoveredObjective]:
        """Top N opportunities by lift."""
        sorted_opps = sorted(self.opportunities, key=lambda x: -x.preference_lift)
        return sorted_opps[:n]

    def top_oversupply(self, n: int = 5) -> List[DiscoveredObjective]:
        """Top N oversupply warnings by inverse lift."""
        sorted_over = sorted(self.oversupply, key=lambda x: x.preference_lift)
        return sorted_over[:n]

    def segment_comparison(self, category: str) -> Dict[str, float]:
        """Compare lift for a category across segments."""
        return {
            d.segment: d.preference_lift
            for d in self.discoveries
            if d.category == category
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "dataset": self.dataset,
                "timestamp": self.timestamp,
                "n_sessions": self.n_sessions,
                "n_articles": self.n_articles,
                "n_categories": self.n_categories,
            },
            "summary": {
                "total_discoveries": len(self.discoveries),
                "high_value_opportunities": len(self.opportunities),
                "oversupply_warnings": len(self.oversupply),
            },
            "segments": {
                name: asdict(stats) for name, stats in self.segments.items()
            },
            "discoveries": [d.to_dict() for d in self.discoveries],
            "top_opportunities": [d.to_dict() for d in self.top_opportunities()],
            "top_oversupply": [d.to_dict() for d in self.top_oversupply()],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        """Save report to JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json())

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "="*70)
        print("DISCOVERY REPORT SUMMARY")
        print("="*70)
        print(f"\nDataset: {self.dataset}")
        print(f"Sessions: {self.n_sessions:,} | Articles: {self.n_articles:,} | Categories: {self.n_categories}")
        print(f"\nFindings: {len(self.discoveries)} total")
        print(f"  - {len(self.opportunities)} high-value opportunities (lift > 1.2)")
        print(f"  - {len(self.oversupply)} oversupply warnings (lift < 0.7)")

        if self.opportunities:
            print("\n" + "-"*70)
            print("TOP OPPORTUNITIES")
            print("-"*70)
            for opp in self.top_opportunities(3):
                print(f"\n  {opp.category.upper()} ({opp.segment})")
                print(f"    Lift: {opp.preference_lift:.2f}x | Action: {opp.action.value}")
                print(f"    {opp.business_insight}")

        if self.oversupply:
            print("\n" + "-"*70)
            print("TOP OVERSUPPLY WARNINGS")
            print("-"*70)
            for warn in self.top_oversupply(3):
                print(f"\n  {warn.category.upper()} ({warn.segment})")
                print(f"    Lift: {warn.preference_lift:.2f}x | Action: {warn.action.value}")
                print(f"    {warn.business_insight}")
