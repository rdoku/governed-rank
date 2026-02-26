"""
MOSAIC Objective Discovery Engine
=================================

Automatically discovers user preferences and content gaps from behavioral data.

Example usage:
    from mosaic.discovery import DiscoveryEngine, DiscoveryConfig

    engine = DiscoveryEngine(config=DiscoveryConfig(min_lift=1.2))
    report = engine.discover(sessions, catalog)

    for opp in report.opportunities:
        print(f"{opp.category}: {opp.lift}x lift in {opp.segment}")
"""

from .engine import DiscoveryEngine, DiscoveryConfig
from .models import (
    DiscoveredObjective,
    DiscoveryReport,
    SegmentStats,
    ActionType,
)

__all__ = [
    "DiscoveryEngine",
    "DiscoveryConfig",
    "DiscoveredObjective",
    "DiscoveryReport",
    "SegmentStats",
    "ActionType",
]
