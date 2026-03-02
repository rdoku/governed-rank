"""
governed-rank — steer any ranked list toward policy objectives without breaking accuracy.

Works for content moderation, fairness, fraud detection, RAG safety,
merchandising, and any domain where you need governed reranking.
"""

from .govern import govern, GovernResult, GovernReceipt
from .mosaic_scorer import MOSAICScorer, MOSAICConfig, MOSAICResult, MOSAICReceipt
from .orthogonalization import orthogonalize_against_base, OrthogonalizationResult
from .gap_calibration import (
    CalibrationResult,
    CalibrationConfig,
    learn_gap_calibration,
    get_protected_edges,
    get_protected_edges_by_budget,
)
from .isotonic_projection import isotonic_project_on_runs, compute_final_ranking
from .discovery import (
    DiscoveryEngine,
    DiscoveryConfig,
    DiscoveredObjective,
    DiscoveryReport,
    ActionType,
)

__all__ = [
    # Simple API
    "govern",
    "GovernResult",
    "GovernReceipt",
    # Full pipeline
    "MOSAICScorer",
    "MOSAICConfig",
    "MOSAICResult",
    "MOSAICReceipt",
    # Core stages
    "orthogonalize_against_base",
    "OrthogonalizationResult",
    "CalibrationResult",
    "CalibrationConfig",
    "learn_gap_calibration",
    "get_protected_edges",
    "get_protected_edges_by_budget",
    "isotonic_project_on_runs",
    "compute_final_ranking",
    # Discovery
    "DiscoveryEngine",
    "DiscoveryConfig",
    "DiscoveredObjective",
    "DiscoveryReport",
    "ActionType",
]

__version__ = "0.1.0"
