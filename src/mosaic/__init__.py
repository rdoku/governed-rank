"""
governed-rank — steer any ranked list toward policy objectives without breaking accuracy.

Works for content moderation, fairness, fraud detection, RAG safety,
merchandising, and any domain where you need governed reranking.
"""

from .govern import govern, GovernResult, GovernReceipt
from .orthogonalization import orthogonalize_against_base, OrthogonalizationResult, compute_target_scores
from .gap_calibration import (
    CalibrationResult,
    CalibrationConfig,
    learn_gap_calibration,
    get_protected_edges,
    get_protected_edges_by_budget,
)
from .isotonic_projection import isotonic_project_on_runs, compute_final_ranking, ProjectionResult

__all__ = [
    # Entry point
    "govern",
    "GovernResult",
    "GovernReceipt",
    # Orthogonalization
    "orthogonalize_against_base",
    "OrthogonalizationResult",
    "compute_target_scores",
    # Gap calibration
    "CalibrationResult",
    "CalibrationConfig",
    "learn_gap_calibration",
    "get_protected_edges",
    "get_protected_edges_by_budget",
    # Isotonic projection
    "isotonic_project_on_runs",
    "compute_final_ranking",
    "ProjectionResult",
]

__version__ = "0.1.0"
