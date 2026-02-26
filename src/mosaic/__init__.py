"""
MOSAIC: Margin-Orthogonal Mission Steering for Accuracy-Preserving, Controllable Recommendations.

governed-rank — a governed ranking framework that treats shopping missions as a
control layer, not an accuracy layer.
"""

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

__all__ = [
    "MOSAICScorer",
    "MOSAICConfig",
    "MOSAICResult",
    "MOSAICReceipt",
    "orthogonalize_against_base",
    "OrthogonalizationResult",
    "CalibrationResult",
    "CalibrationConfig",
    "learn_gap_calibration",
    "get_protected_edges",
    "get_protected_edges_by_budget",
    "isotonic_project_on_runs",
    "compute_final_ranking",
]

__version__ = "0.1.0"
