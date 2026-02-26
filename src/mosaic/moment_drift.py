"""Moment Drift Monitor: detect semantic drift between model rebuilds.

Like a biological immune system - detects when moments have silently
changed meaning, protecting against stale labels applied to new semantics.

Key insight: If "Moment 3" used to mean "Breakfast" but now means "Snacks",
we shouldn't keep showing "Perfect for breakfast" labels.

Approach:
1. Store a signature for each moment (top departments, centroid checksum)
2. On rebuild, compare similarity to previous signature
3. If drift exceeds threshold, flag for review or quarantine labels
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MomentSignature:
    """Signature capturing a moment's semantic identity."""
    moment_id: int
    label: Optional[str] = None

    # Top categories distribution (normalized)
    top_departments: Dict[str, float] = field(default_factory=dict)
    top_aisles: Dict[str, float] = field(default_factory=dict)

    # Centroid in item space
    centroid_checksum: str = ""

    # Stats
    num_items: int = 0
    mean_affinity: float = 0.0

    # Timestamp
    computed_at: str = ""


@dataclass
class DriftReport:
    """Report on drift between two moment signatures."""
    moment_id: int
    old_label: Optional[str]
    new_label: Optional[str]

    # Drift scores (0 = identical, 1 = completely different)
    department_drift: float = 0.0
    aisle_drift: float = 0.0
    centroid_drift: float = 0.0  # 1 if checksum differs
    overall_drift: float = 0.0

    # Status
    status: str = "stable"  # stable, drifted, quarantined
    needs_review: bool = False
    message: str = ""


def compute_category_distribution(
    moment2vec: np.ndarray,
    moment_id: int,
    catalog: Dict[int, dict],
    category_key: str = "dept",
    threshold: float = 0.3,
) -> Dict[str, float]:
    """Compute category distribution for items strongly belonging to a moment.

    Args:
        moment2vec: (N, K) affinity matrix
        moment_id: Which moment to analyze
        catalog: item_id -> {dept, aisle, ...}
        category_key: Which category to extract ("dept" or "aisle")
        threshold: Minimum affinity to include item

    Returns:
        Normalized distribution over categories
    """
    from collections import Counter

    if moment_id >= moment2vec.shape[1]:
        return {}

    # Get items with high affinity for this moment
    affinities = moment2vec[:, moment_id]
    strong_items = np.where(affinities >= threshold)[0]

    # Count categories weighted by affinity
    category_weights = Counter()
    for idx in strong_items:
        if idx in catalog:
            cat = catalog[idx].get(category_key, "unknown")
            if cat:
                category_weights[cat] += float(affinities[idx])

    # Normalize
    total = sum(category_weights.values())
    if total > 0:
        return {k: v / total for k, v in category_weights.most_common(10)}
    return {}


def compute_moment_centroid(
    moment2vec: np.ndarray,
    moment_id: int,
    item_embeddings: Optional[np.ndarray] = None,
) -> Tuple[str, np.ndarray]:
    """Compute centroid of items belonging to a moment.

    If item_embeddings provided, uses weighted centroid in embedding space.
    Otherwise uses the moment column itself as the signature.

    Returns:
        (checksum, centroid_vector)
    """
    if moment_id >= moment2vec.shape[1]:
        return "", np.array([])

    # Use moment column as signature (weighted average of all items)
    affinities = moment2vec[:, moment_id]

    if item_embeddings is not None and len(item_embeddings) == len(affinities):
        # Weighted centroid in embedding space
        weights = affinities / (affinities.sum() + 1e-10)
        centroid = (item_embeddings.T @ weights).astype(np.float32)
    else:
        # Use affinity distribution as signature
        centroid = affinities.astype(np.float32)

    checksum = hashlib.sha256(centroid.tobytes()).hexdigest()[:16]
    return checksum, centroid


def compute_moment_signature(
    moment2vec: np.ndarray,
    moment_id: int,
    catalog: Dict[int, dict],
    label: Optional[str] = None,
    item_embeddings: Optional[np.ndarray] = None,
) -> MomentSignature:
    """Compute full signature for a moment."""
    affinities = moment2vec[:, moment_id] if moment_id < moment2vec.shape[1] else np.array([])

    top_depts = compute_category_distribution(moment2vec, moment_id, catalog, "dept")
    top_aisles = compute_category_distribution(moment2vec, moment_id, catalog, "aisle")
    centroid_checksum, _ = compute_moment_centroid(moment2vec, moment_id, item_embeddings)

    num_items = int((affinities >= 0.3).sum()) if len(affinities) > 0 else 0
    mean_aff = float(affinities.mean()) if len(affinities) > 0 else 0.0

    return MomentSignature(
        moment_id=moment_id,
        label=label,
        top_departments=top_depts,
        top_aisles=top_aisles,
        centroid_checksum=centroid_checksum,
        num_items=num_items,
        mean_affinity=round(mean_aff, 4),
        computed_at=datetime.utcnow().isoformat(),
    )


def compute_distribution_drift(
    old_dist: Dict[str, float],
    new_dist: Dict[str, float],
) -> float:
    """Compute drift between two category distributions.

    Uses Jensen-Shannon-like measure: average of how much each
    distribution differs from the other.

    Returns:
        Drift score in [0, 1] where 0 = identical, 1 = no overlap
    """
    if not old_dist and not new_dist:
        return 0.0
    if not old_dist or not new_dist:
        return 1.0

    all_keys = set(old_dist.keys()) | set(new_dist.keys())

    # Compute symmetric KL-like divergence
    drift = 0.0
    for key in all_keys:
        p = old_dist.get(key, 0.0)
        q = new_dist.get(key, 0.0)
        # Absolute difference weighted by magnitude
        drift += abs(p - q)

    # Normalize to [0, 1]
    return min(drift / 2.0, 1.0)


def compute_drift(
    old_sig: MomentSignature,
    new_sig: MomentSignature,
    dept_weight: float = 0.5,
    aisle_weight: float = 0.3,
    centroid_weight: float = 0.2,
) -> DriftReport:
    """Compute drift between old and new moment signatures.

    Args:
        old_sig: Previous moment signature
        new_sig: New moment signature
        dept_weight: Weight for department drift
        aisle_weight: Weight for aisle drift
        centroid_weight: Weight for centroid drift

    Returns:
        DriftReport with drift scores and status
    """
    dept_drift = compute_distribution_drift(old_sig.top_departments, new_sig.top_departments)
    aisle_drift = compute_distribution_drift(old_sig.top_aisles, new_sig.top_aisles)
    centroid_drift = 0.0 if old_sig.centroid_checksum == new_sig.centroid_checksum else 1.0

    # Weighted overall drift
    overall = (
        dept_weight * dept_drift +
        aisle_weight * aisle_drift +
        centroid_weight * centroid_drift
    )

    # Determine status
    if overall >= 0.5:
        status = "quarantined"
        needs_review = True
        message = f"High drift detected ({overall:.2f}): moment may have changed meaning"
    elif overall >= 0.25:
        status = "drifted"
        needs_review = True
        message = f"Moderate drift ({overall:.2f}): review recommended"
    else:
        status = "stable"
        needs_review = False
        message = f"Low drift ({overall:.2f}): moment semantics stable"

    return DriftReport(
        moment_id=old_sig.moment_id,
        old_label=old_sig.label,
        new_label=new_sig.label,
        department_drift=round(dept_drift, 3),
        aisle_drift=round(aisle_drift, 3),
        centroid_drift=round(centroid_drift, 3),
        overall_drift=round(overall, 3),
        status=status,
        needs_review=needs_review,
        message=message,
    )


def save_signatures(
    signatures: List[MomentSignature],
    output_path: str,
) -> None:
    """Save moment signatures to JSON file."""
    data = []
    for sig in signatures:
        data.append({
            "moment_id": sig.moment_id,
            "label": sig.label,
            "top_departments": sig.top_departments,
            "top_aisles": sig.top_aisles,
            "centroid_checksum": sig.centroid_checksum,
            "num_items": sig.num_items,
            "mean_affinity": sig.mean_affinity,
            "computed_at": sig.computed_at,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_signatures(input_path: str) -> List[MomentSignature]:
    """Load moment signatures from JSON file."""
    if not Path(input_path).exists():
        return []

    with open(input_path) as f:
        data = json.load(f)

    signatures = []
    for item in data:
        signatures.append(MomentSignature(
            moment_id=item["moment_id"],
            label=item.get("label"),
            top_departments=item.get("top_departments", {}),
            top_aisles=item.get("top_aisles", {}),
            centroid_checksum=item.get("centroid_checksum", ""),
            num_items=item.get("num_items", 0),
            mean_affinity=item.get("mean_affinity", 0.0),
            computed_at=item.get("computed_at", ""),
        ))

    return signatures


def check_drift_on_rebuild(
    moment2vec: np.ndarray,
    catalog: Dict[int, dict],
    labels: Optional[List[str]] = None,
    previous_signatures_path: Optional[str] = None,
    item_embeddings: Optional[np.ndarray] = None,
) -> Tuple[List[MomentSignature], List[DriftReport]]:
    """Check for drift when rebuilding moment2vec.

    Args:
        moment2vec: New (N, K) affinity matrix
        catalog: item_id -> {dept, aisle, ...}
        labels: Optional labels for each moment
        previous_signatures_path: Path to previous signatures JSON
        item_embeddings: Optional item embeddings for centroid computation

    Returns:
        (new_signatures, drift_reports)
    """
    K = moment2vec.shape[1]

    # Compute new signatures
    new_signatures = []
    for mid in range(K):
        label = labels[mid] if labels and mid < len(labels) else None
        sig = compute_moment_signature(moment2vec, mid, catalog, label, item_embeddings)
        new_signatures.append(sig)

    # Load previous signatures if available
    old_signatures = []
    if previous_signatures_path:
        old_signatures = load_signatures(previous_signatures_path)

    # Compute drift for each moment
    drift_reports = []
    old_by_id = {s.moment_id: s for s in old_signatures}

    for new_sig in new_signatures:
        old_sig = old_by_id.get(new_sig.moment_id)
        if old_sig:
            report = compute_drift(old_sig, new_sig)
            drift_reports.append(report)
        else:
            # New moment, no drift to compute
            drift_reports.append(DriftReport(
                moment_id=new_sig.moment_id,
                old_label=None,
                new_label=new_sig.label,
                status="new",
                message="New moment, no previous signature",
            ))

    return new_signatures, drift_reports


def get_quarantined_moments(drift_reports: List[DriftReport]) -> List[int]:
    """Get list of moment IDs that should be quarantined."""
    return [r.moment_id for r in drift_reports if r.status == "quarantined"]


def get_drift_summary(drift_reports: List[DriftReport]) -> dict:
    """Get summary of drift status across all moments."""
    stable = sum(1 for r in drift_reports if r.status == "stable")
    drifted = sum(1 for r in drift_reports if r.status == "drifted")
    quarantined = sum(1 for r in drift_reports if r.status == "quarantined")
    new = sum(1 for r in drift_reports if r.status == "new")

    needs_review = [r.moment_id for r in drift_reports if r.needs_review]
    max_drift = max((r.overall_drift for r in drift_reports if r.overall_drift), default=0.0)

    return {
        "total_moments": len(drift_reports),
        "stable": stable,
        "drifted": drifted,
        "quarantined": quarantined,
        "new": new,
        "needs_review": needs_review,
        "max_drift": max_drift,
        "healthy": quarantined == 0 and drifted == 0,
    }
