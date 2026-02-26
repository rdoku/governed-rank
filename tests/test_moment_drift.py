"""Tests for moment drift monitoring."""
import numpy as np
import pytest
import tempfile
import os

from mosaic.moment_drift import (
    MomentSignature,
    DriftReport,
    compute_category_distribution,
    compute_moment_centroid,
    compute_moment_signature,
    compute_distribution_drift,
    compute_drift,
    save_signatures,
    load_signatures,
    check_drift_on_rebuild,
    get_quarantined_moments,
    get_drift_summary,
)


@pytest.fixture
def sample_moment2vec():
    """Create sample moment2vec matrix."""
    # 10 items, 4 moments
    np.random.seed(42)
    m2v = np.random.rand(10, 4).astype(np.float32)
    # Normalize to sum to 1 per item
    m2v = m2v / m2v.sum(axis=1, keepdims=True)
    return m2v


@pytest.fixture
def sample_catalog():
    """Create sample catalog with dept/aisle info."""
    return {
        0: {"dept": "produce", "aisle": "fruits"},
        1: {"dept": "produce", "aisle": "vegetables"},
        2: {"dept": "dairy", "aisle": "milk"},
        3: {"dept": "dairy", "aisle": "cheese"},
        4: {"dept": "bakery", "aisle": "bread"},
        5: {"dept": "bakery", "aisle": "pastries"},
        6: {"dept": "produce", "aisle": "fruits"},
        7: {"dept": "dairy", "aisle": "yogurt"},
        8: {"dept": "snacks", "aisle": "chips"},
        9: {"dept": "snacks", "aisle": "cookies"},
    }


def test_compute_category_distribution(sample_moment2vec, sample_catalog):
    """Category distribution computed correctly."""
    dist = compute_category_distribution(
        sample_moment2vec, moment_id=0, catalog=sample_catalog,
        category_key="dept", threshold=0.2
    )

    assert isinstance(dist, dict)
    # Should have some departments
    assert len(dist) > 0
    # Should be normalized
    total = sum(dist.values())
    assert 0.99 <= total <= 1.01 or total == 0


def test_compute_category_distribution_invalid_moment():
    """Invalid moment ID returns empty dict."""
    m2v = np.random.rand(5, 3).astype(np.float32)
    catalog = {0: {"dept": "test"}}

    dist = compute_category_distribution(m2v, moment_id=10, catalog=catalog)
    assert dist == {}


def test_compute_moment_centroid(sample_moment2vec):
    """Centroid computation produces checksum and vector."""
    checksum, centroid = compute_moment_centroid(sample_moment2vec, moment_id=0)

    assert isinstance(checksum, str)
    assert len(checksum) == 16  # SHA256 truncated to 16 chars
    assert isinstance(centroid, np.ndarray)
    assert len(centroid) == len(sample_moment2vec)  # Same as num items


def test_compute_moment_centroid_with_embeddings():
    """Centroid uses embeddings when provided."""
    m2v = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]], dtype=np.float32)
    embeddings = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    checksum, centroid = compute_moment_centroid(m2v, moment_id=0, item_embeddings=embeddings)

    assert len(centroid) == 3  # Embedding dimension


def test_compute_moment_signature(sample_moment2vec, sample_catalog):
    """Full signature computation."""
    sig = compute_moment_signature(
        sample_moment2vec, moment_id=0, catalog=sample_catalog, label="Test Moment"
    )

    assert sig.moment_id == 0
    assert sig.label == "Test Moment"
    assert isinstance(sig.top_departments, dict)
    assert isinstance(sig.top_aisles, dict)
    assert len(sig.centroid_checksum) == 16
    assert sig.num_items >= 0
    assert sig.computed_at != ""


def test_compute_distribution_drift_identical():
    """Identical distributions have zero drift."""
    dist = {"a": 0.5, "b": 0.3, "c": 0.2}

    drift = compute_distribution_drift(dist, dist)
    assert drift == 0.0


def test_compute_distribution_drift_no_overlap():
    """Completely different distributions have max drift."""
    old = {"a": 0.5, "b": 0.5}
    new = {"c": 0.5, "d": 0.5}

    drift = compute_distribution_drift(old, new)
    assert drift == 1.0


def test_compute_distribution_drift_partial():
    """Partial overlap produces moderate drift."""
    old = {"a": 0.6, "b": 0.4}
    new = {"a": 0.4, "b": 0.3, "c": 0.3}

    drift = compute_distribution_drift(old, new)
    assert 0 < drift < 1


def test_compute_distribution_drift_empty():
    """Empty distributions handled correctly."""
    assert compute_distribution_drift({}, {}) == 0.0
    assert compute_distribution_drift({"a": 1.0}, {}) == 1.0
    assert compute_distribution_drift({}, {"a": 1.0}) == 1.0


def test_compute_drift_stable():
    """Low drift produces stable status."""
    old_sig = MomentSignature(
        moment_id=0,
        label="Breakfast",
        top_departments={"produce": 0.5, "dairy": 0.3, "bakery": 0.2},
        top_aisles={"fruits": 0.3, "milk": 0.3, "bread": 0.2, "eggs": 0.2},
        centroid_checksum="abc123",
    )
    new_sig = MomentSignature(
        moment_id=0,
        label="Breakfast",
        top_departments={"produce": 0.45, "dairy": 0.35, "bakery": 0.2},
        top_aisles={"fruits": 0.3, "milk": 0.3, "bread": 0.2, "eggs": 0.2},
        centroid_checksum="abc123",  # Same checksum
    )

    report = compute_drift(old_sig, new_sig)

    assert report.status == "stable"
    assert not report.needs_review
    assert report.overall_drift < 0.25


def test_compute_drift_quarantined():
    """High drift produces quarantined status."""
    old_sig = MomentSignature(
        moment_id=0,
        label="Breakfast",
        top_departments={"produce": 0.6, "dairy": 0.4},
        centroid_checksum="abc123",
    )
    new_sig = MomentSignature(
        moment_id=0,
        label="Breakfast",
        top_departments={"snacks": 0.7, "beverages": 0.3},  # Completely different
        centroid_checksum="xyz789",  # Different checksum
    )

    report = compute_drift(old_sig, new_sig)

    assert report.status == "quarantined"
    assert report.needs_review
    assert report.overall_drift >= 0.5


def test_save_and_load_signatures(sample_moment2vec, sample_catalog):
    """Signatures can be saved and loaded."""
    sigs = [
        compute_moment_signature(sample_moment2vec, 0, sample_catalog, "Moment 0"),
        compute_moment_signature(sample_moment2vec, 1, sample_catalog, "Moment 1"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "signatures.json")
        save_signatures(sigs, path)

        loaded = load_signatures(path)

        assert len(loaded) == 2
        assert loaded[0].moment_id == 0
        assert loaded[0].label == "Moment 0"
        assert loaded[1].moment_id == 1


def test_load_signatures_missing_file():
    """Missing file returns empty list."""
    loaded = load_signatures("/nonexistent/path.json")
    assert loaded == []


def test_check_drift_on_rebuild_new_moments(sample_moment2vec, sample_catalog):
    """First build with no previous signatures marks all as new."""
    new_sigs, reports = check_drift_on_rebuild(
        sample_moment2vec, sample_catalog, labels=["M0", "M1", "M2", "M3"]
    )

    assert len(new_sigs) == 4
    assert len(reports) == 4
    assert all(r.status == "new" for r in reports)


def test_check_drift_on_rebuild_with_previous(sample_moment2vec, sample_catalog):
    """Rebuild with previous signatures computes drift."""
    # First build
    new_sigs, _ = check_drift_on_rebuild(
        sample_moment2vec, sample_catalog, labels=["M0", "M1", "M2", "M3"]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sigs.json")
        save_signatures(new_sigs, path)

        # Rebuild with same data - should be stable
        _, reports = check_drift_on_rebuild(
            sample_moment2vec, sample_catalog,
            labels=["M0", "M1", "M2", "M3"],
            previous_signatures_path=path
        )

        assert len(reports) == 4
        # All should be stable since data is identical
        assert all(r.status == "stable" for r in reports)


def test_get_quarantined_moments():
    """Extracts quarantined moment IDs correctly."""
    reports = [
        DriftReport(moment_id=0, old_label="A", new_label="A", status="stable"),
        DriftReport(moment_id=1, old_label="B", new_label="B", status="quarantined"),
        DriftReport(moment_id=2, old_label="C", new_label="C", status="drifted"),
        DriftReport(moment_id=3, old_label="D", new_label="D", status="quarantined"),
    ]

    quarantined = get_quarantined_moments(reports)
    assert quarantined == [1, 3]


def test_get_drift_summary():
    """Summary aggregates drift status correctly."""
    reports = [
        DriftReport(moment_id=0, old_label="A", new_label="A", status="stable", overall_drift=0.1),
        DriftReport(moment_id=1, old_label="B", new_label="B", status="drifted", needs_review=True, overall_drift=0.3),
        DriftReport(moment_id=2, old_label=None, new_label="C", status="new", overall_drift=0.0),
        DriftReport(moment_id=3, old_label="D", new_label="D", status="quarantined", needs_review=True, overall_drift=0.6),
    ]

    summary = get_drift_summary(reports)

    assert summary["total_moments"] == 4
    assert summary["stable"] == 1
    assert summary["drifted"] == 1
    assert summary["quarantined"] == 1
    assert summary["new"] == 1
    assert summary["needs_review"] == [1, 3]
    assert summary["max_drift"] == 0.6
    assert summary["healthy"] is False  # Has quarantined/drifted


def test_get_drift_summary_healthy():
    """Healthy summary when all stable."""
    reports = [
        DriftReport(moment_id=0, old_label="A", new_label="A", status="stable", overall_drift=0.05),
        DriftReport(moment_id=1, old_label="B", new_label="B", status="stable", overall_drift=0.1),
    ]

    summary = get_drift_summary(reports)

    assert summary["healthy"] is True
    assert summary["quarantined"] == 0
    assert summary["drifted"] == 0
