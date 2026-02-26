"""Verify evaluation reports match their moment2vec artifacts."""
import json
from pathlib import Path

import numpy as np
import pytest

from mosaic.moment_nmf import compute_moment2vec_stats


def _load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_stats(path: Path, threshold: float) -> dict:
    m2v = np.load(path)
    return compute_moment2vec_stats(m2v, threshold=threshold)


def test_instacart_eval_report_matches_artifact():
    report_path = Path("artifacts/moment_eval_report.json")
    assert report_path.exists()

    report = _load_report(report_path)
    candidate_path = Path(report["candidate"]["path"])
    assert candidate_path.exists()

    stats = _compute_stats(candidate_path, report["threshold"])
    reported = report["candidate"]["stats"]

    assert stats["pct_above_threshold"] == pytest.approx(
        reported["pct_above_threshold"], rel=1e-4, abs=1e-4
    )
    assert stats["max_weight"]["mean"] == pytest.approx(
        reported["max_weight"]["mean"], rel=1e-4, abs=1e-4
    )
    assert stats["margin"]["mean"] == pytest.approx(
        reported["margin"]["mean"], rel=1e-4, abs=1e-4
    )


def test_tafeng_eval_report_matches_artifact():
    report_path = Path("artifacts/tafeng_eval_report.json")
    assert report_path.exists()

    report = _load_report(report_path)
    candidate_path = Path(report["candidate"]["path"])
    assert candidate_path.exists()

    stats = _compute_stats(candidate_path, report["threshold"])
    reported = report["candidate"]["stats"]

    assert stats["pct_above_threshold"] == pytest.approx(
        reported["pct_above_threshold"], rel=1e-4, abs=1e-4
    )
    assert stats["max_weight"]["mean"] == pytest.approx(
        reported["max_weight"]["mean"], rel=1e-4, abs=1e-4
    )
    assert stats["max_weight"]["std"] == pytest.approx(
        reported["max_weight"]["std"], rel=1e-4, abs=1e-4
    )


def test_hm_eval_report_matches_artifact():
    report_path = Path("artifacts/hm_eval_report.json")
    assert report_path.exists()

    report = _load_report(report_path)
    candidate_path = Path(report["candidate"]["path"])
    assert candidate_path.exists()

    stats = _compute_stats(candidate_path, report["threshold"])
    reported = report["candidate"]["stats"]

    assert stats["pct_above_threshold"] == pytest.approx(
        reported["pct_above_threshold"], rel=1e-4, abs=1e-4
    )
    assert stats["max_weight"]["mean"] == pytest.approx(
        reported["max_weight"]["mean"], rel=1e-4, abs=1e-4
    )
    assert stats["max_weight"]["std"] == pytest.approx(
        reported["max_weight"]["std"], rel=1e-4, abs=1e-4
    )
