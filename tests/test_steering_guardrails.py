"""Tests for risk-managed steering guardrails."""
import pytest
from datetime import datetime, timedelta

from mosaic.steering_guardrails import (
    PolicyMetrics,
    GuardrailConfig,
    compute_confidence_factor,
    evaluate_stop_loss,
    get_effective_boost,
    init_policy_metrics,
    update_policy_metrics,
    reset_policy_metrics,
    get_policy_metrics,
)


def test_confidence_factor_ramps_up():
    """Confidence ramps from 0 to 1 based on impressions."""
    config = GuardrailConfig(min_impressions_for_full_effect=100)

    metrics = PolicyMetrics(policy_id="test", shop="test.shop", moment_id=0)
    metrics.impressions = 0
    assert compute_confidence_factor(metrics, config) == 0.0

    metrics.impressions = 50
    assert compute_confidence_factor(metrics, config) == 0.5

    metrics.impressions = 100
    assert compute_confidence_factor(metrics, config) == 1.0

    metrics.impressions = 200
    assert compute_confidence_factor(metrics, config) == 1.0


def test_stop_loss_triggers_on_conversion_drop():
    """Policy stops when conversion drops significantly."""
    config = GuardrailConfig(
        min_clicks_for_evaluation=10,
        warmup_hours=0,
        conversion_drop_threshold=0.30,
    )

    metrics = PolicyMetrics(
        policy_id="test",
        shop="test.shop",
        moment_id=0,
        baseline_conversion=0.10,  # 10% baseline
        current_conversion=0.05,   # Dropped to 5% (50% drop)
        clicks=20,
        window_clicks=20,
        created_at=datetime.utcnow() - timedelta(hours=1),
    )

    should_stop, reason, throttle = evaluate_stop_loss(metrics, config)
    assert should_stop is True
    assert "Conversion dropped" in reason
    assert throttle == 0.0


def test_stop_loss_throttles_on_ctr_drop():
    """Policy throttles when CTR drops moderately."""
    config = GuardrailConfig(
        min_clicks_for_evaluation=10,
        warmup_hours=0,
        ctr_drop_threshold=0.20,
        throttle_step=0.25,
    )

    metrics = PolicyMetrics(
        policy_id="test",
        shop="test.shop",
        moment_id=0,
        baseline_ctr=0.10,
        current_ctr=0.075,  # 25% drop
        clicks=20,
        window_clicks=20,
        throttle_factor=1.0,
        created_at=datetime.utcnow() - timedelta(hours=1),
    )

    should_stop, reason, throttle = evaluate_stop_loss(metrics, config)
    assert should_stop is False
    assert throttle == 0.75  # Reduced by throttle_step


def test_stop_loss_skips_during_warmup():
    """No evaluation during warmup period."""
    config = GuardrailConfig(warmup_hours=6)

    metrics = PolicyMetrics(
        policy_id="test",
        shop="test.shop",
        moment_id=0,
        baseline_ctr=0.10,
        current_ctr=0.01,  # Huge drop
        clicks=100,
        window_clicks=100,
        created_at=datetime.utcnow(),  # Just created = in warmup
    )

    should_stop, reason, throttle = evaluate_stop_loss(metrics, config)
    assert should_stop is False
    assert reason is None


def test_effective_boost_with_no_metrics():
    """Untracked policy returns 0 boost."""
    reset_policy_metrics("test.shop", 0)

    effective, info = get_effective_boost("test.shop", 0, requested_boost=1.5)

    assert effective == 0.0
    assert info["status"] == "untracked"


def test_effective_boost_with_stopped_policy():
    """Stopped policy returns 0 boost."""
    reset_policy_metrics("test.shop", 1)
    metrics = init_policy_metrics("test.shop", 1)
    metrics.is_stopped = True
    metrics.stop_reason = "Test stop"

    effective, info = get_effective_boost("test.shop", 1, requested_boost=2.0)

    assert effective == 0.0
    assert info["status"] == "stopped"


def test_update_policy_metrics():
    """Metrics update correctly with new events."""
    reset_policy_metrics("test.shop", 2)
    init_policy_metrics("test.shop", 2)

    update_policy_metrics("test.shop", 2, impressions=100, clicks=10)
    metrics = get_policy_metrics("test.shop", 2)

    assert metrics.impressions == 100
    assert metrics.clicks == 10
    assert metrics.current_ctr == 0.1

    update_policy_metrics("test.shop", 2, impressions=100, clicks=5, add_to_carts=3)
    metrics = get_policy_metrics("test.shop", 2)

    assert metrics.impressions == 200
    assert metrics.clicks == 15
    assert metrics.add_to_carts == 3
