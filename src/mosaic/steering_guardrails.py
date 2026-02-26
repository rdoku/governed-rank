"""Risk-managed steering: auto-throttle policies that hurt conversion.

Finance-inspired guardrails for merchant steering policies:
- Stop-loss: auto-reduce boost if conversion drops below threshold
- Confidence bands: require minimum event volume before full effect
- Rollback: automatically revert to baseline if metrics degrade

Key insight: "Steering with autopilot safety" - merchants can experiment
without fear of hurting sales.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PolicyMetrics:
    """Tracked metrics for a steering policy."""
    policy_id: str
    shop: str
    moment_id: int

    # Event counts
    impressions: int = 0
    clicks: int = 0
    add_to_carts: int = 0
    purchases: int = 0
    window_impressions: int = 0
    window_clicks: int = 0
    window_add_to_carts: int = 0
    window_purchases: int = 0

    # Recent events for windowed evaluation
    event_log: List[Tuple[datetime, int, int, int, int]] = field(default_factory=list)

    # Baseline (pre-policy) metrics
    baseline_ctr: Optional[float] = None
    baseline_atc_rate: Optional[float] = None
    baseline_conversion: Optional[float] = None

    # Current rates
    current_ctr: float = 0.0
    current_atc_rate: float = 0.0
    current_conversion: float = 0.0

    # Guardrail state
    throttle_factor: float = 1.0  # 1.0 = full effect, 0.0 = disabled
    is_stopped: bool = False
    stop_reason: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GuardrailConfig:
    """Configuration for steering guardrails."""

    # Minimum events before policy takes full effect (confidence band)
    min_impressions_for_full_effect: int = 100
    min_clicks_for_evaluation: int = 20

    # Stop-loss thresholds (relative to baseline)
    ctr_drop_threshold: float = 0.20  # 20% drop triggers throttle
    atc_drop_threshold: float = 0.25  # 25% drop triggers throttle
    conversion_drop_threshold: float = 0.30  # 30% drop = stop

    # Throttle behavior
    throttle_step: float = 0.25  # Reduce by 25% each time
    min_throttle: float = 0.25  # Don't go below 25% effect
    auto_recover: bool = True  # Allow recovery if metrics improve

    # Time windows
    evaluation_window_hours: int = 24
    warmup_hours: int = 6  # Don't evaluate during warmup
    paused: bool = False  # Global kill switch


# In-memory metrics store (would be DB in production)
_policy_metrics: Dict[str, PolicyMetrics] = {}


def _policy_key(shop: str, moment_id: int, policy_id: Optional[str]) -> str:
    if policy_id:
        return f"{shop}:{policy_id}"
    return f"{shop}:{moment_id}"


def get_policy_metrics(
    shop: str,
    moment_id: int,
    policy_id: Optional[str] = None,
) -> Optional[PolicyMetrics]:
    """Get metrics for a policy."""
    key = _policy_key(shop, moment_id, policy_id)
    return _policy_metrics.get(key)


def init_policy_metrics(
    shop: str,
    moment_id: int,
    policy_id: Optional[str] = None,
    baseline_ctr: Optional[float] = None,
    baseline_atc_rate: Optional[float] = None,
    baseline_conversion: Optional[float] = None,
) -> PolicyMetrics:
    """Initialize metrics tracking for a new policy."""
    key = _policy_key(shop, moment_id, policy_id)
    metrics = PolicyMetrics(
        policy_id=policy_id or key,
        shop=shop,
        moment_id=moment_id,
        baseline_ctr=baseline_ctr,
        baseline_atc_rate=baseline_atc_rate,
        baseline_conversion=baseline_conversion,
    )
    _policy_metrics[key] = metrics
    return metrics


def update_policy_metrics(
    shop: str,
    moment_id: int,
    policy_id: Optional[str] = None,
    impressions: int = 0,
    clicks: int = 0,
    add_to_carts: int = 0,
    purchases: int = 0,
    timestamp: Optional[datetime] = None,
    config: Optional[GuardrailConfig] = None,
) -> Optional[PolicyMetrics]:
    """Update metrics for a policy with new event counts."""
    key = _policy_key(shop, moment_id, policy_id)
    metrics = _policy_metrics.get(key)
    if not metrics:
        return None

    now = timestamp or datetime.utcnow()
    config = config or GuardrailConfig()
    metrics.event_log.append((now, impressions, clicks, add_to_carts, purchases))

    metrics.impressions += impressions
    metrics.clicks += clicks
    metrics.add_to_carts += add_to_carts
    metrics.purchases += purchases

    _update_windowed_metrics(metrics, config, now)

    metrics.last_updated = datetime.utcnow()
    return metrics


def compute_confidence_factor(
    metrics: PolicyMetrics,
    config: GuardrailConfig,
) -> float:
    """Compute confidence factor based on event volume.

    Returns value in [0, 1] where:
    - 0 = not enough data, don't apply policy
    - 1 = enough data, full policy effect

    This implements "confidence bands" - gradual ramp-up.
    """
    if metrics.impressions < config.min_impressions_for_full_effect:
        # Linear ramp-up
        return metrics.impressions / config.min_impressions_for_full_effect
    return 1.0


def _update_windowed_metrics(
    metrics: PolicyMetrics,
    config: GuardrailConfig,
    now: datetime,
) -> None:
    """Update windowed metrics based on recent event log."""
    window_start = now - timedelta(hours=config.evaluation_window_hours)
    if metrics.event_log:
        metrics.event_log = [
            entry for entry in metrics.event_log if entry[0] >= window_start
        ]

    window_impressions = 0
    window_clicks = 0
    window_atc = 0
    window_purchases = 0

    for _, impressions, clicks, add_to_carts, purchases in metrics.event_log:
        window_impressions += impressions
        window_clicks += clicks
        window_atc += add_to_carts
        window_purchases += purchases

    metrics.window_impressions = window_impressions
    metrics.window_clicks = window_clicks
    metrics.window_add_to_carts = window_atc
    metrics.window_purchases = window_purchases

    # Windowed rates
    if window_impressions > 0:
        metrics.current_ctr = window_clicks / window_impressions
        metrics.current_conversion = window_purchases / window_impressions
    else:
        metrics.current_ctr = 0.0
        metrics.current_conversion = 0.0

    if window_clicks > 0:
        metrics.current_atc_rate = window_atc / window_clicks
    else:
        metrics.current_atc_rate = 0.0


def evaluate_stop_loss(
    metrics: PolicyMetrics,
    config: GuardrailConfig,
) -> Tuple[bool, Optional[str], float]:
    """Evaluate if policy should be throttled or stopped.

    Returns:
        (should_stop, reason, new_throttle_factor)
    """
    # Not enough data to evaluate
    if metrics.window_clicks < config.min_clicks_for_evaluation:
        return False, None, metrics.throttle_factor

    # Check if in warmup period
    age = datetime.utcnow() - metrics.created_at
    if age < timedelta(hours=config.warmup_hours):
        return False, None, metrics.throttle_factor

    # No baseline = can't evaluate
    if (metrics.baseline_ctr is None and
        metrics.baseline_atc_rate is None and
        metrics.baseline_conversion is None):
        return False, None, metrics.throttle_factor

    # Calculate drops from baseline
    ctr_drop = None
    atc_drop = None
    conv_drop = None

    if metrics.baseline_ctr and metrics.baseline_ctr > 0:
        ctr_drop = (metrics.baseline_ctr - metrics.current_ctr) / metrics.baseline_ctr

    if metrics.baseline_atc_rate and metrics.baseline_atc_rate > 0:
        atc_drop = (metrics.baseline_atc_rate - metrics.current_atc_rate) / metrics.baseline_atc_rate

    if metrics.baseline_conversion and metrics.baseline_conversion > 0:
        conv_drop = (metrics.baseline_conversion - metrics.current_conversion) / metrics.baseline_conversion

    # Check stop conditions
    if conv_drop is not None and conv_drop >= config.conversion_drop_threshold:
        return True, (
            f"Conversion dropped {conv_drop*100:.1f}% "
            f"(threshold: {config.conversion_drop_threshold*100:.0f}%)"
        ), 0.0

    # Check throttle conditions
    new_throttle = metrics.throttle_factor

    throttle_candidates = []
    if ctr_drop is not None and ctr_drop >= config.ctr_drop_threshold:
        throttle_candidates.append(("CTR", ctr_drop, config.ctr_drop_threshold))
    if atc_drop is not None and atc_drop >= config.atc_drop_threshold:
        throttle_candidates.append(("ATC rate", atc_drop, config.atc_drop_threshold))

    if throttle_candidates:
        throttle_candidates.sort(key=lambda x: x[1], reverse=True)
        label, drop, _ = throttle_candidates[0]
        new_throttle = max(metrics.throttle_factor - config.throttle_step, config.min_throttle)
        reason = f"{label} dropped {drop*100:.1f}%"
        if new_throttle < metrics.throttle_factor:
            logger.warning(f"Policy {metrics.policy_id} throttled: {reason}")
        return False, reason, new_throttle

    # Check for recovery (if auto_recover enabled)
    if config.auto_recover and metrics.throttle_factor < 1.0:
        # Metrics improved - gradually restore
        ctr_ok = ctr_drop is None or ctr_drop < config.ctr_drop_threshold * 0.5
        atc_ok = atc_drop is None or atc_drop < config.atc_drop_threshold * 0.5
        if ctr_ok and atc_ok:
            new_throttle = min(metrics.throttle_factor + config.throttle_step, 1.0)
            if new_throttle > metrics.throttle_factor:
                logger.info(f"Policy {metrics.policy_id} recovering: throttle {metrics.throttle_factor:.2f} -> {new_throttle:.2f}")

    return False, None, new_throttle


def get_effective_boost(
    shop: str,
    moment_id: int,
    requested_boost: float,
    policy_id: Optional[str] = None,
    config: Optional[GuardrailConfig] = None,
) -> Tuple[float, dict]:
    """Get the effective boost after applying guardrails.

    Args:
        shop: Shop domain
        moment_id: Moment being boosted
        requested_boost: The boost factor the merchant requested
        config: Guardrail configuration

    Returns:
        (effective_boost, guardrail_info)
    """
    config = config or GuardrailConfig()
    if config.paused:
        return 0.0, {
            "status": "paused",
            "confidence": 0.0,
            "throttle": 0.0,
            "message": "Guardrails paused",
        }

    metrics = get_policy_metrics(shop, moment_id, policy_id)

    if not metrics:
        return 0.0, {
            "status": "untracked",
            "confidence": 0.0,
            "throttle": 1.0,
            "message": "No metrics yet (policy not initialized)",
        }

    # Check if stopped
    if metrics.is_stopped:
        return 0.0, {
            "status": "stopped",
            "confidence": 1.0,
            "throttle": 0.0,
            "message": metrics.stop_reason or "Policy stopped due to performance",
        }

    # Compute confidence factor (ramp-up)
    confidence = compute_confidence_factor(metrics, config)

    # Evaluate stop-loss
    should_stop, reason, new_throttle = evaluate_stop_loss(metrics, config)

    if should_stop:
        metrics.is_stopped = True
        metrics.stop_reason = reason
        metrics.throttle_factor = 0.0
        return 0.0, {
            "status": "stopped",
            "confidence": confidence,
            "throttle": 0.0,
            "message": reason,
        }

    # Update throttle
    metrics.throttle_factor = new_throttle

    # Effective boost = requested * confidence * throttle
    effective = requested_boost * confidence * new_throttle

    status = "active"
    if new_throttle < 1.0:
        status = "throttled"
    elif confidence < 1.0:
        status = "ramping"

    return effective, {
        "status": status,
        "confidence": round(confidence, 3),
        "throttle": round(new_throttle, 3),
        "impressions": metrics.window_impressions,
        "clicks": metrics.window_clicks,
        "current_ctr": round(metrics.current_ctr, 4) if metrics.window_impressions else None,
        "current_atc_rate": round(metrics.current_atc_rate, 4) if metrics.window_clicks else None,
        "current_conversion": round(metrics.current_conversion, 4) if metrics.window_impressions else None,
        "baseline_ctr": round(metrics.baseline_ctr, 4) if metrics.baseline_ctr else None,
        "baseline_atc_rate": round(metrics.baseline_atc_rate, 4) if metrics.baseline_atc_rate else None,
        "baseline_conversion": round(metrics.baseline_conversion, 4) if metrics.baseline_conversion else None,
        "window_hours": config.evaluation_window_hours,
        "message": reason if reason else f"Policy at {confidence*new_throttle*100:.0f}% effect",
    }


def reset_policy_metrics(
    shop: str,
    moment_id: int,
    policy_id: Optional[str] = None,
) -> None:
    """Reset metrics for a policy (e.g., when policy is updated)."""
    key = _policy_key(shop, moment_id, policy_id)
    if key in _policy_metrics:
        del _policy_metrics[key]


def get_all_policy_statuses(shop: str) -> List[dict]:
    """Get status of all policies for a shop."""
    statuses = []
    for key, metrics in _policy_metrics.items():
        if metrics.shop == shop:
            statuses.append({
                "moment_id": metrics.moment_id,
                "policy_id": metrics.policy_id,
                "impressions": metrics.impressions,
                "window_impressions": metrics.window_impressions,
                "clicks": metrics.clicks,
                "window_clicks": metrics.window_clicks,
                "throttle_factor": metrics.throttle_factor,
                "is_stopped": metrics.is_stopped,
                "stop_reason": metrics.stop_reason,
                "current_ctr": metrics.current_ctr,
                "baseline_ctr": metrics.baseline_ctr,
                "age_hours": (datetime.utcnow() - metrics.created_at).total_seconds() / 3600,
            })
    return statuses
