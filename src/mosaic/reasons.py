"""Reason registry for explainable recommendations.

This module provides a clean, extensible system for generating customer-facing
"why" labels for recommendations. It separates:
1. Signal detection (what context triggered this recommendation)
2. Label rendering (what text the customer sees)

This separation enables:
- A/B testing label phrasing without touching logic
- Per-merchant label overrides
- Privacy tiers (safe vs personal labels)
- Stable analytics keys even when labels change

Version 2.0: Now moment-powered! Reasons can reflect the active shopping moment
(e.g., "Perfect for Sunday brunch", "Quick grab for your routine").
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Optional, Set, Tuple

# Signal = a detected trigger from mode/placement/context/attribution/moment
Signal = str


@dataclass(frozen=True)
class ReasonRule:
    """A rule mapping signals to customer-facing labels.

    Attributes:
        key: Stable ID for analytics + A/B tests (never changes)
        priority: Lower = higher priority (checked first)
        label: Customer-facing text (default phrasing)
        when: Condition function that checks if this rule applies
        tier: "safe" (always shown) or "personal" (requires opt-in)
    """
    key: str
    priority: int
    label: str
    when: Callable[[Set[Signal]], bool]
    tier: str = "safe"


def has(sig: Signal) -> Callable[[Set[Signal]], bool]:
    """Helper: returns a condition that checks if a signal is present."""
    return lambda signals: sig in signals


def has_any(*sigs: Signal) -> Callable[[Set[Signal]], bool]:
    """Helper: returns a condition that checks if any of the signals are present."""
    return lambda signals: any(s in signals for s in sigs)


def has_all(*sigs: Signal) -> Callable[[Set[Signal]], bool]:
    """Helper: returns a condition that checks if all signals are present."""
    return lambda signals: all(s in signals for s in sigs)


def has_prefix(prefix: str) -> Callable[[Set[Signal]], bool]:
    """Helper: returns a condition that checks if any signal starts with prefix."""
    return lambda signals: any(s.startswith(prefix) for s in signals)


# -----------------------------------------------------------------------------
# REASON REGISTRY: One source of truth for priorities + labels
# -----------------------------------------------------------------------------
# Priority guide:
#   5-9:   Moment-specific with strong confidence (highest priority)
#   10-19: Mode/holiday overrides
#   20-24: Moment-specific (normal confidence)
#   25-29: Behavioral mode (cart_building, quick_grab, etc.)
#   30-39: Cart context
#   40-49: Time-based trends
#   50-59: Personal/browsing-based
#   10000: Default fallback
# -----------------------------------------------------------------------------

REASON_RULES: List[ReasonRule] = [
    # ==========================================================================
    # MOMENT-POWERED REASONS (highest priority when confident)
    # These fire when a specific moment is dominant AND confident
    # ==========================================================================

    # Strong moment signals (confidence > 0.4)
    ReasonRule(
        key="moment_sunday_brunch",
        priority=5,
        label="Perfect for Sunday brunch",
        when=has_all("moment:sunday_brunch", "moment:high_confidence"),
        tier="safe",
    ),
    ReasonRule(
        key="moment_family_dinner",
        priority=5,
        label="Great for family dinner",
        when=has_all("moment:family_dinner", "moment:high_confidence"),
        tier="safe",
    ),
    ReasonRule(
        key="moment_weeknight_restock",
        priority=5,
        label="Weeknight essential",
        when=has_all("moment:weeknight_restock", "moment:high_confidence"),
        tier="safe",
    ),
    ReasonRule(
        key="moment_party",
        priority=5,
        label="Party favorite",
        when=lambda signals: (
            "moment:high_confidence" in signals
            and ("moment:party_&_treats" in signals or "moment:party" in signals)
        ),
        tier="safe",
    ),
    ReasonRule(
        key="moment_breakfast",
        priority=5,
        label="Breakfast pick",
        when=has_all("moment:breakfast", "moment:high_confidence"),
        tier="safe",
    ),
    ReasonRule(
        key="moment_healthy",
        priority=5,
        label="Healthy choice",
        when=has_all("moment:healthy_picks", "moment:high_confidence"),
        tier="safe",
    ),
    ReasonRule(
        key="moment_late_night",
        priority=5,
        label="Late night favorite",
        when=has_all("moment:late-night_browse", "moment:high_confidence"),
        tier="safe",
    ),
    ReasonRule(
        key="moment_fresh_produce",
        priority=5,
        label="Farm fresh pick",
        when=has_all("moment:fresh_produce", "moment:high_confidence"),
        tier="safe",
    ),

    # Mode/holiday overrides
    ReasonRule(
        key="promo_mode",
        priority=10,
        label="Seasonal pick",
        when=has_any("mode:promo", "context:is_holiday"),
        tier="safe",
    ),

    # ==========================================================================
    # MOMENT-POWERED REASONS (normal priority)
    # These fire when a moment is active but not necessarily dominant
    # ==========================================================================
    ReasonRule(
        key="moment_generic_labeled",
        priority=20,
        label="",  # Will be overridden by dynamic label
        when=lambda signals: (
            "moment:has_label" in signals
            and ("moment:high_confidence" in signals or "moment:medium_confidence" in signals)
        ),
        tier="safe",
    ),

    # ==========================================================================
    # BEHAVIORAL MODE REASONS
    # Based on session behavior patterns (cart_building, stock_up, etc.)
    # ==========================================================================
    ReasonRule(
        key="behavior_cart_building",
        priority=25,
        label="Add to your haul",
        when=has("behavior:cart_building"),
        tier="safe",
    ),
    ReasonRule(
        key="behavior_stock_up",
        priority=25,
        label="Time to stock up",
        when=has("behavior:stock_up"),
        tier="safe",
    ),
    ReasonRule(
        key="behavior_exploration",
        priority=25,
        label="Something new to try",
        when=has("behavior:exploration"),
        tier="safe",
    ),
    ReasonRule(
        key="behavior_quick_grab",
        priority=25,
        label="Quick add",
        when=has("behavior:quick_grab"),
        tier="safe",
    ),

    # ==========================================================================
    # PLACEMENT & CONTEXT REASONS (original rules)
    # ==========================================================================

    # Cart context (user has items in cart)
    ReasonRule(
        key="pairs_with_cart",
        priority=30,
        label="Pairs well with your cart",
        when=has("context:has_cart"),
        tier="safe",
    ),

    # Time-based trends
    ReasonRule(
        key="trending_evening",
        priority=40,
        label="Trending tonight",
        when=has_all("time:evening", "attribution:context_heavy"),
        tier="safe",
    ),
    ReasonRule(
        key="trending_morning",
        priority=41,
        label="Morning favorite",
        when=has_all("time:morning", "attribution:context_heavy"),
        tier="safe",
    ),
    ReasonRule(
        key="trending_now",
        priority=42,
        label="Trending now",
        when=has("attribution:context_heavy"),
        tier="safe",
    ),

    # Personal/browsing-based (requires opt-in)
    ReasonRule(
        key="based_on_browsing",
        priority=50,
        label="Inspired by what you viewed",
        when=has("attribution:history_heavy"),
        tier="personal",
    ),

    # Default fallback (always matches)
    ReasonRule(
        key="default",
        priority=10_000,
        label="Recommended for you",
        when=lambda _: True,
        tier="safe",
    ),
]


# Moment label to reason label mapping
# When a moment is active, we can generate a more specific reason
MOMENT_REASON_TEMPLATES: Dict[str, str] = {
    "sunday_brunch": "Perfect for Sunday brunch",
    "family_dinner": "Great for family dinner",
    "weeknight_restock": "Weeknight essential",
    "party_&_treats": "Party favorite",
    "party": "Party favorite",
    "breakfast": "Breakfast pick",
    "healthy_picks": "Healthy choice",
    "late-night_browse": "Late night favorite",
    "late_night_browse": "Late night favorite",
    "fresh_produce": "Farm fresh pick",
    "quick_grab": "Quick add",
    "stock_up": "Time to stock up",
    "exploration": "Something new to try",
    "cart_building": "Add to your haul",
}


def _normalize_moment_label(label: str) -> str:
    """Normalize moment label for matching (lowercase, underscores)."""
    return label.lower().replace(" ", "_").replace("-", "_").replace("&", "_")


def _soften_moment_label(label: str) -> str:
    """Generate softer copy for medium-confidence moments."""
    if not label:
        return "Moment pick"
    text = label.strip()
    lower = text.lower()
    if lower.endswith("pick"):
        return text
    return f"{text} pick"


def compute_moment_confidence(
    weights: Optional[List[float]],
    high_margin: float = 0.15,
    high_peakedness: float = 0.4,
    medium_margin: float = 0.08,
) -> Tuple[float, float, str]:
    """Compute margin/peakedness confidence tier from a distribution."""
    if not weights or len(weights) < 2:
        return 0.0, 0.0, "low"
    w = [float(x) for x in weights]
    w_sorted = sorted(w, reverse=True)
    margin = w_sorted[0] - w_sorted[1]
    entropy = 0.0
    for v in w:
        if v > 0:
            entropy -= v * math.log(v + 1e-9)
    peakedness = 1.0 - (entropy / math.log(len(w)))
    if margin >= high_margin and peakedness >= high_peakedness:
        return margin, peakedness, "high"
    if margin >= medium_margin:
        return margin, peakedness, "medium"
    return margin, peakedness, "low"


def pick_reason(
    signals: Set[Signal],
    allow_personal: bool = True,
    label_overrides: Optional[Dict[str, str]] = None,
    moment_label: Optional[str] = None,
) -> str:
    """Pick exactly one primary reason label.

    Args:
        signals: Set of detected signals from context
        allow_personal: If False, skip "personal" tier labels (privacy mode)
        label_overrides: Optional dict of key -> custom label for per-merchant overrides
        moment_label: Optional moment label for dynamic reason generation

    Returns:
        The customer-facing reason label
    """
    key, label = pick_reason_with_key(signals, allow_personal, label_overrides, moment_label)
    return label


def pick_reason_with_key(
    signals: Set[Signal],
    allow_personal: bool = True,
    label_overrides: Optional[Dict[str, str]] = None,
    moment_label: Optional[str] = None,
) -> Tuple[str, str]:
    """Pick reason and return both the key and label.

    Useful for analytics where you need the stable key.

    Args:
        signals: Set of detected signals from context
        allow_personal: If False, skip personal tier labels
        label_overrides: Optional dict of key -> custom label
        moment_label: Optional moment label for dynamic reason generation

    Returns:
        Tuple of (key, label)
    """
    label_overrides = label_overrides or {}

    for rule in sorted(REASON_RULES, key=lambda r: r.priority):
        if rule.tier == "personal" and not allow_personal:
            continue
        if rule.when(signals):
            # Special case: dynamic moment label
            if rule.key == "moment_generic_labeled" and moment_label:
                normalized = _normalize_moment_label(moment_label)
                # Try to find a template for this moment
                if "moment:high_confidence" in signals:
                    reason_text = MOMENT_REASON_TEMPLATES.get(
                        normalized,
                        f"Perfect for {moment_label.lower()}"  # Fallback template
                    )
                elif "moment:medium_confidence" in signals:
                    reason_text = _soften_moment_label(moment_label)
                else:
                    reason_text = "Recommended for you"
                return (f"moment_{normalized}", label_overrides.get(f"moment_{normalized}", reason_text))

            label = label_overrides.get(rule.key, rule.label)
            return (rule.key, label)

    return ("default", "Recommended for you")


# -----------------------------------------------------------------------------
# SIGNAL BUILDER: Converts raw context into signals
# -----------------------------------------------------------------------------

def build_signals(
    mode: str,
    placement: str,
    has_cart: bool,
    time_bucket: str,
    is_holiday: bool,
    history_score: float,
    context_score: float,
    # New moment-related parameters
    moment_weights: Optional[List[float]] = None,
    moment_labels: Optional[Dict[int, str]] = None,
    behavioral_mode: Optional[str] = None,
    activation_confidence_tier: Optional[str] = None,
    item_confidence_tier: Optional[str] = None,
) -> Set[Signal]:
    """Build signal set from recommendation context.

    This function detects which signals are active based on:
    - Mode setting (auto/routine/promo)
    - Widget placement (pdp/cart/homepage/collection)
    - User context (cart contents, time of day)
    - Score attribution (H vs C dominance)
    - Active moment and its confidence (NEW)
    - Behavioral mode from session patterns (NEW)

    Args:
        mode: Recommendation mode (auto, routine, promo)
        placement: Widget placement (pdp, cart, homepage, collection)
        has_cart: Whether user has items in cart
        time_bucket: Time of day (morning, afternoon, evening, unknown)
        is_holiday: Whether it's a holiday period
        history_score: H-score (history/WHO component)
        context_score: C-score (context/WHEN component)
        moment_weights: Optional list of moment probabilities from model
        moment_labels: Optional dict mapping moment index to label
        behavioral_mode: Optional detected behavioral mode (cart_building, stock_up, etc.)
        activation_confidence_tier: Optional confidence tier for activation (high/medium/low)
        item_confidence_tier: Optional confidence tier for item affinity (high/medium/low)

    Returns:
        Set of active signal strings
    """
    signals: Set[Signal] = set()

    # Mode signals
    signals.add(f"mode:{mode}")

    # Placement signals
    signals.add(f"placement:{placement}")

    # Context signals
    if has_cart:
        signals.add("context:has_cart")
    if is_holiday:
        signals.add("context:is_holiday")

    # Time signals
    if time_bucket in ("morning", "afternoon", "evening"):
        signals.add(f"time:{time_bucket}")

    # Attribution signals (which scoring component dominated)
    if context_score > history_score and context_score > 0:
        signals.add("attribution:context_heavy")
    elif history_score > 0:
        signals.add("attribution:history_heavy")

    # =========================================================================
    # NEW: Moment-powered signals
    # =========================================================================
    if moment_weights and len(moment_weights) > 0:
        # Find dominant moment
        max_weight = max(moment_weights)
        dominant_idx = moment_weights.index(max_weight)

        # Activation confidence tier
        if activation_confidence_tier is None:
            _, _, activation_confidence_tier = compute_moment_confidence(moment_weights)
        if activation_confidence_tier == "high":
            signals.add("moment:activation_high")
            signals.add("moment:confident")
        elif activation_confidence_tier == "medium":
            signals.add("moment:activation_medium")
            signals.add("moment:confident")

        # Item confidence tier
        if item_confidence_tier == "high":
            signals.add("moment:item_high")
        elif item_confidence_tier == "medium":
            signals.add("moment:item_medium")

        # Combined confidence
        if "moment:activation_high" in signals and "moment:item_high" in signals:
            signals.add("moment:high_confidence")
        elif (
            ("moment:activation_high" in signals or "moment:activation_medium" in signals)
            and ("moment:item_high" in signals or "moment:item_medium" in signals)
        ):
            signals.add("moment:medium_confidence")

        # If we have a label for this moment, add it as a signal
        if moment_labels and dominant_idx in moment_labels:
            label = moment_labels[dominant_idx]
            normalized = _normalize_moment_label(label)
            signals.add(f"moment:{normalized}")
            signals.add("moment:has_label")

    # =========================================================================
    # NEW: Behavioral mode signals
    # =========================================================================
    if behavioral_mode:
        normalized_mode = behavioral_mode.lower().replace(" ", "_")
        signals.add(f"behavior:{normalized_mode}")

    return signals


def get_dominant_moment_info(
    moment_weights: Optional[List[float]],
    moment_labels: Optional[Dict[int, str]],
) -> Tuple[Optional[int], Optional[str], float]:
    """Extract dominant moment index, label, and confidence.

    Args:
        moment_weights: List of moment probabilities
        moment_labels: Dict mapping moment index to label

    Returns:
        Tuple of (dominant_index, label, confidence)
    """
    if not moment_weights or len(moment_weights) == 0:
        return (None, None, 0.0)

    max_weight = max(moment_weights)
    dominant_idx = moment_weights.index(max_weight)

    label = None
    if moment_labels and dominant_idx in moment_labels:
        label = moment_labels[dominant_idx]

    return (dominant_idx, label, max_weight)
