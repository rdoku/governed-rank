"""Feature extraction for moment activation.

Extracts interpretable boolean features from cart + context for the evidence graph.
Features are non-time (time is handled separately by time_mult).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

import numpy as np


@dataclass
class FeatureExtractionResult:
    """Result of feature extraction."""

    fired_features: FrozenSet[str]
    feature_details: Dict[str, any]  # For debugging/receipts


# =============================================================================
# Cart Structure Features
# =============================================================================

def extract_cart_structure_features(cart_size: int) -> Set[str]:
    """Extract features based on cart structure.

    Args:
        cart_size: Number of items in cart

    Returns:
        Set of fired feature names
    """
    features = set()

    if cart_size == 0:
        features.add("empty_cart")
    elif cart_size == 1:
        features.add("single_item")
    elif cart_size <= 3:
        features.add("small_cart")
    elif cart_size <= 8:
        features.add("medium_cart")
    elif cart_size <= 15:
        features.add("large_cart")
    else:
        features.add("very_large_cart")
        features.add("stock_up_cart")

    return features


# =============================================================================
# Cart Content Features (Department/Aisle Presence)
# =============================================================================

# Common department keywords to feature names
DEPT_FEATURE_MAP = {
    "produce": "produce_present",
    "dairy": "dairy_present",
    "bakery": "bakery_present",
    "meat": "meat_present",
    "seafood": "seafood_present",
    "frozen": "frozen_present",
    "deli": "deli_present",
    "snacks": "snacks_present",
    "beverages": "beverages_present",
    "pantry": "pantry_present",
    "household": "household_present",
    "personal care": "personal_care_present",
    "baby": "baby_present",
    "pet": "pet_present",
    "alcohol": "alcohol_present",
    "breakfast": "breakfast_present",
    "canned": "canned_present",
    "international": "international_present",
}

# Aisle-level features for finer granularity
AISLE_FEATURE_MAP = {
    "chips": "chips_present",
    "salsa": "salsa_present",
    "beer": "beer_present",
    "wine": "wine_present",
    "coffee": "coffee_present",
    "tea": "tea_present",
    "cereal": "cereal_present",
    "bread": "bread_present",
    "eggs": "eggs_present",
    "milk": "milk_present",
    "yogurt": "yogurt_present",
    "cheese": "cheese_present",
    "ice cream": "ice_cream_present",
    "candy": "candy_present",
    "soda": "soda_present",
    "juice": "juice_present",
    "pasta": "pasta_present",
    "rice": "rice_present",
    "soup": "soup_present",
}


def extract_cart_content_features(
    cart_items: List[int],
    catalog: Optional[Dict[int, Dict]] = None,
) -> Set[str]:
    """Extract features based on cart contents (departments/aisles).

    Args:
        cart_items: List of item_ids in cart
        catalog: item_id -> {name, aisle, dept}

    Returns:
        Set of fired feature names
    """
    features = set()

    if not catalog or not cart_items:
        return features

    depts_present = set()
    aisles_present = set()

    for item_id in cart_items:
        meta = catalog.get(item_id)
        if not meta:
            continue

        dept = (meta.get("dept") or "").lower()
        aisle = (meta.get("aisle") or "").lower()
        name = (meta.get("name") or "").lower()

        # Check department
        for keyword, feature in DEPT_FEATURE_MAP.items():
            if keyword in dept:
                depts_present.add(keyword)
                features.add(feature)

        # Check aisle
        for keyword, feature in AISLE_FEATURE_MAP.items():
            if keyword in aisle or keyword in name:
                aisles_present.add(keyword)
                features.add(feature)

    # Track unique counts
    if len(depts_present) >= 5:
        features.add("diverse_departments")
    if len(depts_present) <= 2 and len(cart_items) >= 3:
        features.add("focused_shopping")

    return features


# =============================================================================
# Cart Combo Features
# =============================================================================

# Common combos that signal shopping missions
COMBO_RULES = [
    ({"chips_present", "salsa_present"}, "chips_and_salsa"),
    ({"chips_present", "beverages_present"}, "snacks_and_drinks"),
    ({"beer_present", "snacks_present"}, "party_snacks"),
    ({"wine_present", "cheese_present"}, "wine_and_cheese"),
    ({"coffee_present", "breakfast_present"}, "morning_routine"),
    ({"eggs_present", "dairy_present"}, "breakfast_prep"),
    ({"pasta_present", "meat_present"}, "dinner_prep"),
    ({"produce_present", "meat_present"}, "meal_prep"),
    ({"baby_present", "household_present"}, "family_stock_up"),
    ({"frozen_present", "snacks_present"}, "convenience_run"),
    ({"ice_cream_present"}, "treat_yourself"),
    ({"alcohol_present"}, "adult_beverages"),
]


def extract_combo_features(content_features: Set[str]) -> Set[str]:
    """Extract combo features from content features.

    Args:
        content_features: Set of content features already extracted

    Returns:
        Set of combo feature names
    """
    combos = set()

    for required, combo_name in COMBO_RULES:
        if required.issubset(content_features):
            combos.add(combo_name)

    return combos


# =============================================================================
# Context Features (Non-Time)
# =============================================================================

def extract_context_features(context: Dict) -> Set[str]:
    """Extract features from request context (non-time features).

    Args:
        context: Request context dict (device, channel, etc.)

    Returns:
        Set of context feature names
    """
    features = set()

    # Device type
    device = (context.get("device") or "").lower()
    if "mobile" in device or "phone" in device:
        features.add("mobile_device")
    elif "tablet" in device or "ipad" in device:
        features.add("tablet_device")
    elif "desktop" in device or "web" in device:
        features.add("desktop_device")

    # Channel
    channel = (context.get("channel") or "").lower()
    if "delivery" in channel:
        features.add("delivery_channel")
    elif "pickup" in channel:
        features.add("pickup_channel")
    elif "instore" in channel or "in-store" in channel:
        features.add("instore_channel")

    # App vs web
    platform = (context.get("platform") or "").lower()
    if "app" in platform or "ios" in platform or "android" in platform:
        features.add("app_platform")
    elif "web" in platform:
        features.add("web_platform")

    # New vs returning
    is_new = context.get("is_new_user", False)
    if is_new:
        features.add("new_user")
    else:
        features.add("returning_user")

    # Loyalty status
    loyalty = context.get("loyalty_tier", "").lower()
    if loyalty in ("gold", "premium", "plus", "vip"):
        features.add("premium_member")

    return features


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_features(
    cart_items: List[int],
    context: Optional[Dict] = None,
    catalog: Optional[Dict[int, Dict]] = None,
) -> FeatureExtractionResult:
    """Extract all features for moment activation.

    Combines:
    - Cart structure (size-based)
    - Cart content (dept/aisle presence)
    - Cart combos (multi-item signals)
    - Context (device, channel, etc.)

    Time features are NOT included here (handled by time_mult).

    Args:
        cart_items: List of item_ids in cart
        context: Request context dict
        catalog: item_id -> {name, aisle, dept}

    Returns:
        FeatureExtractionResult with fired features and details
    """
    all_features = set()
    details = {}

    # Cart structure
    structure_features = extract_cart_structure_features(len(cart_items))
    all_features.update(structure_features)
    details["structure"] = list(structure_features)

    # Cart content
    content_features = extract_cart_content_features(cart_items, catalog)
    all_features.update(content_features)
    details["content"] = list(content_features)

    # Combos
    combo_features = extract_combo_features(content_features)
    all_features.update(combo_features)
    details["combos"] = list(combo_features)

    # Context
    context_features = extract_context_features(context or {})
    all_features.update(context_features)
    details["context"] = list(context_features)

    details["total_count"] = len(all_features)

    return FeatureExtractionResult(
        fired_features=frozenset(all_features),
        feature_details=details,
    )


def get_feature_list() -> List[str]:
    """Get list of all possible features for documentation."""
    features = []

    # Structure
    features.extend([
        "empty_cart", "single_item", "small_cart", "medium_cart",
        "large_cart", "very_large_cart", "stock_up_cart",
    ])

    # Content
    features.extend(DEPT_FEATURE_MAP.values())
    features.extend(AISLE_FEATURE_MAP.values())
    features.extend(["diverse_departments", "focused_shopping"])

    # Combos
    features.extend([combo for _, combo in COMBO_RULES])

    # Context
    features.extend([
        "mobile_device", "tablet_device", "desktop_device",
        "delivery_channel", "pickup_channel", "instore_channel",
        "app_platform", "web_platform",
        "new_user", "returning_user", "premium_member",
    ])

    return sorted(set(features))
