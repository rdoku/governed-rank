"""Time signature table for moment activation.

Stores time_mult[bucket, m] = clip(P(m|bucket) / P(m), [min, max])

This captures: "in this time bucket, which moments are more/less likely than average?"
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TimeSignatureConfig:
    """Configuration for time signature computation."""

    # Hour bins (24-hour format, end-exclusive)
    hour_bins: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 6),    # Night
        (6, 9),    # Early morning
        (9, 12),   # Morning
        (12, 15),  # Early afternoon
        (15, 18),  # Late afternoon
        (18, 21),  # Evening
        (21, 24),  # Late evening
    ])

    # Days of week (0=Sunday, 6=Saturday)
    use_dow: bool = True

    # Clamp range for multipliers
    min_mult: float = 0.5
    max_mult: float = 2.0

    # Smoothing (Laplace)
    smoothing_alpha: float = 1.0

    # Minimum support for bucket influence
    min_orders_per_bucket: int = 50


def get_time_bucket(
    timestamp: datetime,
    config: Optional[TimeSignatureConfig] = None,
) -> Tuple[int, int]:
    """Get (dow, hour_bin) bucket for a timestamp.

    Args:
        timestamp: Datetime object
        config: Time signature config

    Returns:
        (dow, hour_bin_idx) tuple
    """
    config = config or TimeSignatureConfig()

    if config.use_dow:
        dow = (timestamp.weekday() + 1) % 7  # Convert Mon=0 to Sun=0
    else:
        dow = 0
    hour = timestamp.hour

    hour_bin = 0
    for i, (start, end) in enumerate(config.hour_bins):
        if start <= hour < end:
            hour_bin = i
            break

    return dow, hour_bin


def bucket_to_idx(dow: int, hour_bin: int, n_hour_bins: int) -> int:
    """Convert (dow, hour_bin) to flat index."""
    return dow * n_hour_bins + hour_bin


def idx_to_bucket(idx: int, n_hour_bins: int) -> Tuple[int, int]:
    """Convert flat index to (dow, hour_bin)."""
    dow = idx // n_hour_bins
    hour_bin = idx % n_hour_bins
    return dow, hour_bin


def compute_time_signatures(
    orders: List[Tuple[datetime, List[int]]],
    moment2vec: np.ndarray,
    population_prior: Optional[np.ndarray] = None,
    config: Optional[TimeSignatureConfig] = None,
) -> np.ndarray:
    """Compute time signature table from historical orders.

    Algorithm:
        For each time bucket:
            P(m|bucket) = mean of order profiles in that bucket
        time_mult[bucket, m] = clip(P(m|bucket) / P(m), [min, max])

    Args:
        orders: List of (timestamp, item_ids) tuples
        moment2vec: (N_items, K) affinity matrix
        population_prior: (K,) base prior P(m), or computed from orders
        config: Time signature config

    Returns:
        (n_buckets, K) time multiplier matrix
    """
    config = config or TimeSignatureConfig()
    K = moment2vec.shape[1]
    N = moment2vec.shape[0]
    n_hour_bins = len(config.hour_bins)
    n_dows = 7 if config.use_dow else 1
    n_buckets = n_dows * n_hour_bins

    # Initialize counts with smoothing
    bucket_counts = np.full(n_buckets, config.smoothing_alpha, dtype=np.float32)
    bucket_moment_sums = np.full((n_buckets, K), config.smoothing_alpha / K, dtype=np.float32)

    # Compute P(m) if not provided
    if population_prior is None:
        all_profiles = []
        for _, items in orders:
            valid = [i for i in items if 0 <= i < N]
            if len(valid) >= 2:
                profile = np.mean(moment2vec[valid], axis=0)
                all_profiles.append(profile)
        if all_profiles:
            population_prior = np.mean(np.stack(all_profiles), axis=0)
            population_prior /= population_prior.sum() + 1e-9
        else:
            population_prior = np.ones(K, dtype=np.float32) / K

    # Accumulate order profiles by bucket
    for timestamp, items in orders:
        valid = [i for i in items if 0 <= i < N]
        if len(valid) < 2:
            continue

        order_profile = np.mean(moment2vec[valid], axis=0)
        order_profile /= order_profile.sum() + 1e-9

        dow, hour_bin = get_time_bucket(timestamp, config)
        if not config.use_dow:
            dow = 0
        idx = bucket_to_idx(dow, hour_bin, n_hour_bins)

        bucket_counts[idx] += 1
        bucket_moment_sums[idx] += order_profile

    # Compute P(m|bucket) and time multipliers
    time_mult = np.zeros((n_buckets, K), dtype=np.float32)
    for bucket_idx in range(n_buckets):
        p_m_given_bucket = bucket_moment_sums[bucket_idx] / bucket_counts[bucket_idx]
        # Blend toward population prior if support is low
        bucket_support = max(0.0, bucket_counts[bucket_idx] - config.smoothing_alpha)
        if bucket_support < config.min_orders_per_bucket:
            mix = bucket_support / max(config.min_orders_per_bucket, 1e-9)
            p_m_given_bucket = mix * p_m_given_bucket + (1.0 - mix) * population_prior

        # Normalize
        p_m_given_bucket /= p_m_given_bucket.sum() + 1e-9

        # Compute multiplier: P(m|bucket) / P(m)
        mult = p_m_given_bucket / (population_prior + 1e-9)

        # Clamp to [min, max]
        mult = np.clip(mult, config.min_mult, config.max_mult)

        time_mult[bucket_idx] = mult

    return time_mult


def get_time_multiplier(
    timestamp: datetime,
    time_mult: np.ndarray,
    config: Optional[TimeSignatureConfig] = None,
) -> np.ndarray:
    """Get time multiplier vector for a timestamp.

    Args:
        timestamp: Current time
        time_mult: (n_buckets, K) time multiplier matrix
        config: Time signature config

    Returns:
        (K,) multiplier vector
    """
    config = config or TimeSignatureConfig()
    n_hour_bins = len(config.hour_bins)

    dow, hour_bin = get_time_bucket(timestamp, config)
    idx = bucket_to_idx(dow, hour_bin, n_hour_bins)

    if idx < len(time_mult):
        return time_mult[idx]
    else:
        return np.ones(time_mult.shape[1], dtype=np.float32)


def save_time_signatures(
    time_mult: np.ndarray,
    artifact_dir: str,
    moment_space_id: str,
    config: Optional[TimeSignatureConfig] = None,
) -> str:
    """Save time signatures to disk.

    Args:
        time_mult: (n_buckets, K) time multiplier matrix
        artifact_dir: Directory to save to
        moment_space_id: Version identifier
        config: Config to save alongside

    Returns:
        Path to saved file
    """
    os.makedirs(artifact_dir, exist_ok=True)

    # Save numpy array
    npy_path = os.path.join(artifact_dir, f"time_mult_{moment_space_id}.npy")
    np.save(npy_path, time_mult)

    # Save config as JSON
    if config:
        config_path = os.path.join(artifact_dir, f"time_mult_config_{moment_space_id}.json")
        config_dict = {
            "hour_bins": config.hour_bins,
            "use_dow": config.use_dow,
            "min_mult": config.min_mult,
            "max_mult": config.max_mult,
            "min_orders_per_bucket": config.min_orders_per_bucket,
            "n_buckets": time_mult.shape[0],
            "K": time_mult.shape[1],
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    return npy_path


def load_time_signatures(
    artifact_dir: str,
    moment_space_id: str,
    K: int,
) -> Tuple[Optional[np.ndarray], Optional[TimeSignatureConfig]]:
    """Load time signatures from disk.

    Args:
        artifact_dir: Directory to load from
        moment_space_id: Version identifier
        K: Number of moments (for validation)

    Returns:
        (time_mult, config) or (None, None) if not found
    """
    npy_path = os.path.join(artifact_dir, f"time_mult_{moment_space_id}.npy")
    config_path = os.path.join(artifact_dir, f"time_mult_config_{moment_space_id}.json")

    if not os.path.exists(npy_path):
        return None, None

    try:
        time_mult = np.load(npy_path).astype(np.float32)
        if time_mult.shape[1] != K:
            return None, None
    except Exception:
        return None, None

    config = None
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = TimeSignatureConfig(
                hour_bins=[tuple(x) for x in config_dict.get("hour_bins", [])],
                use_dow=config_dict.get("use_dow", True),
                min_mult=config_dict.get("min_mult", 0.5),
                max_mult=config_dict.get("max_mult", 2.0),
                min_orders_per_bucket=config_dict.get("min_orders_per_bucket", 50),
            )
        except Exception:
            config = TimeSignatureConfig()

    return time_mult, config


# =============================================================================
# Online Updates (EMA)
# =============================================================================

def update_time_signatures_ema(
    time_mult: np.ndarray,
    timestamp: datetime,
    purchased_profile: np.ndarray,
    population_prior: np.ndarray,
    config: Optional[TimeSignatureConfig] = None,
    alpha: float = 0.01,
) -> np.ndarray:
    """Update time signatures with EMA from a conversion event.

    On conversion:
        p_buy = normalize(mean(A[purchased_items]))
        target_mult = p_buy / P_m
        time_mult[bucket] = (1 - alpha) * time_mult[bucket] + alpha * target_mult

    Args:
        time_mult: (n_buckets, K) current time multipliers
        timestamp: Time of conversion
        purchased_profile: (K,) moment profile of purchased items
        population_prior: (K,) base prior P(m)
        config: Time signature config
        alpha: EMA learning rate (small for stability)

    Returns:
        Updated time_mult
    """
    config = config or TimeSignatureConfig()
    n_hour_bins = len(config.hour_bins)

    dow, hour_bin = get_time_bucket(timestamp, config)
    if not config.use_dow:
        dow = 0
    idx = bucket_to_idx(dow, hour_bin, n_hour_bins)

    if idx >= len(time_mult):
        return time_mult

    # Compute target multiplier
    purchased_profile = purchased_profile / (purchased_profile.sum() + 1e-9)
    target_mult = purchased_profile / (population_prior + 1e-9)
    target_mult = np.clip(target_mult, config.min_mult, config.max_mult)

    # EMA update
    time_mult[idx] = (1 - alpha) * time_mult[idx] + alpha * target_mult

    return time_mult


def get_bucket_label(dow: int, hour_bin: int, config: Optional[TimeSignatureConfig] = None) -> str:
    """Get human-readable label for a time bucket."""
    config = config or TimeSignatureConfig()

    day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    day = day_names[dow] if 0 <= dow < 7 else "?"

    if 0 <= hour_bin < len(config.hour_bins):
        start, end = config.hour_bins[hour_bin]
        hour_label = f"{start:02d}:00-{end:02d}:00"
    else:
        hour_label = "?"

    return f"{day} {hour_label}"
