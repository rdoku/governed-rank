"""Patterna platform shared state singleton.

This module holds the runtime state (model weights, embeddings, catalog, etc.)
used when MOSAIC is deployed inside the full PatternaAi server stack. It is
**not** required for standalone governed-rank usage — the core MOSAIC algorithm
(MOSAICScorer, orthogonalization, gap calibration, isotonic projection) works
without this module.
"""
import os, sys
from typing import Dict, Optional

import numpy as np


# Repo root path (five levels up from src/mosaic/_compat/state.py → PatternaAi/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if ROOT not in sys.path:
    sys.path.append(ROOT)


class _State:
    model = None
    device = None
    item2vec = None
    moment2vec = None
    moment2vec_checksum: Optional[str] = None
    moment_space_id: Optional[str] = None
    persona_start: Optional[int] = None
    num_items: int = 0

    context_bias_default: float = float(os.environ.get('PATTERNA_CONTEXT_BIAS', 0.07))
    # Router params tuned via Optuna (research best: tau=0.4162, temp=10.38)
    router_tau: float = float(os.environ.get('PATTERNA_ROUTER_TAU', 0.4162))
    router_temp: float = float(os.environ.get('PATTERNA_ROUTER_TEMP', 10.38))
    router_floor: float = 0.15
    router_ceil: float = 0.85

    moment_activation_mode: str = os.environ.get("PATTERNA_MOMENT_ACTIVATION_MODE", "auto")

    market_enable: bool = bool(int(os.environ.get('PATTERNA_MARKET_ENABLE', '0')))
    market = None

    # Explicit moment boost (small, gated) for provability
    explicit_moment_boost_mode: str = os.environ.get('PATTERNA_EXPLICIT_MOMENT_BOOST_MODE', 'on_small_gated')
    explicit_moment_boost_lambda: float = float(os.environ.get('PATTERNA_EXPLICIT_MOMENT_BOOST_LAMBDA', '0.03'))
    explicit_moment_boost_min_conf: float = float(os.environ.get('PATTERNA_EXPLICIT_MOMENT_BOOST_MIN_CONF', '0.35'))
    policy_boost_lambda: float = float(os.environ.get('PATTERNA_POLICY_BOOST_LAMBDA', '0.15'))

    # Satiation: diminishing returns for over-represented moments in cart
    satiation_enabled: bool = os.environ.get('PATTERNA_SATIATION_ENABLED', '1') == '1'
    satiation_rate: float = float(os.environ.get('PATTERNA_SATIATION_RATE', '0.6'))
    satiation_top_m: int = int(os.environ.get('PATTERNA_SATIATION_TOP_M', '2'))
    satiation_floor: float = float(os.environ.get('PATTERNA_SATIATION_FLOOR', '0.25'))

    # Catalog and labels
    catalog = None  # item_id -> {name, aisle, dept}
    audit_dir: str = os.path.join(ROOT, 'artifacts', 'audit_logs')
    moment_labels_path: str = os.path.join(ROOT, 'artifacts', 'moment_labels.json')
    moment_labels = None
    moment_labels_by_shop = {}
    moment_meta_by_shop = {}
    moment_policies_by_shop = {}

    # ==========================================================================
    # New Algorithm Components (from new_algo.md)
    # ==========================================================================

    # Feature flag for new activation pipeline
    new_activation_enabled: bool = os.environ.get('PATTERNA_NEW_ACTIVATION_ENABLED', '0') == '1'

    # Population prior P_m (cold-start fallback)
    population_prior: Optional[np.ndarray] = None

    # Time signature table: time_mult[bucket, m]
    time_mult: Optional[np.ndarray] = None
    time_mult_config = None  # TimeSignatureConfig

    # Evidence graph: feature -> {moment: weight}
    evidence_graph: Optional[Dict[str, Dict[int, float]]] = None

    # Rank protection config
    rank_decay_start: int = int(os.environ.get('PATTERNA_RANK_DECAY_START', '5'))
    rank_decay_end: int = int(os.environ.get('PATTERNA_RANK_DECAY_END', '12'))
    mission_boost_cap: float = float(os.environ.get('PATTERNA_MISSION_BOOST_CAP', '0.04'))

    # Exploration pool
    exploration_enabled: bool = os.environ.get('PATTERNA_EXPLORATION_ENABLED', '1') == '1'
    exploration_rate: float = float(os.environ.get('PATTERNA_EXPLORATION_RATE', '0.05'))

    # Counterfactual logging
    counterfactual_enabled: bool = os.environ.get('PATTERNA_COUNTERFACTUAL_ENABLED', '0') == '1'
    counterfactual_top_n: int = int(os.environ.get('PATTERNA_COUNTERFACTUAL_TOP_N', '50'))

    # Impression counts for cold-start (loaded separately)
    impression_counts: Optional[Dict[int, int]] = None


S = _State()
