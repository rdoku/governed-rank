"""Tests for new algorithm components (from new_algo.md)."""
import pytest
import numpy as np
from datetime import datetime
from typing import Dict, List


# =============================================================================
# Test Moment Priors
# =============================================================================

class TestMomentPriors:
    """Tests for server/core/moment_priors.py extensions."""

    @pytest.fixture
    def moment2vec(self):
        """Sample moment2vec matrix (10 items, 4 moments)."""
        A = np.array([
            [0.6, 0.2, 0.1, 0.1],  # Item 0: Party-heavy
            [0.5, 0.3, 0.1, 0.1],  # Item 1: Party-heavy
            [0.1, 0.7, 0.1, 0.1],  # Item 2: Breakfast-heavy
            [0.1, 0.6, 0.2, 0.1],  # Item 3: Breakfast-heavy
            [0.1, 0.1, 0.6, 0.2],  # Item 4: StockUp-heavy
            [0.2, 0.1, 0.5, 0.2],  # Item 5: StockUp-heavy
            [0.1, 0.1, 0.2, 0.6],  # Item 6: Healthy-heavy
            [0.1, 0.2, 0.1, 0.6],  # Item 7: Healthy-heavy
            [0.25, 0.25, 0.25, 0.25],  # Item 8: Uniform
            [0.4, 0.3, 0.2, 0.1],  # Item 9: Mixed
        ], dtype=np.float32)
        return A

    def test_compute_cart_prior_party_items(self, moment2vec):
        """Cart with party items should have high party prior."""
        from mosaic.moment_priors import compute_cart_prior

        cart = [0, 1]  # Party-heavy items
        prior = compute_cart_prior(cart, moment2vec)

        assert prior.shape == (4,)
        assert abs(prior.sum() - 1.0) < 1e-6
        assert prior[0] > 0.4  # Party moment should dominate

    def test_compute_cart_prior_empty(self, moment2vec):
        """Empty cart should return uniform prior."""
        from mosaic.moment_priors import compute_cart_prior

        prior = compute_cart_prior([], moment2vec)

        assert prior.shape == (4,)
        assert abs(prior.sum() - 1.0) < 1e-6
        assert all(abs(p - 0.25) < 1e-6 for p in prior)

    def test_compute_history_prior_recency(self, moment2vec):
        """More recent items should have higher weight."""
        from mosaic.moment_priors import compute_history_prior

        # History: old items are party, recent items are healthy
        history = [0, 1, 6, 7]  # Party (old) -> Healthy (recent)
        prior = compute_history_prior(history, moment2vec, decay=0.5)

        assert prior.shape == (4,)
        assert abs(prior.sum() - 1.0) < 1e-6
        # Healthy should be higher due to recency weighting
        assert prior[3] > prior[0]

    def test_compute_population_prior(self, moment2vec):
        """Population prior should aggregate across orders."""
        from mosaic.moment_priors import compute_population_prior

        orders = [
            [0, 1],  # Party order
            [2, 3],  # Breakfast order
            [4, 5],  # StockUp order
            [6, 7],  # Healthy order
        ]
        P_m = compute_population_prior(orders, moment2vec)

        assert P_m.shape == (4,)
        assert abs(P_m.sum() - 1.0) < 1e-6
        # Should be roughly balanced across moments
        assert all(p > 0.1 for p in P_m)

    def test_compute_combined_prior_with_cart(self, moment2vec):
        """Combined prior with cart should prioritize cart."""
        from mosaic.moment_priors import compute_combined_prior

        cart = [0, 1]  # Party items
        history = [6, 7]  # Healthy items
        prior, source = compute_combined_prior(cart, history, moment2vec)

        assert source == "cart"
        assert prior.shape == (4,)
        # Party should still be prominent due to cart weight
        assert prior[0] > prior[3]

    def test_compute_combined_prior_no_cart(self, moment2vec):
        """Combined prior without cart should use history."""
        from mosaic.moment_priors import compute_combined_prior

        cart = []
        history = [6, 7]  # Healthy items
        prior, source = compute_combined_prior(cart, history, moment2vec)

        assert source == "history"
        assert prior[3] > prior[0]  # Healthy should dominate


# =============================================================================
# Test Time Signatures
# =============================================================================

class TestTimeSignatures:
    """Tests for server/core/time_signatures.py."""

    def test_get_time_bucket(self):
        """Test time bucket extraction."""
        from mosaic.time_signatures import get_time_bucket, TimeSignatureConfig

        config = TimeSignatureConfig()

        # Sunday 8am -> dow=0, hour_bin=1 (6-9)
        ts = datetime(2024, 1, 7, 8, 0)  # Sunday
        dow, hour_bin = get_time_bucket(ts, config)
        assert dow == 0
        assert hour_bin == 1

        # Friday 7pm -> dow=5, hour_bin=5 (18-21)
        ts = datetime(2024, 1, 12, 19, 0)  # Friday
        dow, hour_bin = get_time_bucket(ts, config)
        assert dow == 5
        assert hour_bin == 5

    def test_compute_time_signatures(self):
        """Test time signature computation."""
        from mosaic.time_signatures import compute_time_signatures, TimeSignatureConfig

        # Simple moment2vec
        moment2vec = np.array([
            [0.8, 0.2],
            [0.2, 0.8],
            [0.5, 0.5],
        ], dtype=np.float32)

        # Orders at different times
        orders = [
            (datetime(2024, 1, 7, 8, 0), [0]),  # Sunday morning -> item 0
            (datetime(2024, 1, 7, 8, 30), [0]),
            (datetime(2024, 1, 12, 19, 0), [1]),  # Friday evening -> item 1
            (datetime(2024, 1, 12, 20, 0), [1]),
        ]

        config = TimeSignatureConfig()
        time_mult = compute_time_signatures(orders, moment2vec, config=config)

        assert time_mult.shape[1] == 2  # K=2 moments
        assert time_mult.shape[0] == 7 * len(config.hour_bins)  # dow x hour_bins

    def test_time_multiplier_clamping(self):
        """Time multipliers should be clamped to [min, max]."""
        from mosaic.time_signatures import compute_time_signatures, TimeSignatureConfig

        moment2vec = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        orders = [
            (datetime(2024, 1, 7, 8, 0), [0]),
        ] * 100  # All at same time

        config = TimeSignatureConfig(min_mult=0.5, max_mult=2.0)
        time_mult = compute_time_signatures(orders, moment2vec, config=config)

        assert np.all(time_mult >= 0.5)
        assert np.all(time_mult <= 2.0)

    def test_update_time_signatures_ema(self):
        """EMA update should adjust only the matching time bucket."""
        from mosaic.time_signatures import (
            TimeSignatureConfig,
            update_time_signatures_ema,
            get_time_bucket,
            bucket_to_idx,
        )

        config = TimeSignatureConfig()
        K = 2
        n_buckets = (7 if config.use_dow else 1) * len(config.hour_bins)
        time_mult = np.ones((n_buckets, K), dtype=np.float32)

        population_prior = np.array([0.5, 0.5], dtype=np.float32)
        purchased_profile = np.array([0.8, 0.2], dtype=np.float32)

        ts = datetime(2024, 1, 7, 8, 0)  # Sunday 8am
        dow, hour_bin = get_time_bucket(ts, config)
        idx = bucket_to_idx(dow, hour_bin, len(config.hour_bins))

        updated = update_time_signatures_ema(
            time_mult.copy(),
            timestamp=ts,
            purchased_profile=purchased_profile,
            population_prior=population_prior,
            config=config,
            alpha=0.5,
        )

        target_mult = purchased_profile / population_prior
        target_mult = np.clip(target_mult, config.min_mult, config.max_mult)
        expected = 0.5 * np.ones(K) + 0.5 * target_mult

        assert np.allclose(updated[idx], expected, atol=1e-6)
        other_idx = 0 if idx != 0 else 1
        assert np.allclose(updated[other_idx], np.ones(K), atol=1e-6)


# =============================================================================
# Test Feature Extractor
# =============================================================================

class TestFeatureExtractor:
    """Tests for server/core/feature_extractor.py."""

    @pytest.fixture
    def catalog(self):
        """Sample catalog."""
        return {
            0: {"name": "Tortilla Chips", "aisle": "chips pretzels", "dept": "snacks"},
            1: {"name": "Salsa", "aisle": "salsa", "dept": "snacks"},
            2: {"name": "Beer", "aisle": "beer", "dept": "alcohol"},
            3: {"name": "Greek Yogurt", "aisle": "yogurt", "dept": "dairy"},
            4: {"name": "Eggs", "aisle": "eggs", "dept": "dairy"},
            5: {"name": "Bread", "aisle": "bread", "dept": "bakery"},
        }

    def test_extract_cart_structure_features(self):
        """Test cart structure feature extraction."""
        from mosaic.feature_extractor import extract_cart_structure_features

        assert "empty_cart" in extract_cart_structure_features(0)
        assert "single_item" in extract_cart_structure_features(1)
        assert "small_cart" in extract_cart_structure_features(3)
        assert "large_cart" in extract_cart_structure_features(12)
        assert "very_large_cart" in extract_cart_structure_features(20)

    def test_extract_cart_content_features(self, catalog):
        """Test cart content feature extraction."""
        from mosaic.feature_extractor import extract_cart_content_features

        cart = [0, 1, 2]  # Chips, salsa, beer
        features = extract_cart_content_features(cart, catalog)

        assert "snacks_present" in features
        assert "chips_present" in features
        assert "salsa_present" in features
        assert "alcohol_present" in features

    def test_extract_combo_features(self):
        """Test combo feature extraction."""
        from mosaic.feature_extractor import extract_combo_features

        content = {"chips_present", "salsa_present", "snacks_present"}
        combos = extract_combo_features(content)

        assert "chips_and_salsa" in combos

    def test_extract_context_features(self):
        """Test context feature extraction."""
        from mosaic.feature_extractor import extract_context_features

        context = {
            "device": "mobile",
            "channel": "delivery",
            "is_new_user": True,
        }
        features = extract_context_features(context)

        assert "mobile_device" in features
        assert "delivery_channel" in features
        assert "new_user" in features

    def test_extract_features_integration(self, catalog):
        """Test full feature extraction."""
        from mosaic.feature_extractor import extract_features

        cart = [0, 1]  # Chips and salsa
        context = {"device": "mobile"}

        result = extract_features(cart, context, catalog)

        assert "chips_and_salsa" in result.fired_features
        assert "mobile_device" in result.fired_features
        assert "small_cart" in result.fired_features


# =============================================================================
# Test Rank Protection
# =============================================================================

class TestRankProtection:
    """Tests for mosaic/core/rank_protection.py."""

    def test_smoothstep(self):
        """Test smoothstep interpolation."""
        from mosaic.rank_protection import smoothstep

        assert smoothstep(0.0) == 0.0
        assert smoothstep(1.0) == 1.0
        assert 0.4 < smoothstep(0.5) < 0.6  # Smooth middle

    def test_compute_rank_weights(self):
        """Test rank weight computation."""
        from mosaic.rank_protection import compute_rank_weights, RankProtectionConfig

        config = RankProtectionConfig(decay_start=5, decay_end=12)
        weights = compute_rank_weights(20, config)

        # Ranks 1-5 should be 0 (protected)
        assert all(weights[i] == 0.0 for i in range(5))

        # Ranks 6-11 should be transitioning
        assert all(0.0 < weights[i] < 1.0 for i in range(5, 11))

        # Ranks 12+ should be 1.0
        assert all(weights[i] == 1.0 for i in range(11, 20))

    def test_get_base_ranks(self):
        """Test rank computation from scores."""
        from mosaic.rank_protection import get_base_ranks

        scores = np.array([0.5, 0.9, 0.3, 0.7])
        ranks = get_base_ranks(scores)

        # Highest score (0.9 at idx 1) should be rank 1
        assert ranks[1] == 1
        # Second highest (0.7 at idx 3) should be rank 2
        assert ranks[3] == 2

    def test_apply_rank_protection_high_confidence(self):
        """High confidence should not reduce boosts."""
        from mosaic.rank_protection import apply_rank_protection, RankProtectionConfig

        config = RankProtectionConfig(decay_start=3, decay_end=6)
        ranks = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        boosts = np.array([0.01] * 8)

        gated, _ = apply_rank_protection(ranks, boosts, "high", config)

        # Ranks 1-3 should be protected (boost = 0)
        assert all(gated[i] == 0.0 for i in range(3))

        # Ranks 6+ should have full boost (with conf=1.0)
        assert gated[5] > 0

    def test_apply_rank_protection_low_confidence(self):
        """Low confidence should zero out all boosts."""
        from mosaic.rank_protection import apply_rank_protection

        ranks = np.array([1, 5, 10])
        boosts = np.array([0.01, 0.01, 0.01])

        gated, _ = apply_rank_protection(ranks, boosts, "low")

        assert all(g == 0.0 for g in gated)

    def test_compute_protected_mission_boost(self):
        """Mission boost should be rank-gated and capped."""
        from mosaic.rank_protection import compute_protected_mission_boost, RankProtectionConfig

        scores = np.array([1.0, 0.9, 0.8])
        config = RankProtectionConfig(decay_start=1, decay_end=2, mission_cap=0.02)

        boost, debug = compute_protected_mission_boost(
            item_idx=2,
            model_scores=scores,
            mission_alignment=1.0,
            satiation_factor=1.0,
            confidence_tier="high",
            mission_lambda=0.08,
            config=config,
        )

        assert abs(boost - 0.02) < 1e-6
        assert debug["was_capped"] is True

    def test_batch_compute_protected_mission_boost(self):
        """Batch protected boosts should honor rank gating and caps."""
        from mosaic.rank_protection import batch_compute_protected_mission_boost, RankProtectionConfig

        scores = np.array([1.0, 0.9])
        align = np.array([1.0, 1.0])
        sat = np.array([1.0, 1.0])
        config = RankProtectionConfig(decay_start=1, decay_end=2, mission_cap=0.01)

        boosts, _ = batch_compute_protected_mission_boost(
            model_scores=scores,
            mission_alignments=align,
            satiation_factors=sat,
            confidence_tier="high",
            mission_lambda=0.08,
            config=config,
        )

        assert boosts.shape == (2,)
        assert boosts[0] == 0.0
        assert abs(boosts[1] - 0.01) < 1e-6


# =============================================================================
# Test Activation
# =============================================================================

class TestActivation:
    """Tests for server/core/activation.py."""

    def test_compute_evidence_contribution(self):
        """Test evidence vector computation."""
        from mosaic.activation import compute_evidence_contribution

        evidence_graph = {
            "chips_present": {0: 0.5, 1: -0.1},
            "salsa_present": {0: 0.4},
            "chips_and_salsa": {0: 0.8},
        }
        fired = frozenset(["chips_present", "salsa_present", "chips_and_salsa"])

        e, contribs = compute_evidence_contribution(fired, evidence_graph, K=4)

        assert e.shape == (4,)
        assert e[0] > 1.0  # Should accumulate
        assert len(contribs) == 3

    def test_compute_rule_posterior(self):
        """Test rule posterior computation."""
        from mosaic.activation import compute_rule_posterior

        p_prior = np.array([0.4, 0.3, 0.2, 0.1])
        time_mult = np.array([1.5, 1.0, 0.8, 1.0])
        evidence = np.array([1.0, 0.0, 0.0, 0.0])

        p_rule = compute_rule_posterior(p_prior, time_mult, evidence, beta=1.0)

        assert p_rule.shape == (4,)
        assert abs(p_rule.sum() - 1.0) < 1e-6
        # Moment 0 should dominate due to prior, time, and evidence
        assert p_rule[0] > p_rule[1]

    def test_compute_evidence_volume(self):
        """Evidence volume should scale with cart and feature signal."""
        from mosaic.activation import compute_evidence_volume

        # Max cart + max features -> ev = 1
        assert compute_evidence_volume(cart_size=5, n_fired_features=8) == 1.0

        # No features -> ev = 0
        assert compute_evidence_volume(cart_size=2, n_fired_features=0) == 0.0

    def test_compute_feature_agreement(self):
        """Agreement should reflect weight share on the top moment."""
        from mosaic.activation import compute_feature_agreement

        evidence_graph = {
            "chips_present": {0: 0.5, 1: 0.5},
            "salsa_present": {0: 0.5},
        }
        fired = frozenset(["chips_present", "salsa_present"])

        agreement, has_edges = compute_feature_agreement(
            fired, evidence_graph, top_moment=0
        )

        # total weight = 1.5, top moment weight = 1.0
        assert abs(agreement - (1.0 / 1.5)) < 1e-6
        assert has_edges is True

    def test_compute_blending_gate(self):
        """Blending gate should respond to evidence/model conditions."""
        from mosaic.activation import compute_blending_gate, ActivationConfig

        config = ActivationConfig()

        alpha, reason = compute_blending_gate(
            evidence_volume=0.5,
            agreement=0.5,
            model_confidence=None,
            history_strength=0.0,
            has_model=False,
            config=config,
        )
        assert alpha == 0.0
        assert reason == "no_model"

        alpha, reason = compute_blending_gate(
            evidence_volume=0.9,
            agreement=0.8,
            model_confidence=0.4,
            history_strength=0.2,
            has_model=True,
            config=config,
        )
        assert alpha == config.alpha_strong_rules
        assert reason == "strong_evidence"

        alpha, reason = compute_blending_gate(
            evidence_volume=0.1,
            agreement=0.2,
            model_confidence=0.2,
            history_strength=0.1,
            has_model=True,
            config=config,
        )
        assert alpha == config.alpha_weak_evidence
        assert reason == "weak_evidence"

        alpha, reason = compute_blending_gate(
            evidence_volume=0.5,
            agreement=0.3,
            model_confidence=0.7,
            history_strength=0.6,
            has_model=True,
            config=config,
        )
        assert alpha == config.alpha_strong_model
        assert reason == "model_confident"

    def test_compute_temperature(self):
        """Test temperature computation."""
        from mosaic.activation import compute_temperature, apply_temperature

        # High evidence -> low temperature (confident)
        T_high_ev = compute_temperature(0.9)
        assert T_high_ev < 1.5

        # Low evidence -> high temperature (uncertain)
        T_low_ev = compute_temperature(0.1)
        assert T_low_ev > 2.0

        # Temperature should flatten the distribution
        p = np.array([0.7, 0.2, 0.1])
        p_flat = apply_temperature(p, T=2.0)
        assert abs(p_flat.sum() - 1.0) < 1e-6
        assert p_flat.max() < p.max()

    def test_compute_confidence_tier(self):
        """Test confidence tiering."""
        from mosaic.activation import compute_confidence_tier

        # Strong signal: high tier
        p_strong = np.array([0.8, 0.1, 0.05, 0.05])
        tier, _, _ = compute_confidence_tier(p_strong, 0.7, 0.8)
        assert tier == "high"

        # Weak signal: low tier
        p_weak = np.array([0.3, 0.25, 0.25, 0.2])
        tier, _, _ = compute_confidence_tier(p_weak, 0.2, 0.3)
        assert tier == "low"

    def test_compute_hybrid_activation(self):
        """Hybrid activation should blend rule and model posteriors."""
        from mosaic.activation import compute_hybrid_activation, ActivationConfig

        moment2vec = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
        ], dtype=np.float32)

        evidence_graph = {"chips_present": {0: 1.0}}
        fired = frozenset(["chips_present"])

        p_model = np.array([0.2, 0.8], dtype=np.float32)
        config = ActivationConfig()

        result = compute_hybrid_activation(
            cart_items=[0, 1],
            history_items=[],
            fired_features=fired,
            moment2vec=moment2vec,
            time_mult=None,
            evidence_graph=evidence_graph,
            population_prior=None,
            p_model=p_model,
            config=config,
        )

        assert abs(result.p.sum() - 1.0) < 1e-6
        assert result.gate_reason == "weak_evidence"
        # With weak evidence, model should dominate
        assert result.alpha == config.alpha_weak_evidence
        assert result.p[1] > result.p[0]


# =============================================================================
# Test Counterfactual
# =============================================================================

class TestCounterfactual:
    """Tests for server/core/counterfactual.py."""

    def test_compute_ranking(self):
        """Test ranking computation."""
        from mosaic.counterfactual import compute_ranking

        scores = np.array([0.5, 0.9, 0.3, 0.7, 0.6])
        ranking, score_list = compute_ranking(scores, top_n=3)

        assert ranking == [1, 3, 4]  # Indices of top 3
        assert np.allclose(score_list, [0.9, 0.7, 0.6])

    def test_compute_rank_changes(self):
        """Test rank change computation."""
        from mosaic.counterfactual import compute_rank_changes

        baseline = [0, 1, 2, 3, 4]
        comparison = [0, 2, 1, 4, 3]  # Swapped some items

        changes = compute_rank_changes(baseline, comparison)

        assert changes["n_common"] == 5
        assert changes["n_dropped"] == 0
        assert changes["n_added"] == 0
        assert len(changes["top_rank_changes"]) > 0

    def test_compute_counterfactuals(self):
        """Test full counterfactual computation."""
        from mosaic.counterfactual import compute_counterfactuals

        model_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        mission_boosts = np.array([0.0, 0.0, 0.1, 0.0, 0.0])  # Boost item 2
        policy_boosts = np.array([0.0, 0.0, 0.0, 0.0, 0.05])  # Boost item 4

        served_scores = model_scores + mission_boosts + policy_boosts
        result = compute_counterfactuals(
            base_scores=model_scores,
            mission_component=mission_boosts,
            policy_component=policy_boosts,
            served_scores=served_scores,
        )

        assert result.final_ranking is not None
        assert result.pure_model_ranking is not None
        assert result.mission_impact is not None


# =============================================================================
# Test Exploration Pool
# =============================================================================

class TestExplorationPool:
    """Tests for server/core/exploration_pool.py."""

    @pytest.fixture
    def moment2vec(self):
        """Sample moment2vec (10 items, 4 moments)."""
        return np.random.rand(10, 4).astype(np.float32)

    def test_sample_cold_start_items(self):
        """Test cold-start item sampling."""
        from mosaic.exploration_pool import sample_cold_start_items

        available = set(range(10))
        impressions = {0: 50, 1: 150, 2: 30, 3: 200, 4: 10}

        rng = np.random.default_rng(1)
        cold = sample_cold_start_items(
            available, impressions, n_items=3, threshold=100, rng=rng
        )

        assert len(cold) == 3
        assert 1 not in cold  # 150 impressions
        assert 3 not in cold  # 200 impressions

    def test_generate_exploration_pool(self, moment2vec):
        """Test exploration pool generation."""
        from mosaic.exploration_pool import generate_exploration_pool

        history_pool = [0, 1, 2]
        context_pool = [3, 4]
        available = set(range(10))
        activation = np.array([0.5, 0.3, 0.15, 0.05])

        result = generate_exploration_pool(
            history_pool=history_pool,
            context_pool=context_pool,
            available_items=available,
            moment2vec=moment2vec,
            activation=activation,
            confidence_tier="medium",
        )

        assert len(result.item_ids) > 0
        # Should not include items from H or C pools
        assert not any(i in history_pool for i in result.item_ids)
        assert not any(i in context_pool for i in result.item_ids)

    def test_merge_pools_with_provenance(self, moment2vec):
        """Test pool merging with provenance tracking."""
        from mosaic.exploration_pool import (
            generate_exploration_pool,
            merge_pools_with_provenance,
        )

        history_pool = [0, 1]
        context_pool = [2, 3]
        cart = [5]
        available = set(range(10))
        activation = np.array([0.5, 0.3, 0.15, 0.05])

        explore_result = generate_exploration_pool(
            history_pool=history_pool,
            context_pool=context_pool,
            available_items=available,
            moment2vec=moment2vec,
            activation=activation,
        )

        merged, provenance = merge_pools_with_provenance(
            history_pool, context_pool, explore_result, cart
        )

        # Cart items should be excluded
        assert 5 not in merged

        # Check provenance
        assert "history" in provenance.get(0, set())
        assert "context" in provenance.get(2, set())


# =============================================================================
# Test Evidence Graph
# =============================================================================

class TestEvidenceGraph:
    """Tests for server/core/evidence_graph.py."""

    @pytest.fixture
    def moment2vec(self):
        """Sample moment2vec (10 items, 4 moments)."""
        A = np.array([
            [0.7, 0.1, 0.1, 0.1],  # Item 0: Party
            [0.6, 0.2, 0.1, 0.1],  # Item 1: Party
            [0.1, 0.7, 0.1, 0.1],  # Item 2: Breakfast
            [0.1, 0.6, 0.2, 0.1],  # Item 3: Breakfast
            [0.1, 0.1, 0.7, 0.1],  # Item 4: StockUp
            [0.2, 0.1, 0.6, 0.1],  # Item 5: StockUp
            [0.1, 0.1, 0.1, 0.7],  # Item 6: Healthy
            [0.1, 0.2, 0.1, 0.6],  # Item 7: Healthy
            [0.25, 0.25, 0.25, 0.25],  # Item 8: Uniform
            [0.4, 0.3, 0.2, 0.1],  # Item 9: Mixed
        ], dtype=np.float32)
        return A

    def test_build_evidence_graph(self, moment2vec):
        """Test evidence graph building from orders."""
        from mosaic.evidence_graph import build_evidence_graph, EvidenceGraphConfig

        # Orders with features
        orders = [
            ([0, 1], frozenset(["chips_present", "salsa_present"])),  # Party
            ([0, 1], frozenset(["chips_present", "snacks_present"])),  # Party
            ([2, 3], frozenset(["eggs_present", "breakfast_present"])),  # Breakfast
            ([2, 3], frozenset(["cereal_present", "breakfast_present"])),  # Breakfast
            ([4, 5], frozenset(["large_cart", "household_present"])),  # StockUp
            ([6, 7], frozenset(["produce_present", "yogurt_present"])),  # Healthy
        ] * 20  # Repeat to meet min_support

        config = EvidenceGraphConfig(
            min_feature_support=10,
            weight_threshold=0.05,
        )
        graph = build_evidence_graph(orders, moment2vec, config)

        assert graph.n_features > 0
        assert graph.n_edges > 0
        assert graph.n_moments == 4

        # chips_present should boost party (moment 0)
        chips_weight = graph.get_weight("chips_present", 0)
        assert chips_weight > 0

    def test_evidence_graph_serialization(self, moment2vec):
        """Test evidence graph save/load."""
        from mosaic.evidence_graph import (
            build_evidence_graph,
            save_evidence_graph,
            load_evidence_graph,
            EvidenceGraphConfig,
        )
        import tempfile

        orders = [
            ([0, 1], frozenset(["feature_a"])),
            ([2, 3], frozenset(["feature_b"])),
        ] * 100

        config = EvidenceGraphConfig(min_feature_support=10)
        graph = build_evidence_graph(orders, moment2vec, config)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_evidence_graph(graph, tmpdir, "test_space")
            loaded = load_evidence_graph(tmpdir, "test_space")

            assert loaded is not None
            assert loaded.n_features == graph.n_features
            assert loaded.n_edges == graph.n_edges

    def test_create_default_evidence_graph(self):
        """Test default evidence graph creation."""
        from mosaic.evidence_graph import create_default_evidence_graph

        graph = create_default_evidence_graph(K=8)

        assert graph.n_moments == 8
        assert graph.n_features > 0

        # Should have party-related features
        assert "chips_and_salsa" in graph.weights

    def test_get_top_features_for_moment(self, moment2vec):
        """Test getting top features for a moment."""
        from mosaic.evidence_graph import (
            build_evidence_graph,
            get_top_features_for_moment,
            EvidenceGraphConfig,
        )

        orders = [
            ([0, 1], frozenset(["party_feature"])),
        ] * 100

        config = EvidenceGraphConfig(min_feature_support=10)
        graph = build_evidence_graph(orders, moment2vec, config)

        top = get_top_features_for_moment(graph, 0, top_n=5)
        assert len(top) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
