"""Tests for LENS ad insertion."""
import numpy as np

from mosaic.lens_ads import AdCandidate, LENSConfig, LENSInserter


def _make_ads(n, moment_affinities=None):
    ads = []
    for i in range(n):
        ads.append(
            AdCandidate(
                ad_id=f"ad_{i}",
                campaign_id=f"camp_{i}",
                bid=1.0,
                p_click=0.1,
                quality_score=1.0,
                throttle=1.0,
                moment_affinities=moment_affinities,
            )
        )
    return ads


def test_boundary_confidence_prefers_explicit():
    config = LENSConfig(
        max_ad_load=1.0,
        allowed_slots=[3],
        slot_indexing="one",
        rho_insert=0.8,
        min_spacing=1,
    )
    inserter = LENSInserter(config=config)

    organic_ranking = list(range(5))
    organic_scores = {i: 1.0 / (i + 1) for i in organic_ranking}
    boundary_confidences = {1: 0.1}

    result = inserter.insert_ads(
        organic_ranking=organic_ranking,
        organic_scores=organic_scores,
        ad_candidates=_make_ads(1),
        activation_p=np.array([0.5, 0.5]),
        boundary_confidences=boundary_confidences,
    )

    assert len(result.placements) == 1
    assert result.placements[0].boundary_confidence == 0.1
    assert result.placements[0].slot_position == 2
    assert result.blocked_slots == []


def test_max_ad_load_enforced():
    config = LENSConfig(
        max_ad_load=0.1,
        allowed_slots=[3, 6, 9, 12, 15, 18],
        slot_indexing="one",
        rho_insert=0.9,
        min_spacing=1,
    )
    inserter = LENSInserter(config=config)

    organic_ranking = list(range(20))
    organic_scores = {i: 1.0 / (i + 1) for i in organic_ranking}
    boundary_confidences = {i: 0.0 for i in range(19)}

    result = inserter.insert_ads(
        organic_ranking=organic_ranking,
        organic_scores=organic_scores,
        ad_candidates=_make_ads(10),
        activation_p=np.array([0.5, 0.5]),
        max_ads=10,
        boundary_confidences=boundary_confidences,
    )

    assert len(result.placements) == 2


def test_slot_indexing_one_based():
    config = LENSConfig(
        max_ad_load=1.0,
        allowed_slots=[3, 8],
        slot_indexing="one",
        rho_insert=0.9,
        min_spacing=1,
    )
    inserter = LENSInserter(config=config)

    organic_ranking = list(range(10))
    organic_scores = {i: 1.0 / (i + 1) for i in organic_ranking}
    boundary_confidences = {1: 0.1, 6: 0.1}

    result = inserter.insert_ads(
        organic_ranking=organic_ranking,
        organic_scores=organic_scores,
        ad_candidates=_make_ads(2),
        activation_p=np.array([0.5, 0.5]),
        boundary_confidences=boundary_confidences,
    )

    boundaries = {p.organic_boundary for p in result.placements}
    assert boundaries == {1, 6}
    for placement in result.placements:
        slot_type, slot_item = result.final_feed[placement.slot_position]
        assert slot_type == "sponsored"
        assert slot_item == placement.ad_id


def test_receipts_values():
    config = LENSConfig(
        max_ad_load=1.0,
        allowed_slots=[3],
        slot_indexing="one",
        rho_insert=0.9,
        min_spacing=1,
        lambda_ad=0.1,
    )
    inserter = LENSInserter(config=config)

    organic_ranking = list(range(6))
    organic_scores = {i: 1.0 / (i + 1) for i in organic_ranking}
    boundary_confidences = {1: 0.1}

    activation_p = np.array([0.25, 0.75])
    affinities = np.array([0.8, 0.2])
    ad = AdCandidate(
        ad_id="ad_1",
        campaign_id="camp_1",
        bid=2.0,
        p_click=0.5,
        quality_score=0.5,
        throttle=1.0,
        moment_affinities=affinities,
    )
    result = inserter.insert_ads(
        organic_ranking=organic_ranking,
        organic_scores=organic_scores,
        ad_candidates=[ad],
        activation_p=activation_p,
        boundary_confidences=boundary_confidences,
    )

    placement = result.placements[0]
    ev = ad.throttle * ad.bid * ad.p_click * ad.quality_score
    alignment = float(np.dot(affinities, activation_p))
    utility = ev + config.lambda_ad * alignment

    assert np.isclose(placement.expected_value, ev)
    assert np.isclose(placement.mission_alignment, alignment)
    assert np.isclose(placement.utility, utility)


def test_revenue_proxy_and_topk_churn():
    config = LENSConfig(
        max_ad_load=1.0,
        allowed_slots=[3, 8],
        slot_indexing="one",
        rho_insert=0.9,
        min_spacing=1,
    )
    inserter = LENSInserter(config=config)

    organic_ranking = list(range(12))
    organic_scores = {i: 1.0 / (i + 1) for i in organic_ranking}
    boundary_confidences = {i: 0.1 for i in range(11)}

    baseline = inserter.insert_ads(
        organic_ranking=organic_ranking,
        organic_scores=organic_scores,
        ad_candidates=[],
        activation_p=np.array([0.5, 0.5]),
        boundary_confidences=boundary_confidences,
    )

    ads = _make_ads(3, moment_affinities=np.array([0.6, 0.4]))
    with_ads = inserter.insert_ads(
        organic_ranking=organic_ranking,
        organic_scores=organic_scores,
        ad_candidates=ads,
        activation_p=np.array([0.5, 0.5]),
        boundary_confidences=boundary_confidences,
    )

    assert with_ads.total_ad_value > baseline.total_ad_value

    top_k = 5
    topk_before = set(organic_ranking[:top_k])
    topk_after = [
        item for typ, item in with_ads.final_feed[:top_k] if typ == "organic"
    ]
    churn = len([item for item in topk_before if item not in topk_after])

    assert churn <= 1


def test_high_confidence_boundary_blocks_slot():
    config = LENSConfig(
        max_ad_load=1.0,
        allowed_slots=[3],
        slot_indexing="one",
        rho_insert=0.8,
        min_spacing=1,
    )
    inserter = LENSInserter(config=config)

    organic_ranking = list(range(6))
    organic_scores = {i: 1.0 / (i + 1) for i in organic_ranking}
    boundary_confidences = {1: 0.95}

    result = inserter.insert_ads(
        organic_ranking=organic_ranking,
        organic_scores=organic_scores,
        ad_candidates=_make_ads(1),
        activation_p=np.array([0.5, 0.5]),
        boundary_confidences=boundary_confidences,
    )

    assert result.placements == []
    assert 3 in result.blocked_slots


def test_min_spacing_enforced():
    config = LENSConfig(
        max_ad_load=1.0,
        allowed_slots=[3, 5, 7],
        slot_indexing="one",
        rho_insert=0.9,
        min_spacing=5,
    )
    inserter = LENSInserter(config=config)

    organic_ranking = list(range(12))
    organic_scores = {i: 1.0 / (i + 1) for i in organic_ranking}
    boundary_confidences = {i: 0.0 for i in range(11)}

    result = inserter.insert_ads(
        organic_ranking=organic_ranking,
        organic_scores=organic_scores,
        ad_candidates=_make_ads(3),
        activation_p=np.array([0.5, 0.5]),
        boundary_confidences=boundary_confidences,
    )

    assert len(result.placements) == 1
