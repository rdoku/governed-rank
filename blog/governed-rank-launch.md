# You Added a Safety Score. Your Ranker Got Worse. Here's the Fix.

You add a toxicity score to your content ranker. Accuracy drops 5%. Your PM asks why. You try weighting it lower — still drops. You try a threshold filter — now you're cutting good content. The safety team wants more steering; the relevance team wants less. Everyone's frustrated.

This is the injection problem, and it happens in every domain: content moderation, fairness, fraud, RAG safety. The moment you bolt a policy signal onto a ranked list, you break the ordering that your base model learned.

We built `governed-rank` to solve it.

## Three Lines

```python
from mosaic import govern

result = govern(
    base_scores={"doc1": 0.9, "doc2": 0.8, "doc3": 0.7, "doc4": 0.6, "doc5": 0.5},
    steering_scores={"doc1": -0.5, "doc2": 0.3, "doc3": 0.8, "doc4": 0.1, "doc5": 0.6},
    budget=0.3,
)

print(result.ranked_items)   # reranked order
print(result.receipts)       # per-item audit trail
```

That's it. `base_scores` is whatever your ranker produces. `steering_scores` is your policy signal — toxicity penalties, fairness boosts, groundedness scores, margin targets. `budget` controls how much of the original ordering is protected.

The result: a reranked list that maximizes your policy effect while preserving accuracy where the base model is confident. Plus an audit receipt for every item.

## Why Naive Injection Fails

When your policy signal correlates with your base scores — and it almost always does — adding them together creates interference. A document that's both relevant and toxic gets a mid-range blended score, which pushes it into positions it doesn't belong. The math is simple: if `corr(base, steering) != 0`, then `base + w*steering` distorts the ranking in ways that neither signal intended.

`governed-rank` handles this in three steps:

1. **Orthogonalize.** Project the steering signal into the null space of the base score direction. This removes the component that would interfere with accuracy. The steering signal that remains is "pure policy" — it can only move items where the base ranker doesn't have a strong opinion.

2. **Protect edges.** The `budget` parameter locks the most confident ordering decisions. If `budget=0.3`, the top 30% of adjacent-pair score gaps are frozen. Steering cannot flip these pairs. This is your accuracy guarantee.

3. **Project.** Constrained isotonic regression finds the final scores that maximize policy effect while respecting every protected constraint. The result is Pareto-optimal: you literally cannot get more policy effect without sacrificing more accuracy.

No hyperparameter tuning. No retraining. Works on any ranked list.

## Three Domains, One API

### Content Moderation

Demote toxic content without hurting engagement:

```python
result = govern(
    base_scores=engagement_scores,
    steering_scores=toxicity_penalties,  # negative = toxic
    budget=0.3,
)
```

Toxic items drop in ranking. Engagement-critical ordering preserved. Every demotion logged in `result.receipts`.

### Fairness

Boost underrepresented groups without sacrificing quality:

```python
result = govern(
    base_scores=quality_scores,
    steering_scores=fairness_boosts,  # positive for underrepresented
    budget=0.3,
)
```

On the COMPAS recidivism dataset, this achieves an adverse impact ratio of 0.963 while retaining 98.65% of base quality. That's near-parity fairness with almost no accuracy cost.

### RAG Safety

Steer retrieval toward grounded, policy-safe documents:

```python
result = govern(
    base_scores=retrieval_scores,       # embedding similarity
    steering_scores=groundedness_scores, # factuality signal
    budget=0.5,
)
```

Grounded documents promoted. Top retrieval results still relevant. Hallucination risk reduced at the ranking layer — before the LLM ever sees the context.

## The Budget Knob

`budget` is the single parameter that controls the accuracy-policy tradeoff, and it's interpretable: it's the fraction of ordering decisions that are locked.

| Budget | Meaning | Use case |
|--------|---------|----------|
| 0.0 | Full reorder by policy | Hard compliance (block all flagged items) |
| 0.3 | Protect 30% of edges | Balanced — default for most domains |
| 0.5 | Protect half | Conservative steering, high-stakes ranking |
| 0.8 | Protect 80% | Light nudge, minimal disruption |
| 1.0 | No change | Pass-through (useful for A/B tests) |

This isn't a weight you have to tune. It's a guarantee: at `budget=0.3`, the 30% most confident ordering decisions in your base ranker will not be altered, period. The Pareto curve from our experiments shows that 0.3 achieves 0.890 rank stability at 0.344 policy exposure — dominant over naive weighted combination at every point on the curve.

## Validated on 17 Datasets

We didn't validate on one benchmark. We tested across 6 domains and 17 datasets:

| Domain | Datasets | Key Result |
|--------|----------|------------|
| Recommendations | Ta Feng, Instacart, RetailRocket, Criteo, MovieLens, Amazon Reviews, Yelp | 0.890 stability @ 0.344 exposure |
| Fairness | COMPAS, Adult Income, German Credit | adverse_impact_ratio = 0.963 |
| Healthcare | MIMIC-IV, SynPUF | 71.6% HIGH tier, 5.0x NMF lift |
| Content / NLP | AG News, BBC News, Mind News | Cross-domain discovery |
| Fraud | IEEE-CIS Fraud | Policy-steered detection |
| Cookieless Targeting | RetailRocket, Criteo | 4.65x CVR lift |

122 tests. 99 KB wheel. Zero heavy dependencies beyond NumPy, SciPy, and scikit-learn.

## The Discovery Engine

Here's a bonus that surprised us during development. Before you even deploy a steering policy, you can ask: *which policies are naturally aligned with my users?*

```python
from mosaic.discovery import DiscoveryEngine

engine = DiscoveryEngine()
report = engine.discover(sessions, catalog)

for opp in report.top_opportunities(5):
    print(f"{opp.category}: {opp.preference_lift:.1f}x lift")
```

The discovery engine analyzes behavioral data to find objectives that users already gravitate toward. The insight: don't optimize diversity directly — optimize quality in the direction users are already moving. This turns governed reranking from a compliance tool into a growth tool.

## Install

```bash
pip install governed-rank
```

Requires Python 3.9+. Optional extras:

```bash
pip install governed-rank[torch]   # PyTorch integration
pip install governed-rank[viz]     # matplotlib + pandas for Pareto plots
pip install governed-rank[dev]     # pytest + dev tools
```

Source and docs: [github.com/rdoku/governed-rank](https://github.com/rdoku/governed-rank)

## What's Next

`governed-rank` is Apache 2.0 licensed and ready to drop into any Python pipeline. We're working on:

- **Streaming mode** for real-time ranking (sub-ms latency target)
- **Pre-built policy packs** for common domains (content safety, fair lending, RAG)
- **Dashboard** for visualizing Pareto curves and auditing receipts

If your team is wrestling with the accuracy-policy tradeoff in ranked systems, give `govern()` a try. Three lines. No retraining. Full audit trail.

---

*Built by [PatternaAi](https://github.com/rdoku). Questions or feedback: ronald.doku@gmail.com*
