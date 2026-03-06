# Content Moderation for News Feeds — Demote Toxic Content Without Killing Engagement

Your content ranker maximizes engagement. Your trust & safety team flags toxic content. You add toxicity penalties to the ranking. Engagement drops 5%. The product team asks you to revert.

`governed-rank` solves this: demote toxic content while preserving the engagement-critical ordering decisions your ranker learned.

## The Setup

News feeds rank articles by predicted engagement (click probability, read time, shares). Toxicity classifiers produce a separate signal. The naive approach fails because engagement and toxicity are correlated — controversial content drives both.

```python
from mosaic import govern

# engagement_scores: predicted engagement from your ranker
# toxicity_penalties: negative for toxic content, 0 for clean
#   e.g., toxicity_classifier returns 0-1, we negate it
toxicity_penalties = {
    article_id: -toxicity_classifier.predict(article)
    for article_id, article in articles.items()
}

result = govern(
    base_scores=engagement_scores,
    steering_scores=toxicity_penalties,
    budget=0.3,
)
```

## Cross-Domain Validation

We validated this pattern across three news datasets:

| Dataset | Articles | Categories | Use |
|---------|----------|------------|-----|
| AG News | 120K | 4 (World, Sports, Business, Sci/Tech) | Topic steering |
| BBC News | 2,225 | 5 (Business, Entertainment, Politics, Sport, Tech) | Category balance |
| MIND News | 160K+ | 18 categories | Personalized feed steering |

The `DiscoveryEngine` identified natural category imbalances across all three datasets — some categories are overrepresented in user reads relative to the catalog, others are underserved.

## Using Discovery to Find What to Steer

Before you decide what to demote or promote, ask: what does user behavior tell you?

```python
from mosaic.discovery import DiscoveryEngine, DiscoveryConfig

engine = DiscoveryEngine(config=DiscoveryConfig(
    min_lift=1.2,     # categories users read more than catalog rate
    max_lift=0.7,     # categories that are oversupplied
    min_reads=50,
))

report = engine.discover(
    sessions=user_sessions,   # {session_id: [{"article_id": ..., "category": ...}, ...]}
    catalog=article_catalog,  # {article_id: {"category": "Politics", ...}}
)

print(f"Opportunities (users want more):")
for opp in report.opportunities:
    print(f"  {opp.category}: {opp.preference_lift:.1f}x lift — {opp.action.value}")

print(f"\nOversupplied (too much in feed):")
for over in report.oversupply:
    print(f"  {over.category}: {over.preference_lift:.1f}x — {over.action.value}")
```

This might reveal: users read Politics at 2.3x the catalog rate (opportunity to promote), while Sports is at 0.6x (oversupplied — users scroll past it).

## Steering Based on Discovery

Turn discovery insights into steering scores:

```python
import numpy as np

# Build steering signal from discovery
steering_scores = {}
for article_id, article in articles.items():
    category = article["category"]
    # Find the discovery insight for this category
    discovery = next((d for d in report.discoveries if d.category == category), None)
    if discovery and discovery.action.value == "PROMOTE":
        steering_scores[article_id] = discovery.preference_lift - 1.0  # positive boost
    elif discovery and discovery.action.value in ("REDUCE", "CUT"):
        steering_scores[article_id] = -(1.0 - discovery.preference_lift)  # negative
    else:
        steering_scores[article_id] = 0.0

result = govern(
    base_scores=engagement_scores,
    steering_scores=steering_scores,
    budget=0.3,
)
```

## Full Pipeline with MOSAICScorer

For production news feeds with category-aware moments:

```python
from mosaic import MOSAICScorer, MOSAICConfig

# Category affinities: A[i, c] = 1.0 if article i has category c
A = np.zeros((n_articles, n_categories))
for i, article_id in enumerate(article_ids):
    cat_idx = category_to_idx[articles[article_id]["category"]]
    A[i, cat_idx] = 1.0

scorer = MOSAICScorer(
    moment_affinities=A,
    config=MOSAICConfig(
        lambda_m=0.3,
        protection_mode="budget",
        budget_pct=0.30,
    ),
)

# Per-user activation based on their reading history
activation_p = compute_user_category_preferences(user_history)

result = scorer.rank(
    candidates=candidate_article_ids,
    base_scores=engagement_scores,
    activation_p=activation_p,
    activation_confidence="medium",
)
```

## The Audit Story

When a reporter asks "why was this article shown?" or "why was this article suppressed?", you have a complete answer:

```python
receipt = result.receipts[article_id]
print(f"Article: {article_id}")
print(f"  Engagement score: {receipt.base_score:.4f}")
print(f"  Safety steering: {receipt.orthogonalized_utility:.4f}")
print(f"  Final score: {receipt.final_score:.4f}")
print(f"  Constrained: {receipt.was_constrained}")
print(f"  Reason: {receipt.primary_reason}")
```

## Install

```bash
pip install governed-rank
```

---

*From the governed-rank validation suite. [Source](https://github.com/rdoku/governed-rank)*
