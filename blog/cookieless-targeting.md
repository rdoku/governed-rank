# Cookieless Targeting: 4.65x CVR Lift with governed-rank

Third-party cookies are dying. Without them, ad targeting falls back to contextual signals — page content, time of day, broad demographics. But contextual targeting is noisy: it correlates with user intent in messy ways.

We applied `governed-rank` to RetailRocket and Criteo datasets and achieved **4.65x conversion rate (CVR) lift** by steering contextual recommendations toward purchase-likely items without breaking the base relevance model.

## The Problem

Post-cookie targeting relies on:
- **Contextual signals**: what page the user is on, search terms, content category
- **First-party data**: purchase history, on-site behavior
- **Aggregate models**: collaborative filtering on anonymized data

These signals are weaker than cookie-based profiles. When you add a "purchase likelihood" steering signal to the base relevance model, the signals interfere because contextually relevant items are often (but not always) purchase-likely.

## The Fix

```python
from mosaic import govern

# base_scores: relevance from contextual/first-party model
# purchase_intent: predicted purchase likelihood (contextual feature)
result = govern(
    base_scores=relevance_scores,
    steering_scores=purchase_intent_scores,
    budget=0.3,
)
```

The orthogonalization step is critical here: it removes the component of purchase intent that already correlates with contextual relevance. The remaining signal pushes purchase-likely items up only where the relevance model is uncertain — exactly the gap where cookie-based targeting used to help.

## Building Purchase Intent Signals

### From Behavioral Features

```python
# Category purchase rate from aggregate first-party data
category_purchase_rates = compute_category_cvr(transaction_log)

purchase_intent = {
    item_id: category_purchase_rates.get(item_category[item_id], 0.0)
    for item_id in candidate_items
}
```

### From Session Context

```python
# Items similar to what the user just viewed/carted
session_items = get_session_items(session_id)
purchase_intent = {
    item_id: cosine_similarity(
        item_embedding[item_id],
        mean_embedding(session_items)
    )
    for item_id in candidate_items
}
```

### From Moment Discovery

```python
from mosaic.discovery import DiscoveryEngine, DiscoveryConfig

engine = DiscoveryEngine(config=DiscoveryConfig(
    min_lift=1.5,
    min_reads=100,
))

# Discover which categories convert above base rate
report = engine.discover(
    sessions=anonymous_sessions,
    catalog=product_catalog,
)

# Use discovered high-conversion categories as steering signal
high_cvr_categories = {
    d.category for d in report.opportunities
    if d.preference_lift > 2.0
}

purchase_intent = {
    item_id: 1.0 if item_category[item_id] in high_cvr_categories else 0.0
    for item_id in candidate_items
}
```

## Full Pipeline with MOSAICScorer

For production cookieless targeting with category moments:

```python
from mosaic import MOSAICScorer, MOSAICConfig
import numpy as np

# Category affinities
A = build_category_affinities(items, categories)

scorer = MOSAICScorer(
    moment_affinities=A,
    config=MOSAICConfig(
        lambda_m=0.5,
        protection_mode="budget",
        budget_pct=0.30,
    ),
)

# Activation from session context
# e.g., user is browsing electronics → activate electronics moment
activation_p = compute_session_activation(session_context, categories)

result = scorer.rank(
    candidates=candidate_item_ids,
    base_scores=relevance_scores,
    activation_p=activation_p,
    activation_confidence="medium",
)
```

## Results

Validation on RetailRocket and Criteo:

| Metric | Base Model | governed-rank (budget=0.3) |
|--------|-----------|---------------------------|
| CVR Lift | 1.0x | **4.65x** |
| Relevance stability | 100% | >89% |

The 4.65x CVR lift means that items surfaced by governed-rank are 4.65 times more likely to convert than the base model's top results. This happens because orthogonalization isolates the "pure purchase intent" signal — the part that's independent of what the base relevance model already captures.

## Why This Replaces Cookies (Partially)

Cookies gave you a direct signal: "this user bought shoes last week, show them shoes." Without cookies, you're guessing from context. Governed-rank's contribution is:

1. **Separating intent from relevance**: The orthogonalization step ensures purchase intent steering doesn't degrade contextual relevance
2. **Budget protection**: High-confidence relevance decisions (user searched for "laptop" → show laptops) are locked
3. **Filling the gap**: The uncertain middle of the ranking — where cookie data used to help — is where purchase intent steering operates

This isn't a full replacement for behavioral targeting. But it recovers meaningful conversion lift from first-party and contextual signals that would otherwise be lost to interference.

## Install

```bash
pip install governed-rank
```

---

*From the governed-rank validation suite. [Source](https://github.com/rdoku/governed-rank)*
