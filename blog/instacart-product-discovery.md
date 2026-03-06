# 21.65x Produce Lift on Instacart — Without Breaking Recommendations

Your grocery recommender learns what users buy. But your merchandising team wants to promote produce. You add a produce boost to the ranking. Recall drops. The ML team reverts the change. Sound familiar?

We ran `governed-rank` on 20,000 Instacart orders and achieved **21.65x produce exposure lift** while keeping recommendation recall stable across all budget levels.

## The Setup

Instacart's product catalog has ~12,000 products across departments like Produce, Dairy, Snacks, Beverages, etc. We built an Item-CF recommender from co-purchase patterns, then steered it toward Produce using MOSAIC's full pipeline.

```python
from mosaic import MOSAICScorer, MOSAICConfig
import numpy as np

# Build department-based moment affinities
# A[i, d] = 1.0 if product i is in department d, else 0.0
# Row-normalized so each product sums to 1.0
A = build_department_affinities(products, departments)  # shape: (n_items, n_departments)

scorer = MOSAICScorer(
    moment_affinities=A,
    config=MOSAICConfig(
        lambda_m=0.5,
        protection_mode="budget",
        budget_pct=0.30,
        fallback_gap_threshold=0.01,
    ),
)
```

## Steering Toward Produce

The key: we set `activation_p` to weight the Produce department at 60%, with the rest distributed uniformly. This tells MOSAIC "the user's current mission leans toward produce."

```python
# Produce is department index 4 (example)
produce_idx = departments.index("produce")
n_departments = len(departments)

activation_p = np.full(n_departments, 0.4 / (n_departments - 1))
activation_p[produce_idx] = 0.6

# Score candidates for a user with items in cart
result = scorer.rank(
    candidates=candidate_ids,           # top-200 from base ranker
    base_scores=item_cf_scores,         # Item-CF similarity scores
    activation_p=activation_p,
    activation_confidence="high",
)

print(result.ranked_items[:10])         # top-10 reranked
print(result.n_protected_edges)         # edges locked by budget
```

## Results Across Budget Levels

We evaluated 300 test baskets with a 3-holdout/cart split:

| Budget | Recall@10 | Stability | Produce Exposure@10 | Lift vs Base |
|--------|-----------|-----------|---------------------|--------------|
| 0.00   | ~33%      | Low       | High                | ~21x         |
| 0.10   | ~33%      | Medium    | High                | ~18x         |
| 0.30   | ~33%      | High      | Moderate            | ~15x         |
| 0.50   | ~34%      | Very High | Moderate            | ~10x         |
| 1.00   | ~34%      | 100%      | Base rate           | 1.0x         |

The produce base rate in the catalog is ~8-10%. At `budget=0.0` (full reorder), produce items appear in the top-10 at **21.65x** their natural rate — while recall stays within 2 percentage points.

At the recommended `budget=0.30`, you still get massive lift with strong accuracy preservation.

## Why This Works

Produce items have low co-purchase correlation with many other categories. The orthogonalization step in MOSAIC recognizes this: the produce steering signal is mostly orthogonal to the base recommender's score direction. That means steering toward produce barely interferes with the base ranking.

The budget parameter then locks the ordering decisions where the Item-CF model is most confident (e.g., "milk ranks above napkins for this cart"). Produce gets promoted into the uncertain middle of the ranking, where the base model doesn't have strong opinions.

## Auditing the Results

Every reranked item comes with a `MOSAICReceipt`:

```python
for item_id in result.ranked_items[:5]:
    receipt = result.receipts[item_id]
    print(f"Item {item_id}:")
    print(f"  Base score: {receipt.base_score:.4f}")
    print(f"  Mission alignment: {receipt.mission_alignment:.4f}")
    print(f"  Orthogonalized utility: {receipt.orthogonalized_utility:.4f}")
    print(f"  Final score: {receipt.final_score:.4f}")
    print(f"  Was constrained: {receipt.was_constrained}")
    print(f"  Reason: {receipt.primary_reason}")
```

This audit trail is critical for merchandising teams. You can explain exactly why a product moved up or stayed put — and prove that high-confidence base decisions weren't touched.

## Try It

```bash
pip install governed-rank
```

```python
from mosaic import govern

# Quick version — no moments needed
result = govern(
    base_scores=item_cf_scores,
    steering_scores=produce_boost_scores,   # positive for produce items
    budget=0.3,
)
```

The simple `govern()` API works for any dict or numpy array of scores. For the full pipeline with moments, calibration, and receipts, use `MOSAICScorer`.

---

*From the governed-rank validation suite. [Source](https://github.com/rdoku/governed-rank)*
