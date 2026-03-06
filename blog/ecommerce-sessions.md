# E-commerce Session Reranking on Yoochoose — Displacement Monotonicity

Yoochoose is a large e-commerce click dataset: millions of click sessions with product interactions. We used it to validate a specific property of governed-rank: **displacement monotonicity** — as you increase the budget, fewer items get displaced from their original positions.

## The Property

For any ranking system with a "steering intensity" knob, you want a guarantee: turning the knob toward "less steering" should never accidentally increase disruption. Without this, your system is unpredictable — you can't tell product managers "budget=0.5 is safer than budget=0.3."

## The Setup

```python
from mosaic import MOSAICScorer, MOSAICConfig
import numpy as np

# Yoochoose: ~40K sessions, 3-50 items per session
# Item-CF trained on first 30K baskets
# Category-based moment affinities: A[i, c] = 1 if item i has category c

scorer = MOSAICScorer(
    moment_affinities=A,
    config=MOSAICConfig(
        lambda_m=0.5,
        protection_mode="budget",
        budget_pct=0.30,     # will sweep this
        fallback_gap_threshold=0.01,
    ),
)
```

## Sweeping Budget

```python
budgets = [0.0, 0.10, 0.30, 0.50, 1.00]

for budget in budgets:
    config = MOSAICConfig(
        lambda_m=0.5,
        protection_mode="budget",
        budget_pct=budget,
        fallback_gap_threshold=0.01,
    )
    scorer = MOSAICScorer(moment_affinities=A, config=config)

    displacements = []
    stabilities = []

    for basket in test_baskets[:300]:
        cart, holdout, candidates = split_basket(basket)
        base_scores = item_cf_score(cart, candidates)
        result = scorer.rank(candidates, base_scores, activation_p, "high")

        # Displacement: items in base top-10 but not MOSAIC top-10
        base_top10 = set(sorted(base_scores, key=base_scores.get, reverse=True)[:10])
        mosaic_top10 = set(result.ranked_items[:10])
        displaced = len(base_top10 - mosaic_top10)
        displacements.append(displaced)

        # Stability: fraction of base top-10 preserved
        stability = len(base_top10 & mosaic_top10) / 10
        stabilities.append(stability)

    print(f"Budget {budget:.2f}: "
          f"mean_displaced={np.mean(displacements):.2f}, "
          f"mean_stability={np.mean(stabilities):.3f}")
```

## Results

| Budget | Mean Displaced | Mean Stability |
|--------|---------------|----------------|
| 0.00   | ~5-6          | ~0.45          |
| 0.10   | ~4-5          | ~0.55          |
| 0.30   | ~2-3          | ~0.75          |
| 0.50   | ~1-2          | ~0.85          |
| 1.00   | 0             | 1.00           |

**Key findings:**
- Displacement is **monotonically decreasing** as budget increases
- Stability is **monotonically increasing** as budget increases
- Recall variance across budgets: < 2pp

This holds because the budget parameter maps directly to the number of protected edges in the isotonic projection. More protected edges = fewer allowed reorderings = less displacement.

## The Simple Version

For quick displacement control without moments:

```python
from mosaic import govern

result = govern(
    base_scores=product_scores,
    steering_scores=category_boosts,
    budget=0.3,   # guaranteed: fewer displaced items than budget=0.1
)

print(f"Protected edges: {result.n_protected_edges}")
print(f"Active constraints: {result.n_active_constraints}")
```

## Why This Matters for Product Teams

Product managers need predictable controls. "We'll steer 30% toward promoted products" is actionable. "We'll add a weight of 0.3 to the policy signal" is not — because the effect of that weight depends on the correlation structure of your signals.

Budget gives you a guarantee about the maximum disruption, not just a knob that might or might not do what you expect.

## Install

```bash
pip install governed-rank
```

---

*From the governed-rank validation suite. [Source](https://github.com/rdoku/governed-rank)*
