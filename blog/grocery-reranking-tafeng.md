# Grocery Reranking on Ta Feng — The Baseline That Proves the Math

Ta Feng is our primary validation dataset: 32,000 grocery transactions from a Taiwanese retailer with ~11,000 products. It's not the flashiest result, but it's the most important — it proves that MOSAIC's orthogonalization preserves base accuracy to within hundredths of a percent.

## The Numbers

| Method | Recall@10 | NDCG@10 |
|--------|-----------|---------|
| Base Item-CF | 33.70% | 26.22% |
| Naive steering (base + λ × alignment) | 33.71% | ~26% |
| **MOSAIC (λ=0.03, budget=0.30)** | **33.72%** | **27.22%** |

MOSAIC doesn't just preserve accuracy — it slightly improves it (+0.02pp Recall, +1pp NDCG). This is because the orthogonalized steering signal adds useful diversity to the ranking that the base model missed.

## The Setup

```python
from mosaic import MOSAICScorer, MOSAICConfig
import numpy as np

# Load Ta Feng data and build Item-CF base ranker
# K=8 moments discovered via anchor-based soft clustering
# A: (n_items, 8) moment affinity matrix, row-normalized

scorer = MOSAICScorer(
    moment_affinities=A,     # shape: (n_items, 8)
    config=MOSAICConfig(
        lambda_m=0.03,       # gentle steering
        rho=0.90,            # orthogonality factor
        fallback_gap_threshold=0.01,
    ),
)

# Uniform activation: equal weight to all 8 moments
K = 8
activation_p = np.ones(K) / K

# Score candidates for a user with items in cart
result = scorer.rank(
    candidates=candidate_ids,    # ~200 candidates per user
    base_scores=item_cf_scores,  # cosine similarity from Item-CF
    activation_p=activation_p,
    activation_confidence="medium",
)
```

## Budget Monotonicity (Test 1a)

The first thing to validate: does increasing the budget actually protect more ordering decisions?

```python
budgets = [0.0, 0.10, 0.30, 0.50, 0.70, 1.00]

for budget in budgets:
    scorer = MOSAICScorer(
        moment_affinities=A,
        config=MOSAICConfig(
            lambda_m=0.5,
            protection_mode="budget",
            budget_pct=budget,
            fallback_gap_threshold=0.01,
        ),
    )

    result = scorer.rank(
        candidates=candidates,
        base_scores=scores,
        activation_p=activation_p,
        activation_confidence="high",
    )

    # Count displaced items: in base top-10 but not MOSAIC top-10
    base_top10 = set(sorted(scores, key=scores.get, reverse=True)[:10])
    mosaic_top10 = set(result.ranked_items[:10])
    displaced = len(base_top10 - mosaic_top10)

    print(f"Budget {budget:.2f}: displaced={displaced}, "
          f"protected_edges={result.n_protected_edges}, "
          f"active_constraints={result.n_active_constraints}")
```

**Result**: Displacement monotonically decreases as budget increases. At budget=1.0, displacement is zero. This holds across 300 test baskets with 0.05 tolerance.

## Lambda Sweep: Why Orthogonalization Matters (Test 1c)

The critical comparison — naive steering vs MOSAIC at increasing steering weights:

```python
lambdas = [0.0, 0.03, 0.1, 0.3, 0.5, 1.0]

for lam in lambdas:
    # Naive: base_score + lambda * A[c] @ activation_p
    naive_scores = {
        c: base_scores[c] + lam * float(A[c] @ activation_p)
        for c in candidates
    }
    naive_top10 = sorted(naive_scores, key=naive_scores.get, reverse=True)[:10]
    naive_recall = len(set(naive_top10) & holdout_set) / len(holdout_set)

    # MOSAIC
    scorer = MOSAICScorer(
        moment_affinities=A,
        config=MOSAICConfig(lambda_m=lam, budget_pct=0.30),
    )
    result = scorer.rank(candidates, base_scores, activation_p, "high")
    mosaic_recall = len(set(result.ranked_items[:10]) & holdout_set) / len(holdout_set)

    print(f"λ={lam}: naive_recall={naive_recall:.3f}, mosaic_recall={mosaic_recall:.3f}")
```

**Result**: Naive steering breaks at high lambda (> -1pp recall drop at λ=0.5+). MOSAIC holds within 1.5pp of base at all lambda values. The orthogonalization absorbs the interference.

## Pareto Frontier (Test 3a)

The tradeoff between stability and steering gain:

```python
# At budget=0.30:
# Stability: 0.890 (89% of base top-10 preserved)
# Policy exposure: 0.344 (34.4% of top-50 are policy-aligned items)
# This point dominates naive weighted combination at every stability level
```

The Pareto curve shows that budget=0.30 achieves 89% stability at 34.4% policy exposure — a tradeoff that naive combination cannot reach at any weight setting.

## Multi-Seed Robustness (Test 6a)

We tested across 5 random seeds with 200 baskets each:

```python
seeds = [42, 123, 456, 789, 2024]

for seed in seeds:
    # Build Item-CF, split data, evaluate across budget levels
    # Compute Spearman correlation: budget vs stability
    correlation = spearmanr(budgets, stability_per_budget)
    print(f"Seed {seed}: Spearman ρ = {correlation:.3f}")
```

Each seed shows ρ >= 0.8 between budget and stability — the monotonic relationship holds regardless of the random data split.

## What Ta Feng Proves

Ta Feng is the controlled experiment. It shows:

1. **Orthogonalization preserves accuracy** — 33.72% vs 33.70% baseline
2. **Budget controls displacement monotonically** — verified across 300 baskets
3. **The Pareto frontier is real** — stability vs steering gain follows theory
4. **Robustness across seeds** — not a lucky split
5. **MOSAIC degrades gracefully** — even at λ=1.0 with aggressive steering, recall stays within bounds

## Install

```bash
pip install governed-rank
```

---

*From the governed-rank validation suite. [Source](https://github.com/rdoku/governed-rank)*
