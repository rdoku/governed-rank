# The Budget Knob: One Parameter to Control Accuracy vs Policy

Every reranking system needs a control surface — a way to tell the system "steer this much, no more." Most systems use a weight parameter (lambda) that controls how much the policy signal contributes. The problem: lambda's effect is unpredictable. The same lambda value produces wildly different results depending on the correlation structure between your base scores and policy signal.

`governed-rank` uses a **budget** parameter instead. Budget directly controls what fraction of ordering decisions are locked. It's interpretable, monotonic, and predictable.

## Budget vs Lambda

| Property | Lambda (weight) | Budget (fraction) |
|----------|----------------|-------------------|
| What it controls | Signal contribution | Locked ordering decisions |
| Interpretability | "Add 0.3 of the policy signal" (unclear effect) | "Lock 30% of ordering decisions" (clear) |
| Monotonicity | Not guaranteed — higher lambda can sometimes improve accuracy | **Guaranteed** — higher budget = less disruption |
| Predictability | Depends on signal correlation | Independent of signal correlation |
| Domain-specific tuning | Yes — different lambdas needed per domain | Less — budget=0.3 works across domains |

## How Budget Works

```python
from mosaic import govern

result = govern(
    base_scores=scores,
    steering_scores=policy,
    budget=0.3,   # lock the top 30% of score gaps
)

print(f"Protected edges: {result.n_protected_edges}")
print(f"Active constraints: {result.n_active_constraints}")
```

Budget=0.3 means: sort all adjacent-pair score gaps by size, lock the top 30%. These pairs cannot be flipped by steering. The remaining 70% of pairs are open to reordering.

## The Monotonicity Guarantee

We validated across 300 Ta Feng baskets:

```python
from mosaic import MOSAICScorer, MOSAICConfig
import numpy as np

budgets = [0.0, 0.10, 0.30, 0.50, 0.70, 1.00]

for budget in budgets:
    scorer = MOSAICScorer(
        moment_affinities=A,
        config=MOSAICConfig(
            lambda_m=0.5,
            protection_mode="budget",
            budget_pct=budget,
        ),
    )
    result = scorer.rank(candidates, base_scores, activation_p, "high")

    base_top10 = set(sorted(base_scores, key=base_scores.get, reverse=True)[:10])
    mosaic_top10 = set(result.ranked_items[:10])
    displaced = len(base_top10 - mosaic_top10)
    stability = len(base_top10 & mosaic_top10) / 10

    print(f"Budget {budget:.2f}: displaced={displaced}, stability={stability:.2f}")
```

**Results**: Displacement monotonically decreases. Stability monotonically increases. Tested across 5 random seeds, all with Spearman rho >= 0.8 between budget and stability.

## Budget Guidance by Domain

| Domain | Recommended Budget | Why |
|--------|-------------------|-----|
| Content moderation | 0.3 | Balanced — most engagement decisions preserved |
| Fairness (hiring, lending) | 0.5 | Conservative — protect quality model |
| RAG safety | 0.5-0.7 | Retrieval relevance is critical |
| Healthcare | 0.7 | Clinical ordering must dominate |
| Product discovery | 0.1-0.3 | More room for exploration |
| Ad insertion | N/A | LENS uses confidence thresholds instead |
| Fraud detection | 0.5 | Protect fraud model, steer review queue |

## The Pareto Curve

At each budget level, you get a point on the stability-vs-policy-exposure curve:

```
Budget 0.0: max policy exposure, min stability
Budget 0.3: 0.890 stability, 0.344 policy exposure  ← sweet spot
Budget 0.5: higher stability, less policy exposure
Budget 1.0: max stability (100%), zero policy effect
```

The budget=0.3 point achieves **89% stability at 34.4% policy exposure**. This dominates the naive weighted combination at every point on the Pareto frontier.

## Using Budget with get_protected_edges_by_budget

For direct access to the protection mechanism:

```python
from mosaic.gap_calibration import get_protected_edges_by_budget

# Get which edges are protected at budget=0.30
protected = get_protected_edges_by_budget(
    base_order=sorted_item_indices,    # items sorted by base score descending
    base_scores=base_scores,           # {item_id: score}
    budget_pct=0.30,
    max_rank=50,                       # only consider top-50
)
# protected: list of edge indices that are locked
```

## Auto-Budget: Targeting Stability

Don't want to pick a budget? Let MOSAIC find the right one:

```python
scorer = MOSAICScorer(
    moment_affinities=A,
    config=MOSAICConfig(
        lambda_m=0.5,
        target_stability=0.95,            # "I want 95% stability"
        auto_budget_steps=(0.10, 0.30, 0.50, 0.70, 1.00),
        auto_budget_max_passes=2,
        budget_pct=0.10,                  # start low
    ),
)

result = scorer.rank(candidates, base_scores, activation_p, "high")

print(f"Budget used: {result.budget_used:.2f}")
print(f"Stability achieved: {result.achieved_stability:.3f}")
print(f"Auto-budget passes: {result.auto_budget_passes}")
```

Auto-budget starts at the lowest step and escalates until the target stability is reached or max passes is hit. It's a search over the budget dimension — each step is cheap because the core algorithm is O(N log N).

## VaR/CVaR Risk Bounds

For risk-sensitive applications, you can bound the tail of displacement:

```python
scorer = MOSAICScorer(
    moment_affinities=A,
    config=MOSAICConfig(
        lambda_m=0.5,
        budget_pct=0.30,
        risk_var_quantile=0.95,    # track 95th percentile displacement
        risk_cvar_quantile=0.95,   # track conditional tail mean
    ),
)
```

We validated on Ta Feng: test VaR95 <= train VaR95 + 0.5 at all budget levels. The budget parameter controls not just mean displacement but tail risk.

## Install

```bash
pip install governed-rank
```

---

*From the governed-rank validation suite. [Source](https://github.com/rdoku/governed-rank)*
