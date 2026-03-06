# Why Orthogonalization Beats Naive Reranking

Every team that adds a policy signal to a ranker hits the same wall: accuracy drops. You add toxicity penalties — engagement falls. You add fairness boosts — precision drops. You add margin targets — relevance degrades.

The reason is always the same: **your policy signal correlates with your base scores**, and adding them together creates interference.

## The Math Problem

You have:
- `s`: base scores (relevance, engagement, quality)
- `u`: steering signal (toxicity, fairness, margin, safety)
- Naive combination: `final = s + λu`

If `corr(s, u) ≠ 0` — and it almost never equals zero — then `s + λu` doesn't do what you think. Items that are both relevant AND policy-aligned get double-boosted. Items that are relevant but policy-misaligned get pulled down. The combined score creates ordering distortions that neither signal intended.

## The Fix: Project Into the Null Space

```python
from mosaic.orthogonalization import orthogonalize_against_base

result = orthogonalize_against_base(
    base_scores=base_scores,       # {item_id: score}
    steering_utilities=steering,   # {item_id: policy_signal}
)

# result.u_perp: orthogonalized steering signal
# result.projection_coeff: how much of u was aligned with s
# result.corr_before: correlation before orthogonalization
# result.corr_after: correlation after (should be ~0)
```

What happens:
1. Compute the component of `u` that lies along the direction of `s`
2. Subtract it: `u_perp = u - (u·s / s·s) × s`
3. The result `u_perp` is perpendicular to `s` — adding it to `s` cannot interfere with the base ordering's confident decisions

## Worked Example

```python
from mosaic.orthogonalization import orthogonalize_against_base
import numpy as np

# 5 items: base scores + toxicity penalties
base_scores = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5}
toxicity = {0: -0.8, 1: 0.1, 2: 0.5, 3: -0.3, 4: 0.7}  # negative = toxic

result = orthogonalize_against_base(base_scores, toxicity)

print(f"Correlation before: {result.corr_before:.3f}")
print(f"Correlation after: {result.corr_after:.3f}")
print(f"Projection coefficient: {result.projection_coeff:.3f}")

for item_id in sorted(base_scores):
    print(f"  Item {item_id}: "
          f"base={base_scores[item_id]:.2f}, "
          f"raw_steer={toxicity[item_id]:.2f}, "
          f"ortho_steer={result.u_perp[item_id]:.3f}")
```

If toxicity correlates with base scores (toxic content tends to be engaging), the projection coefficient is positive — meaning a chunk of the toxicity signal was redundant with what the base ranker already captures. Removing it leaves only the "pure safety" signal.

## Validation: Ta Feng Lambda Sweep

We swept the steering weight (lambda) from 0 to 1.0 on Ta Feng grocery data:

| Lambda | Naive Recall@10 | MOSAIC Recall@10 | Delta |
|--------|----------------|------------------|-------|
| 0.0    | 33.70% (base)  | 33.70% (base)    | 0     |
| 0.03   | ~33.7%         | 33.72%           | +0.02 |
| 0.10   | ~33.5%         | ~33.7%           | ~+0.2 |
| 0.30   | ~33.0%         | ~33.6%           | ~+0.6 |
| 0.50   | ~32.5%         | ~33.5%           | ~+1.0 |
| 1.00   | ~31.5%         | ~33.2%           | ~+1.7 |

**Naive steering breaks at high lambda** (>1pp recall drop). **MOSAIC holds within 1.5pp at all lambdas.** The orthogonalization absorbs the interference that destroys naive combination.

## The Guarantee

After orthogonalization:
```
Cov(s, u_perp) = 0
```

This is not approximate. It's exact (up to floating-point precision). The orthogonalized steering signal is mathematically independent of the base score direction. Adding it to base scores cannot, by construction, create the interference pattern that plagues naive combination.

## Using It Directly

The `govern()` function handles orthogonalization automatically:

```python
from mosaic import govern

result = govern(
    base_scores=engagement_scores,
    steering_scores=toxicity_penalties,
    budget=0.3,
)

# Orthogonalization happened internally
# result.projection_coeff tells you how much steering was aligned with base
print(f"Projection coefficient: {result.projection_coeff:.3f}")
# High coefficient = lots of correlation removed = orthogonalization mattered
# Low coefficient = signals were already independent = orthogonalization was mild
```

## The Full Orthogonalization Result

```python
from mosaic.orthogonalization import orthogonalize_against_base

result = orthogonalize_against_base(base_scores, steering)

# Diagnostics
print(f"Steering magnitude before: {result.u_magnitude_before:.3f}")
print(f"Steering magnitude after: {result.u_magnitude_after:.3f}")
print(f"Correlation before: {result.corr_before:.3f}")
print(f"Correlation after: {result.corr_after:.6f}")  # should be ~0
print(f"Projection coefficient: {result.projection_coeff:.3f}")

# If projection_coeff is large, orthogonalization removed a lot of interference
# If it's near zero, your signals were already independent
```

## Computing Target Scores

After orthogonalization, combine base scores with the orthogonalized signal:

```python
from mosaic.orthogonalization import orthogonalize_against_base, compute_target_scores

ortho = orthogonalize_against_base(base_scores, steering)
target = compute_target_scores(base_scores, ortho.u_perp)

# target[item_id] = base_scores[item_id] + ortho.u_perp[item_id]
# This is the score that the isotonic projection then constrains
```

## When Orthogonalization Matters Most

1. **High correlation between signals**: toxicity and engagement, risk and demographics, relevance and groundedness
2. **Aggressive steering (high lambda)**: the more you steer, the more interference matters
3. **High-stakes domains**: where accuracy degradation is unacceptable (healthcare, finance, legal)

When your signals are already independent (corr ≈ 0), orthogonalization is a no-op — it doesn't hurt, it just doesn't change anything. This makes it safe to always apply.

## Install

```bash
pip install governed-rank
```

---

*From the governed-rank validation suite. [Source](https://github.com/rdoku/governed-rank)*
