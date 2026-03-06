# Gap Calibration: Learning Confidence from Your Data

The budget parameter protects ordering decisions by locking the largest score gaps. But score gaps aren't confidence: a gap of 0.1 might be highly confident in one domain and meaningless in another, depending on your base model's calibration.

Gap calibration learns a mapping from score gaps to actual confidence: P(the higher-scored item is truly better | gap). This makes protection decisions data-driven rather than heuristic.

## The Problem

Without calibration, MOSAIC uses a fallback: any gap above a threshold (default 0.01) gets the same treatment. This works but it's crude — a gap of 0.5 and a gap of 0.02 get treated similarly within the protected set.

With calibration, MOSAIC knows: "a gap of 0.5 means 95% confidence the ordering is correct" vs "a gap of 0.02 means 52% confidence (basically a coin flip)."

## Training Calibration

```python
from mosaic.gap_calibration import (
    learn_gap_calibration,
    CalibrationConfig,
    extract_pairs_adjacent_boundary,
    CalibrationResult,
)

# Step 1: Extract (gap, correct) pairs from training data
pairs = extract_pairs_adjacent_boundary(
    baskets=training_baskets,      # list of item lists (purchase baskets)
    base_scorer=item_cf_score,     # function(cart, candidates) → {id: score}
    n_items=n_items,
    n_baskets=2000,                # sample 2000 baskets
    max_rank=50,                   # only consider top-50
    rng_seed=42,
)
# pairs: [(gap, correct), ...] where correct = True if higher-scored item was purchased

print(f"Extracted {len(pairs)} training pairs")

# Step 2: Learn the calibration curve
calibration = learn_gap_calibration(
    pairs,
    config=CalibrationConfig(n_buckets=15),
)

# Step 3: Inspect the learned curve
for i in range(len(calibration.bucket_edges) - 1):
    lo = calibration.bucket_edges[i]
    hi = calibration.bucket_edges[i + 1]
    conf = calibration.bucket_confidences[i]
    n = calibration.samples_per_bucket[i]
    print(f"Gap [{lo:.3f}, {hi:.3f}): confidence={conf:.3f}, n={n}")
```

## Using Calibration

```python
from mosaic import MOSAICScorer, MOSAICConfig

# With calibration: protection is data-driven
scorer_calibrated = MOSAICScorer(
    moment_affinities=A,
    calibration=calibration,         # learned calibration
    config=MOSAICConfig(
        lambda_m=0.5,
        use_calibration=True,
        rho=0.90,                    # protect edges with P(correct) > 0.90
    ),
)

# Without calibration: protection uses fallback threshold
scorer_uncalibrated = MOSAICScorer(
    moment_affinities=A,
    config=MOSAICConfig(
        lambda_m=0.5,
        use_calibration=False,
        fallback_gap_threshold=0.01,  # protect edges with gap > 0.01
    ),
)
```

## The Calibration API

### CalibrationResult

```python
# Query confidence for a specific gap
confidence = calibration.gap_to_conf(0.05)
print(f"Gap=0.05 → confidence={confidence:.3f}")

# Query gap for a target confidence
gap = calibration.conf_to_gap(0.90)
print(f"Confidence=0.90 → gap={gap:.3f}")

# Save/load for production
calibration.save("models/gap_calibration.json")
loaded = CalibrationResult.load("models/gap_calibration.json")
```

### Pair Extraction Methods

Two methods for extracting training pairs:

```python
# Method 1: Adjacent boundary pairs
# Compare item at rank k vs item at rank k+1
# "correct" = higher-ranked item was actually in the basket
pairs_adjacent = extract_pairs_adjacent_boundary(
    baskets, base_scorer, n_items,
    n_baskets=2000, max_rank=50,
)

# Method 2: Positive-negative pairs
# Compare purchased items vs random non-purchased items
from mosaic.gap_calibration import extract_pairs_pos_neg

pairs_posneg = extract_pairs_pos_neg(
    baskets, base_scorer, n_items,
    n_baskets=5000, neg_per_pos=10, max_rank=50,
)
```

Adjacent boundary pairs are preferred for calibration because they reflect the actual ordering decisions MOSAIC needs to protect.

## Getting Protected Edges

```python
from mosaic.gap_calibration import get_protected_edges, get_protected_edges_by_budget

# Method 1: Threshold-based (with calibration)
protected = get_protected_edges(
    base_order=sorted_items,
    base_scores=scores,
    calibration=calibration,
    rho=0.90,                # protect where P(correct) > 0.90
)

# Method 2: Budget-based (with or without calibration)
protected = get_protected_edges_by_budget(
    base_order=sorted_items,
    base_scores=scores,
    calibration=calibration,  # optional — improves edge selection
    budget_pct=0.30,
    max_rank=50,
)
```

## Validation Results

We tested calibrated vs uncalibrated on Ta Feng across multiple lambda values:

| Lambda | Calibrated Stability | Uncalibrated Stability | Winner |
|--------|---------------------|----------------------|--------|
| 0.03   | ~0.97               | ~0.97                | Tie    |
| 0.10   | ~0.95               | ~0.94                | Calibrated |
| 0.30   | ~0.90               | ~0.88                | Calibrated |
| 0.50   | ~0.85               | ~0.82                | Calibrated |

Calibrated protection matches or beats uncalibrated in >50% of lambda settings. The advantage grows with aggressive steering — exactly when you need it most.

Training statistics: 7,303 pairs extracted from 2,000 baskets. Calibration curve ranged from 0.49 confidence (near-random at small gaps) to 0.71 (moderately confident at large gaps).

## When to Use Calibration

**Always use it if you have training data.** Calibration never hurts — in the worst case it matches the fallback. The benefit grows with:

- Larger steering weight (lambda > 0.1)
- More heterogeneous score distributions
- Domains where score gaps have varying meaning

**Skip calibration when:**
- You don't have historical correctness labels
- You're prototyping and want the simplest possible setup
- You're using `govern()` (which uses budget-based protection)

## Install

```bash
pip install governed-rank
```

---

*From the governed-rank validation suite. [Source](https://github.com/rdoku/governed-rank)*
