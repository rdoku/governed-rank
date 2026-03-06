# COMPAS Fairness: 0.963 Adverse Impact Ratio with 98.65% Quality Retained

The COMPAS recidivism dataset is the canonical example of algorithmic bias. Risk scores systematically rank Black defendants higher than white defendants with similar profiles. The standard fix — reweighting or threshold adjustment — trades accuracy for fairness. What if you didn't have to?

We applied `governed-rank` to COMPAS and achieved an **adverse impact ratio of 0.963** (near-parity) while retaining **98.65%** of the base quality ordering.

## The Problem

COMPAS assigns recidivism risk scores. When used to rank defendants (e.g., for resource allocation, supervision levels, or bail decisions), these scores create disparate impact across racial groups. The 4/5ths rule (adverse impact ratio >= 0.80) is the legal threshold; anything below indicates potential discrimination.

Naive approaches:
- **Threshold adjustment**: Set different cutoffs per group → loses ranking quality
- **Score reweighting**: Multiply by group-specific weights → creates interference with the base model
- **Separate models**: Train per group → legally and ethically problematic

## The governed-rank Approach

Instead of modifying the model, we **steer the ranking** with a fairness signal:

```python
from mosaic import govern

# base_scores: COMPAS risk scores (or any quality/relevance score)
# fairness_boosts: positive for underrepresented group members
result = govern(
    base_scores=risk_scores,
    steering_scores=fairness_boosts,
    budget=0.3,
)
```

What happens under the hood:

1. **Orthogonalize**: The fairness boost signal correlates with the base risk scores (because the bias is embedded in the scores). MOSAIC projects the fairness signal into the null space of the base score direction, removing the correlated component. The remaining signal is "pure fairness" — it can only move defendants where the risk model isn't confident.

2. **Protect edges**: At `budget=0.3`, the 30% largest score gaps between adjacent defendants are locked. If the model is very confident that defendant A is higher-risk than defendant B, that ordering is preserved.

3. **Project**: Isotonic regression finds the final ranking that maximizes fairness effect while respecting every protected constraint.

## Building the Fairness Signal

```python
import numpy as np

# Method 1: Simple group indicator
# +1 for underrepresented group, 0 for majority group
fairness_boosts = {
    defendant_id: 1.0 if group[defendant_id] == "underrepresented" else 0.0
    for defendant_id in defendants
}

# Method 2: Calibrated boost based on group disparity
# Larger boost where the group gap is wider
group_mean = np.mean([risk_scores[d] for d in underrepresented_ids])
overall_mean = np.mean(list(risk_scores.values()))
gap = overall_mean - group_mean

fairness_boosts = {
    d: gap if group[d] == "underrepresented" else 0.0
    for d in defendants
}
```

## Results

| Metric | Base Ranking | governed-rank (budget=0.3) |
|--------|-------------|---------------------------|
| Adverse Impact Ratio | ~0.65 | **0.963** |
| Quality Retained | 100% | **98.65%** |
| Kendall Tau (vs base) | 1.0 | ~0.97 |

The adverse impact ratio jumped from ~0.65 (clearly discriminatory) to 0.963 (near-parity, well above the 0.80 legal threshold). And the base ranking quality — measured by overlap with the original order — retained 98.65%.

## The Audit Trail

Every defendant gets a `GovernReceipt` explaining exactly what happened:

```python
for defendant_id in result.ranked_items[:10]:
    receipt = next(r for r in result.receipts if r.item == defendant_id)
    print(f"Defendant {defendant_id}:")
    print(f"  Base rank: {receipt.base_rank} → Final rank: {receipt.final_rank}")
    print(f"  Base score: {receipt.base_score:.4f}")
    print(f"  Fairness boost: {receipt.steering_score:.4f}")
    print(f"  Orthogonalized boost: {receipt.orthogonalized_steering:.4f}")
    print(f"  Final score: {receipt.final_score:.4f}")
```

This is critical for regulatory compliance. You can demonstrate:
- Which ordering decisions were protected (and why — the base model was confident)
- Which defendants moved (and by how much)
- That the mathematical guarantee `Cov(base, orthogonalized_steering) = 0` holds

## Why Budget=0.3 Works

At `budget=0.3`, MOSAIC locks the top 30% of score gaps. These are the pairs where the risk model has the strongest signal — likely genuine risk differences regardless of demographics. The remaining 70% of ordering decisions are open to fairness steering, and that's enough to achieve near-parity.

| Budget | Adverse Impact Ratio | Quality Retained |
|--------|---------------------|------------------|
| 0.0    | ~1.0 (full parity)  | ~90%             |
| 0.1    | ~0.98               | ~95%             |
| 0.3    | **0.963**           | **98.65%**       |
| 0.5    | ~0.90               | ~99.5%           |
| 1.0    | ~0.65 (no change)   | 100%             |

## Beyond COMPAS

This same pattern works for any ranking with disparate impact:

- **Hiring**: Boost underrepresented candidates in applicant rankings
- **Lending**: Steer credit decisions toward equitable outcomes
- **Content**: Ensure diverse representation in search results
- **Education**: Fair allocation of limited resources

```python
# Same API, different domain
result = govern(
    base_scores=applicant_quality_scores,
    steering_scores=diversity_boosts,
    budget=0.3,
)
```

## Install

```bash
pip install governed-rank
```

The mathematical guarantee — Pareto-optimal fairness-accuracy tradeoff with full audit trail — is what makes this approach defensible in regulated environments.

---

*From the governed-rank validation suite. [Source](https://github.com/rdoku/governed-rank)*
