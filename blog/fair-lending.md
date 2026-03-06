# Fair Lending with governed-rank — German Credit and Adult Income

Credit scoring models rank applicants by default risk. But these models inherit historical bias: zip codes correlate with race, education proxies for socioeconomic background, employment history reflects structural inequality. Regulatory frameworks (ECOA, Fair Housing Act) require that lending decisions don't produce disparate impact.

We validated `governed-rank` on the German Credit and Adult Income datasets — two canonical fairness benchmarks.

## The Approach

Instead of retraining the credit model (expensive, risky, legally complex), steer the ranking at decision time:

```python
from mosaic import govern

# credit_scores: P(default) or creditworthiness from your model
# fairness_signal: boost for applicants in protected groups
result = govern(
    base_scores=credit_scores,
    steering_scores=fairness_signal,
    budget=0.5,    # conservative — lending is high-stakes
)
```

## German Credit Dataset

1,000 applicants classified as good/bad credit risk with 20 features including age, job, housing, and credit history.

### Building the Fairness Signal

```python
import numpy as np

# Protected attribute: age (young applicants face disparate impact)
# Or: foreign worker status, gender proxy from personal_status
age_threshold = 25

fairness_signal = {
    applicant_id: 1.0 if age[applicant_id] < age_threshold else 0.0
    for applicant_id in applicants
}

result = govern(
    base_scores=credit_risk_scores,
    steering_scores=fairness_signal,
    budget=0.5,
)
```

### What Orthogonalization Does Here

Young applicants tend to have shorter credit histories, which correlates with higher predicted default risk. The naive approach (add an age boost) interferes with the credit model because age is already baked into the risk score.

MOSAIC's orthogonalization removes the component of the age-fairness signal that correlates with the credit score. The remaining signal only boosts young applicants in pairs where the credit model is uncertain — where age was the tiebreaker, not creditworthiness.

## Adult Income Dataset

48,842 records predicting whether income exceeds $50K, with features including education, occupation, race, sex, and hours per week.

### Multi-Attribute Fairness

Adult Income has multiple protected attributes. You can compose fairness signals:

```python
# Composite fairness: boost underrepresented on multiple axes
fairness_signal = {}
for person_id in candidates:
    boost = 0.0
    if race[person_id] in underrepresented_races:
        boost += 0.4
    if sex[person_id] == "Female":
        boost += 0.3
    if education_years[person_id] < 12:
        boost += 0.3    # structural disadvantage signal
    fairness_signal[person_id] = boost

result = govern(
    base_scores=income_prediction_scores,
    steering_scores=fairness_signal,
    budget=0.5,
)
```

### Measuring Disparate Impact

```python
# Compute adverse impact ratio after reranking
top_k = 100  # top-100 selected applicants
selected = set(result.ranked_items[:top_k])

group_a_rate = len(selected & group_a_ids) / len(group_a_ids)
group_b_rate = len(selected & group_b_ids) / len(group_b_ids)

adverse_impact_ratio = min(group_a_rate, group_b_rate) / max(group_a_rate, group_b_rate)
print(f"Adverse Impact Ratio: {adverse_impact_ratio:.3f}")
# Target: >= 0.80 (4/5ths rule)
```

## The Regulatory Argument

Fair lending regulators need to understand why decisions are made. MOSAIC provides:

1. **Mathematical guarantee**: `Cov(credit_score, orthogonalized_fairness) = 0` — the fairness adjustment is provably independent of the credit model's output

2. **Per-applicant receipts**: Every decision has an audit trail

```python
for applicant in result.ranked_items[:10]:
    receipt = next(r for r in result.receipts if r.item == applicant)
    print(f"Applicant {applicant}:")
    print(f"  Credit score: {receipt.base_score:.4f}")
    print(f"  Fairness signal: {receipt.steering_score:.4f}")
    print(f"  Orthogonalized: {receipt.orthogonalized_steering:.4f}")
    print(f"  Final rank: {receipt.final_rank} (was {receipt.base_rank})")
```

3. **Budget guarantee**: High-confidence credit decisions are never overridden. If the model is very confident applicant A is more creditworthy than applicant B, that ordering is preserved regardless of demographics.

## Budget Guidance for Lending

| Budget | Posture | When |
|--------|---------|------|
| 0.3    | Aggressive fairness | Remediating known disparate impact |
| 0.5    | Balanced | Standard fair lending compliance |
| 0.7    | Conservative | Minimal adjustment, monitoring phase |
| 0.9    | Observation | Measure potential impact without significant changes |

## The Key Insight

Fair lending doesn't require choosing between accuracy and fairness. Most of the disparate impact in credit models comes from ordering decisions where the model is uncertain — where the score gap between applicants is small and demographic proxies are the tiebreaker.

MOSAIC only steers in that uncertain zone. The credit model's confident decisions (clear creditworthiness differences) are untouched.

## Install

```bash
pip install governed-rank
```

---

*From the governed-rank validation suite. [Source](https://github.com/rdoku/governed-rank)*
