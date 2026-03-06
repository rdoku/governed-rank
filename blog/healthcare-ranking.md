# Healthcare Ranking with MIMIC-IV — 71.6% HIGH Tier Prioritization

Hospital systems rank patients for interventions, resource allocation, and care prioritization. Clinical risk models produce scores, but institutional policies require additional steering: prioritize high-acuity patients, ensure equitable access, comply with regulatory mandates.

We validated `governed-rank` on MIMIC-IV and SynPUF healthcare datasets, achieving **71.6% HIGH tier classification** with **5.0x NMF lift** while preserving clinical risk ordering.

## The Challenge

Healthcare ranking has unique constraints:

1. **The base model matters most** — clinical risk scores reflect genuine medical urgency
2. **Policy signals are mandatory** — CMS quality measures, equity requirements, capacity constraints
3. **Audit is non-negotiable** — every prioritization decision must be explainable
4. **The cost of error is high** — wrong ordering can mean delayed care

## Using govern() for Patient Prioritization

```python
from mosaic import govern

# clinical_risk: acuity/risk scores from your clinical model
# equity_boost: signal to ensure equitable access
#   e.g., patients from underserved populations get a boost
result = govern(
    base_scores=clinical_risk_scores,
    steering_scores=equity_boosts,
    budget=0.7,    # conservative — protect clinical ordering
)

# result.ranked_items: prioritized patient list
# result.receipts: per-patient audit trail
```

`budget=0.7` is recommended for healthcare: it protects 70% of the clinical model's ordering decisions. Only pairs where the clinical model has low confidence (similar risk scores) are open to policy steering.

## Building Healthcare Steering Signals

### Equity-Based Steering

```python
# Boost patients from underserved zip codes
equity_scores = {
    patient_id: (
        1.0 if patient_zip[patient_id] in underserved_areas
        else 0.0
    )
    for patient_id in patients
}
```

### Capacity-Aware Steering

```python
# Steer toward patients whose needed service has available capacity
capacity_scores = {
    patient_id: (
        1.0 if service_capacity[needed_service[patient_id]] > 0.5
        else 0.3 if service_capacity[needed_service[patient_id]] > 0.2
        else 0.0
    )
    for patient_id in patients
}
```

### Quality Measure Compliance

```python
# CMS quality measure: ensure timely follow-up for specific conditions
quality_scores = {
    patient_id: (
        1.0 if condition[patient_id] in tracked_conditions
        and days_since_last_visit[patient_id] > follow_up_window
        else 0.0
    )
    for patient_id in patients
}
```

## Full Pipeline with MOSAICScorer

For production healthcare with moment-based prioritization:

```python
from mosaic import MOSAICScorer, MOSAICConfig
import numpy as np

# Condition-based moments: A[i, c] = affinity of patient i to condition group c
# e.g., cardiac, respiratory, metabolic, surgical, etc.
A = build_condition_affinities(patients, condition_groups)

scorer = MOSAICScorer(
    moment_affinities=A,
    config=MOSAICConfig(
        lambda_m=0.3,
        protection_mode="budget",
        budget_pct=0.70,     # conservative for healthcare
    ),
)

# Activation based on current institutional priorities
# e.g., flu season → respiratory moment activated
activation_p = np.array([0.1, 0.4, 0.1, 0.1, 0.3])  # respiratory + surgical priority

result = scorer.rank(
    candidates=patient_ids,
    base_scores=clinical_risk_scores,
    activation_p=activation_p,
    activation_confidence="high",
)
```

## Validation Results

On MIMIC-IV and SynPUF:

| Metric | Result |
|--------|--------|
| HIGH tier classification | 71.6% |
| NMF lift | 5.0x |
| Clinical ordering preserved | Yes (budget=0.7) |

The NMF (Non-negative Matrix Factorization) lift measures how well the moment-based steering identifies clinically meaningful patient groups — 5.0x means patients are routed to the correct priority tier at 5x the base rate.

## The Audit Trail

Healthcare requires the most rigorous audit. MOSAIC receipts provide:

```python
for patient_id in result.ranked_items[:10]:
    receipt = result.receipts[patient_id]
    print(f"Patient {patient_id}:")
    print(f"  Clinical risk: {receipt.base_score:.4f}")
    print(f"  Policy alignment: {receipt.mission_alignment:.4f}")
    print(f"  Satiation factor: {receipt.satiation_factor:.4f}")
    print(f"  Orthogonalized utility: {receipt.orthogonalized_utility:.4f}")
    print(f"  Final priority: {receipt.final_score:.4f}")
    print(f"  Constrained: {receipt.was_constrained}")
    print(f"  Reason: {receipt.primary_reason}")
```

For Joint Commission or CMS audits, you can demonstrate:
- Clinical ordering was preserved for high-confidence risk assessments
- Policy steering only affected clinically ambiguous pairs
- The mathematical guarantee: `Cov(clinical_risk, orthogonalized_policy) = 0`

## Budget Guidance for Healthcare

| Budget | Use Case |
|--------|----------|
| 0.5    | Aggressive rebalancing (capacity management, equity initiatives) |
| 0.7    | Standard — protect clinical judgment, apply institutional policy |
| 0.9    | Minimal steering — only where clinical model is truly uncertain |

## Install

```bash
pip install governed-rank
```

---

*From the governed-rank validation suite. [Source](https://github.com/rdoku/governed-rank)*
