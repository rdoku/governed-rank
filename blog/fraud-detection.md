# Fraud Detection with governed-rank — Steering Transaction Rankings Toward Policy Objectives

Fraud detection models produce risk scores. But risk isn't the only signal. Compliance teams have policies: flag high-value transactions first, prioritize specific merchant categories, ensure review queues reflect regulatory requirements. Bolting policy signals onto fraud scores creates the same injection problem that plagues every ranking system.

## The Problem

Your fraud model ranks transactions by P(fraud). Your compliance team wants to prioritize:
- High-value transactions (regulatory requirement)
- Cross-border transactions (AML policy)
- Specific merchant categories (elevated risk profile)

Naive approach: `final_score = fraud_score + w * policy_score`. This fails because fraud scores correlate with many policy signals — high-value transactions already tend to have higher fraud scores, creating interference.

## The governed-rank Solution

```python
from mosaic import govern

# fraud_scores: P(fraud) from your model for each transaction
# policy_scores: composite policy signal
#   positive = higher priority for review
policy_scores = {
    txn_id: (
        0.4 * (amount[txn_id] / max_amount) +     # value-weighted
        0.3 * (1.0 if is_cross_border[txn_id] else 0.0) +  # AML flag
        0.3 * (1.0 if mcc[txn_id] in elevated_mccs else 0.0)  # merchant risk
    )
    for txn_id in transactions
}

result = govern(
    base_scores=fraud_scores,
    steering_scores=policy_scores,
    budget=0.5,
)

# result.ranked_items: review queue in priority order
# result.receipts: audit trail per transaction
```

`budget=0.5` is appropriate here — you want to protect the fraud model's most confident detections while allowing policy to influence the uncertain middle.

## Why This Matters for Fraud

Fraud review queues have limited capacity. Analysts can review maybe 100-500 transactions per day. The ordering matters: reviewing a $50,000 cross-border transaction before a $50 domestic one could be the difference between catching money laundering and missing it.

But if you reorder purely by policy, you'll push genuinely fraudulent transactions down the queue. MOSAIC guarantees that the transactions your fraud model is most confident about stay at the top.

## Building Policy Signals

### Value-Weighted Priority

```python
import numpy as np

# Transactions above the SAR threshold get maximum priority
SAR_THRESHOLD = 10000

value_scores = {
    txn_id: min(amount[txn_id] / SAR_THRESHOLD, 1.0)
    for txn_id in transactions
}
```

### AML Cross-Border Signal

```python
aml_scores = {
    txn_id: (
        1.0 if is_cross_border[txn_id] and country[txn_id] in high_risk_countries
        else 0.5 if is_cross_border[txn_id]
        else 0.0
    )
    for txn_id in transactions
}
```

### Composite with govern()

```python
# Combine policy signals
composite_policy = {
    txn_id: 0.5 * value_scores[txn_id] + 0.5 * aml_scores[txn_id]
    for txn_id in transactions
}

result = govern(
    base_scores=fraud_scores,
    steering_scores=composite_policy,
    budget=0.5,
)
```

## The Audit Requirement

Financial regulators require explainable decisions. Every transaction in the review queue needs documentation of why it's prioritized:

```python
for txn_id in result.ranked_items[:20]:  # top-20 review queue
    receipt = next(r for r in result.receipts if r.item == txn_id)
    print(f"Transaction {txn_id}:")
    print(f"  Fraud score: {receipt.base_score:.4f}")
    print(f"  Policy signal: {receipt.steering_score:.4f}")
    print(f"  Orthogonalized policy: {receipt.orthogonalized_steering:.4f}")
    print(f"  Priority rank: {receipt.final_rank}")
    print(f"  Moved from rank: {receipt.base_rank}")
```

The `orthogonalized_steering` field shows the "pure policy" component — after removing the part that already correlates with the fraud score. This is the defensible explanation: "this transaction moved up because of policy considerations independent of its fraud risk."

## IEEE-CIS Fraud Validation

We validated this approach on the IEEE-CIS Fraud Detection dataset. The key finding: policy-steered ranking preserves fraud detection performance (measured by the base model's ordering of true positives) while reordering transactions according to policy objectives.

## Budget Guidance for Fraud

| Budget | When to Use |
|--------|-------------|
| 0.3    | Aggressive policy steering — compliance-heavy environments |
| 0.5    | Balanced — protect fraud model while applying policy |
| 0.7    | Conservative — fraud model mostly drives, light policy nudge |
| 0.9    | Minimal — only steer where fraud model is very uncertain |

## Install

```bash
pip install governed-rank
```

---

*From the governed-rank validation suite. [Source](https://github.com/rdoku/governed-rank)*
