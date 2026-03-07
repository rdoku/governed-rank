# Understanding MOSAIC: A Gentle Introduction

## The Problem Every Ranking System Has

Every software system that shows a list of things to a user has the same structure: a model scores items, and the highest-scoring items appear first. Search engines rank by relevance. Recommendation systems rank by predicted engagement. Lending platforms rank applicants by creditworthiness. Healthcare systems rank patients by clinical priority.

These models work. The problem starts when the business has a second objective.

An e-commerce company wants to promote fresh produce — not because it's what the model predicts you'll click, but because it's a strategic priority. A content platform wants to suppress toxic content — not because the engagement model values safety, but because the policy team requires it. A lending institution needs to ensure fair treatment across demographics — not because the credit model encodes fairness, but because regulators demand it.

Every organization eventually needs to **steer** a ranked list toward some policy objective that the base model doesn't optimize for.

---

## Why the Obvious Fix Doesn't Work

The standard approach is straightforward: add the policy signal to the base score.

```
final_score = base_score + weight * policy_score
```

A toxicity penalty gets subtracted from engagement. A fairness boost gets added to quality. A produce-promotion signal gets mixed into the recommendation score.

This seems reasonable. It is also reliably broken.

### The Interference Problem

Suppose you have a content feed ranked by engagement. You want to demote toxic content. You compute a toxicity score for each item and subtract it:

```
final = engagement - 0.3 * toxicity
```

Here's what actually happens. Toxic content tends to be *engaging* — outrage drives clicks. So your toxicity signal is **correlated** with your engagement signal. When you subtract a correlated signal, you don't just demote toxic items. You reshape the entire ranking in ways you didn't intend:

- Items that are both engaging and non-toxic get a relative boost they didn't need
- Items that are mildly engaging and mildly toxic get pulled down disproportionately
- The ranking changes are unpredictable — some items move a lot, others barely move, and the pattern depends on the specific correlation structure

The ML team sees recall drop. They revert the change. The policy team escalates. This cycle repeats across every organization that tries to bolt a policy signal onto a ranking model.

The root cause is **correlation between the two signals**. When your policy signal shares information with your base scores — and it almost always does — adding them creates interference. The combined score amplifies some items, suppresses others, and the net effect on the ranking is a mess.

### A Concrete Example

Five documents ranked by engagement:

| Doc | Engagement | Toxicity Penalty | Naive Combined |
|-----|-----------|-----------------|----------------|
| A   | 0.95      | -0.40           | 0.83           |
| B   | 0.88      | +0.10           | 0.91           |
| C   | 0.72      | +0.90           | 0.99           |
| D   | 0.60      | +0.30           | 0.69           |
| E   | 0.45      | +0.70           | 0.66           |

The base ranking is A > B > C > D > E. After naive combination (engagement + 0.3 * penalty), the ranking becomes C > B > A > D > E.

Document C — which was ranked third by engagement — jumps to first. Not because the policy demanded it, but because C happened to be both moderately engaging *and* strongly non-toxic. The combination double-counted C's qualities. Meanwhile, document A dropped from first to third — a significant demotion — even though A may have been penalized more than necessary.

The policy team wanted to "demote toxic content." What they got was a wholesale reshuffling driven by the correlation between engagement and toxicity.

---

## The MOSAIC Insight

MOSAIC starts from a simple observation: **the problem is the correlation, so remove the correlation first.**

Before you combine two signals, make them independent. Specifically, take the policy signal and remove whatever part of it is redundant with the base scores. What remains is a "pure policy" signal — the part that provides genuinely new information about how items should move.

This is called **orthogonalization**, and it is the key idea behind MOSAIC.

After orthogonalization, the steering signal cannot — by mathematical construction — interfere with the base ranker's decisions. It can only move items in directions that the base ranker has no opinion about.

---

## The Three Steps

MOSAIC works in three steps. Each one solves a specific problem.

### Step 1: Orthogonalize — Remove the Interference

Take the steering signal and project it into the **null space** of the base scores. In plain language: find the part of the steering signal that is just restating what the base ranker already knows, and strip it out.

The math is a single formula:

```
steering_clean = steering - (steering · base / base · base) * base
```

This is the same operation as removing a shadow. If you shine a light along the base-score direction, the steering signal casts a shadow onto it. That shadow is the correlated component — the part of the steering signal that just repeats the base ranker's information. MOSAIC subtracts the shadow and keeps only the perpendicular remainder.

After this step, a mathematical guarantee holds:

```
Correlation(base_scores, steering_clean) = 0
```

This is not approximate. It is exact (up to floating-point precision). The cleaned steering signal is perfectly uncorrelated with the base scores.

**What this means in practice:** Adding the cleaned signal to the base scores cannot amplify items that were already highly ranked. It cannot systematically penalize items based on their base score. The double-counting problem from naive combination is eliminated by construction.

#### Returning to the Example

When we orthogonalize the toxicity penalties against the engagement scores, the projection removes the engagement-correlated component. The remaining signal reflects *only* the toxicity information that engagement doesn't already capture.

Document A had a raw penalty of -0.40 — but part of that penalty was because A is highly engaging, and engaging content correlates with toxicity. After orthogonalization, A's penalty might shrink to -0.15, reflecting only the "pure toxicity" signal. Document C's raw boost of +0.90 might shrink to +0.50, because part of C's non-toxicity was already reflected in C's moderate engagement score.

The orthogonalized signal gives a fairer picture of what the policy is actually asking for.

### Step 2: Protect — Lock the Confident Decisions

Even after removing interference, you might not want the policy to override *every* ordering decision. Some pairs of items have a huge gap in the base scores — the model is very confident that item A should rank above item B. Other pairs have almost identical scores — the model barely distinguishes them.

MOSAIC uses a **budget** parameter to protect the base ranker's most confident decisions.

```
budget = 0.30  →  Protect the top 30% most confident ordering decisions
```

How it works:

1. Look at every adjacent pair in the base ranking
2. Compute the score gap between them (larger gap = more confident)
3. Sort all gaps from largest to smallest
4. Lock the top `budget%` of pairs — these become **protected edges** that the policy cannot reverse

At `budget=0.30`, the 30% of ordering decisions where the base ranker is most sure of itself become untouchable. The policy can only rearrange items where the base ranker is uncertain.

This gives the ML team a single knob that controls risk:

| Budget | What Happens |
|--------|-------------|
| `0.00` | Nothing is protected. The policy can fully reorder the list. Maximum policy effect. |
| `0.30` | The 30% most confident ordering decisions are locked. Good default. |
| `0.50` | Half the ordering is locked. More conservative. |
| `1.00` | Everything is locked. The output is identical to the base ranking. |

The budget provides a smooth, monotonic tradeoff between accuracy (fidelity to the base ranker) and policy compliance (how much steering takes effect). There is no cliff, no sudden collapse — just a dial that slides from "full policy" to "no policy."

#### Why Gaps, Not Positions?

A natural question: why protect based on score gaps rather than just protecting the top-K positions?

Because position doesn't tell you about confidence. If item 1 has a score of 0.95 and item 2 has a score of 0.94, the model barely distinguishes them — protecting that ordering wastes budget on a coin flip. But if item 5 has a score of 0.70 and item 6 has a score of 0.30, the model is very confident about that ordering — protecting it preserves a genuinely informative decision.

Gap-based protection spends the budget where it matters most.

### Step 3: Project — Find the Best Feasible Ranking

Now we have:
- **Target scores**: base scores + orthogonalized steering (what we'd like the ranking to look like)
- **Protected edges**: ordering constraints that cannot be violated

The target scores represent our ideal outcome — full policy effect with interference removed. But some of those target scores might violate the protected edges. Item 3 might have a higher target score than item 2, even though the edge between positions 2 and 3 is protected (meaning item 2 must stay above item 3).

MOSAIC needs to find final scores that are **as close as possible to the targets** while **respecting every protected constraint**.

This is a constrained optimization problem:

```
Find z that minimizes:  sum of (z_i - target_i)²
Subject to:             z_k >= z_{k+1}  for every protected edge k
```

In words: make the final scores as close to the targets as you can, but wherever the base ranker said "this item must stay above that item," honor it.

MOSAIC solves this using the **Pool Adjacent Violators (PAV)** algorithm — a classic method from isotonic regression, adapted here for ranking constraints.

#### How PAV Works (Intuitively)

Walk through the items in base-rank order. If the current item's target score respects the constraint with its neighbor, leave it alone. If it violates a constraint (the lower-ranked item has a higher target score), **pool** the two items together — replace both scores with their average.

After pooling, check again: does the new pooled block violate the constraint with its predecessor? If so, merge again. Keep merging until all constraints are satisfied.

The result is the closest set of scores to the targets that respects every protected constraint. This is mathematically provable — PAV finds the global optimum of the constrained least-squares problem in a single linear pass.

#### What This Means

The final ranking is **Pareto-optimal**: you cannot get more policy effect without violating a protected constraint, and you cannot protect more constraints without giving up policy effect. Every item is in the best position it can be, given the constraints.

---

## Putting It Together

The full pipeline:

```
Input: base_scores, steering_scores, budget

1. Orthogonalize  →  Remove interference between steering and base
                      Guarantee: Corr(base, steering_clean) = 0

2. Compute targets → target_i = base_i + steering_clean_i

3. Protect edges   → Lock top budget% of edges by score gap
                      These become hard ordering constraints

4. Project (PAV)   → Find scores closest to targets respecting constraints
                      Guarantee: Pareto-optimal solution

Output: reranked list + per-item audit receipts
```

In code:

```python
from mosaic import govern

result = govern(
    base_scores={"doc1": 0.95, "doc2": 0.88, "doc3": 0.72, "doc4": 0.60, "doc5": 0.45},
    steering_scores={"doc1": -0.4, "doc2": 0.1, "doc3": 0.9, "doc4": 0.3, "doc5": 0.7},
    budget=0.30,
)

print(result.ranked_items)      # The reranked list
print(result.projection_coeff)  # How much interference was removed
```

Three arguments. One function call.

---

## The Audit Trail

Every item gets a receipt explaining exactly what happened to it:

```python
for receipt in result.receipts:
    print(f"Item: {receipt.item}")
    print(f"  Base score:      {receipt.base_score:.3f}")
    print(f"  Steering score:  {receipt.steering_score:.3f}")
    print(f"  After orthogonalization: {receipt.orthogonalized_steering:.3f}")
    print(f"  Final score:     {receipt.final_score:.3f}")
    print(f"  Rank movement:   {receipt.base_rank} → {receipt.final_rank}")
```

For each item, the receipt shows:
- **base_score**: Where the base ranker placed it
- **steering_score**: What the policy signal said
- **orthogonalized_steering**: The policy signal after interference removal
- **final_score**: The score after constrained projection
- **base_rank → final_rank**: How much the item moved

This matters for regulated domains. If a regulator asks "why did this item move?", you can point to the receipt. If they ask "why didn't it move?", you can show that a protected edge prevented it. Every decision is traceable.

---

## What Makes MOSAIC Different

### Versus Naive Reranking (score addition)

Naive combination fails because of signal correlation. MOSAIC removes the correlation first. In the content moderation notebook (`notebooks/content_moderation.ipynb`), naive toxicity subtraction reshuffles the entire ranking (tau = 0.438 vs base), while MOSAIC achieves comparable toxicity reduction with better quality retention (tau = 0.510), because orthogonalization absorbs the interference.

### Versus Constrained Optimization (ILP, slot-based)

Many fairness and diversity approaches frame reranking as an integer linear program: "place at least K items from group G in the top N positions." These approaches are powerful but brittle — they require explicit constraint formulation per domain, and they scale poorly (ILP is NP-hard in the worst case).

MOSAIC doesn't require domain-specific constraints. You provide a continuous steering signal — any real-valued score reflecting your policy objective — and MOSAIC handles the rest. The algorithm runs in O(N log N) time regardless of domain.

### Versus Learning-to-Rank (multiobjective)

You could retrain the ranking model with multiple objectives. This is expensive, requires retraining whenever policies change, and provides no guarantees about the tradeoff between objectives. MOSAIC is a post-processing step that works with any base ranker, requires no retraining, and provides a mathematical guarantee of Pareto-optimality.

### The Core Novelty

MOSAIC combines three ideas that, to our knowledge, have not been composed before:

1. **Orthogonalization of the steering signal against the base scores.** This is borrowed from linear algebra (Gram-Schmidt), but its application to ranking — projecting a policy signal into the null space of the relevance direction — is new. It provides an exact guarantee of zero interference, not an approximate one.

2. **Gap-based edge protection with a budget knob.** Rather than protecting fixed positions (top-K) or requiring domain-specific constraints, MOSAIC uses the base ranker's own confidence (score gaps) to decide what to protect. The budget parameter provides a single, interpretable control for the accuracy-policy tradeoff.

3. **Constrained isotonic projection for the final ranking.** PAV (Pool Adjacent Violators) is well-known for probability calibration, but its use as a ranking projection operator — finding the closest feasible ranking to a target ranking subject to ordering constraints — is a novel application.

The composition of these three steps is what makes MOSAIC work: orthogonalization removes interference, budget protection preserves confidence, and isotonic projection finds the optimum. Each step is simple. Together, they solve a problem that has frustrated ranking teams for years.

---

## When to Use MOSAIC

MOSAIC applies whenever you have:

1. **A base ranking** — any model that produces scores for items
2. **A policy objective** — any signal you want the ranking to reflect
3. **A need to control the tradeoff** — you can't just replace the base ranker

Common applications:

- **Content moderation**: Demote toxic content without killing engagement
- **Fairness**: Promote underrepresented groups without destroying quality
- **E-commerce merchandising**: Boost strategic products without breaking recommendations
- **Healthcare compliance**: Enforce equitable access without overriding clinical judgment
- **RAG safety**: Steer retrieval toward grounded documents
- **Fraud detection**: Reorder review queues by policy priority
- **Lending**: Ensure fair treatment across demographics

In each case, the pattern is the same: a base ranker that works well, a policy signal that needs to influence the ranking, and a requirement that the base ranker's quality not collapse.

---

## Complexity and Performance

The entire pipeline runs in **O(N log N)** time:

| Step | Complexity | Why |
|------|-----------|-----|
| Sort items by base score | O(N log N) | Standard sort |
| Orthogonalize | O(N) | One dot product, one vector subtraction |
| Compute target scores | O(N) | Element-wise addition |
| Identify protected edges | O(N) | One pass over sorted gaps |
| Isotonic projection (PAV) | O(N) | Single linear scan with backtracking |
| Final sort by projected scores | O(N log N) | Standard sort |

For a typical reranking of 100 to 1,000 items, the entire pipeline runs in sub-millisecond time. MOSAIC adds negligible latency to a serving path.

The package is 99 KB of pure Python with no compiled extensions. NumPy is the only runtime dependency (and is optional for small inputs).

---

## Summary

MOSAIC solves the problem of steering ranked lists toward policy objectives without breaking accuracy.

**The problem**: Policy signals correlate with base scores. Naive combination creates interference that degrades the ranking.

**The fix**: Three steps — orthogonalize (remove interference), protect (lock confident decisions), project (find the optimum).

**The guarantees**:
- Zero correlation between base scores and cleaned steering signal (exact)
- Pareto-optimal final ranking (no free improvements exist)
- Full audit trail for every item

**The interface**: Three arguments, one function call, one budget knob.

```python
from mosaic import govern

result = govern(base_scores, steering_scores, budget=0.30)
```

---

## Install

```bash
pip install governed-rank
```

## Links

- [GitHub: rdoku/governed-rank](https://github.com/rdoku/governed-rank)
- [PyPI: governed-rank](https://pypi.org/project/governed-rank/)
- [Tutorial: Quick Start to Production](./TUTORIAL.md)
