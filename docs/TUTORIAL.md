# governed-rank Tutorial

A progressive, 5-chapter guide to governed reranking — from a 3-line quick start to advanced gap calibration and isotonic projection.

```
pip install governed-rank
```

---

## Chapter 1 — Quick Start with govern()

`govern()` is the simplest entry point. Three arguments, one function call, and you have a governed ranking with a full audit trail.

### Minimal Example

```python
from mosaic import govern

result = govern(
    base_scores={"doc1": 0.95, "doc2": 0.88, "doc3": 0.72, "doc4": 0.60, "doc5": 0.45},
    steering_scores={"doc1": -0.4, "doc2": 0.1, "doc3": 0.9, "doc4": 0.3, "doc5": 0.7},
    budget=0.30,
)
```

**Arguments:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_scores` | `dict`, `list`, or `np.ndarray` | The base ranker's scores. Higher = more relevant. |
| `steering_scores` | `dict`, `list`, or `np.ndarray` | The policy signal. Higher = more desirable per your objective. |
| `budget` | `float` (0.0–1.0) | Fraction of adjacent-pair edges to protect. Default `0.30`. |

### Reading the Result

`govern()` returns a `GovernResult`:

```python
print(result.ranked_items)       # reranked item order, e.g. ["doc1", "doc3", "doc2", ...]
print(result.scores)             # {item: final_score} mapping
print(result.n_protected_edges)  # how many ordering edges were locked
print(result.n_active_constraints)  # how many constraints were binding
print(result.projection_coeff)   # alignment between steering and base (see Ch 2)
```

### Per-Item Receipts

Every item gets a `GovernReceipt` — a full audit trail:

```python
for receipt in result.receipts:
    print(f"Item: {receipt.item}")
    print(f"  Base rank: {receipt.base_rank} → Final rank: {receipt.final_rank}")
    print(f"  Base score: {receipt.base_score:.3f}")
    print(f"  Steering score: {receipt.steering_score:.3f}")
    print(f"  Orthogonalized steering: {receipt.orthogonalized_steering:.3f}")
    print(f"  Final score: {receipt.final_score:.3f}")
```

The receipt shows exactly why each item moved (or didn't). `orthogonalized_steering` is the steering signal after interference with the base score has been removed (Chapter 2 explains why this matters).

### Domain Examples

**Content Moderation** — demote toxic content without hurting engagement:

```python
result = govern(
    base_scores=engagement_scores,         # from your engagement model
    steering_scores=toxicity_penalties,     # negative = toxic, positive = safe
    budget=0.30,
)
# Toxic items move down; high-engagement safe items stay put.
```

**Fairness** — promote underrepresented groups without sacrificing quality:

```python
result = govern(
    base_scores=quality_scores,            # from your quality model
    steering_scores=fairness_boosts,       # higher for underrepresented candidates
    budget=0.30,
)
# Protected-group candidates rise; top-quality decisions are locked.
```

**RAG Safety** — steer retrieval toward grounded documents:

```python
result = govern(
    base_scores=retrieval_similarity,      # cosine similarity from your vector DB
    steering_scores=groundedness_scores,   # higher = more factually grounded
    budget=0.50,                           # higher budget for safety-critical
)
# Grounded docs surface; the most relevant results are protected.
```

### Run the Notebooks

The best way to see `govern()` in action is to run the included notebooks. Each one is a self-contained worked example with reproducible results:

| Notebook | Domain | What it shows |
|----------|--------|---------------|
| `notebooks/demo.ipynb` | Fairness (COMPAS) | AIR 0.773 → 0.916, quality 95% |
| `notebooks/content_moderation.ipynb` | Content feeds | Toxicity reduction with quality preservation |
| `notebooks/fraud_detection.ipynb` | Fraud review queues | 4.1x fraud value improvement |
| `notebooks/objective_discovery.ipynb` | Policy selection | Which objectives align with users |

---

## Chapter 2 — Understanding Orthogonalization

### Why `base + λ * policy` Fails

The naive approach to multi-objective ranking is:

```
final_score = base_score + λ * policy_score
```

This fails because the policy signal is often **correlated** with the base score. When you add a correlated signal, you don't steer — you amplify. Items that were already ranked high get ranked higher. Items that need to move don't move.

When the correlation is negative, you actively break the ranking with unpredictable changes.

### The Fix: Orthogonalization

`orthogonalize_against_base()` removes the component of the steering signal that is aligned with the base scores:

```
u_perp = u - (u · s / s · s) * s
```

**The shadow metaphor**: Think of the base scores as a direction in space. The steering signal casts a "shadow" along that direction — the part of the steering that just restates what the base ranker already knows. Orthogonalization subtracts the shadow, leaving only the perpendicular remainder — the genuinely new information about how items should move.

After this projection, the steering signal has **zero correlation** with the base scores. It can only move items in directions the base ranker has no opinion about.

**Concrete example**: In `notebooks/content_moderation.ipynb`, engagement and toxicity correlate at r = 0.424 (toxic content is engaging). After orthogonalization, the correlation drops to ≈ 0 — the cleaned safety signal can only move posts where the engagement model is uncertain.

### Standalone Usage

```python
from mosaic.orthogonalization import orthogonalize_against_base

base = {"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3, "e": 0.1}
steering = {"a": 0.8, "b": 0.6, "c": 0.4, "d": 0.2, "e": 0.0}

orth = orthogonalize_against_base(base_scores=base, steering_utilities=steering)
```

### OrthogonalizationResult Fields

```python
print(orth.projection_coeff)   # alignment of steering with base (high = were very aligned)
print(orth.corr_before)        # Pearson correlation before orthogonalization
print(orth.corr_after)         # Pearson correlation after (~0.0)
print(orth.u_magnitude_before) # RMS of steering signal before
print(orth.u_magnitude_after)  # RMS after (may shrink if signals were aligned)
print(orth.u_perp)             # {item: orthogonalized_score} dict
```

**Interpreting `projection_coeff`:** A high value means the steering signal was strongly aligned with base scores — most of what it was saying was already captured by the base ranker. Orthogonalization removed that redundant component, leaving only the genuinely new information.

### Computing Target Scores

After orthogonalization, combine base scores with the orthogonalized steering:

```python
from mosaic.orthogonalization import compute_target_scores

target = compute_target_scores(base_scores=base, u_perp=orth.u_perp)
# target[item] = base_scores[item] + u_perp[item]
```

These target scores represent where items *should* rank if the steering signal were fully applied. The isotonic projection step (Chapter 5) then finds the closest feasible ranking that respects the protected edges.

---

## Chapter 3 — The Budget Knob

### What Budget Controls

The `budget` parameter (0.0 to 1.0) controls what fraction of adjacent-pair ordering edges are protected — i.e., locked so steering cannot reverse them.

- **`budget=0.0`** — No edges protected. Full reordering. Maximum policy effect.
- **`budget=0.30`** — The 30% of edges with the largest base-score gaps are locked. Good default.
- **`budget=1.0`** — All edges protected. No reordering. Output matches base ranking exactly.

The edges with the largest gaps are where the base ranker is most confident. Protecting these first gives maximum accuracy preservation for any budget level.

### Sweeping the Budget

To see how budget affects your ranking, sweep from 0.0 to 1.0:

```python
from mosaic import govern

base = {"a": 0.95, "b": 0.85, "c": 0.70, "d": 0.55, "e": 0.40}
steer = {"a": -0.3, "b": 0.1, "c": 0.8, "d": 0.5, "e": 0.9}

for budget in [0.0, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0]:
    r = govern(base, steer, budget=budget)
    print(f"budget={budget:.2f}  ranking={r.ranked_items}  "
          f"protected={r.n_protected_edges}  active={r.n_active_constraints}")
```

At low budget, the ranking shifts significantly toward the steering objective. As budget increases, more of the base ordering is locked in place.

### Standalone Edge Protection

You can compute protected edges without running the full pipeline:

```python
from mosaic.gap_calibration import get_protected_edges_by_budget

base_order = ["a", "b", "c", "d", "e"]     # sorted by descending base score
base_scores = {"a": 0.95, "b": 0.85, "c": 0.70, "d": 0.55, "e": 0.40}

protected = get_protected_edges_by_budget(
    base_order=base_order,
    base_scores=base_scores,
    budget_pct=0.30,
    max_rank=50,
)
print(f"Protected edge indices: {protected}")
# These are 0-based indices into base_order: edge i means pair (base_order[i], base_order[i+1])
```

### Budget in Practice: Content Moderation Example

From `notebooks/content_moderation.ipynb` — the budget sweep shows a smooth tradeoff between toxicity reduction and quality retention:

| Budget | Toxic in top-10 | Mean Toxicity | Quality |
|--------|----------------|--------------|---------|
| 0.00 | 4 | 0.268 | 72.8% |
| 0.30 | 5 | 0.280 | 75.5% |
| 0.50 | 6 | 0.308 | 81.2% |
| 1.00 | 7 | 0.339 | 100.0% |

No cliff, no sudden collapse — just a smooth dial from maximum steering to no change.

### Budget Guidance by Domain

| Domain | Suggested Budget | Rationale |
|--------|-----------------|-----------|
| E-commerce recommendations | 0.20–0.40 | Moderate protection; engagement is forgiving |
| Content moderation | 0.20–0.40 | Must steer; some disruption is acceptable |
| Fairness / compliance | 0.20–0.40 | Needs real steering; tight budget defeats purpose |
| Healthcare / clinical | 0.50–0.70 | High stakes; protect clinical accuracy aggressively |
| RAG / retrieval safety | 0.40–0.60 | Balance relevance with groundedness |
| Fraud review queues | 0.30–0.50 | Must catch fraud; moderate protection |

---

## Chapter 4 — Gap Calibration

> **Note**: Gap calibration is advanced usage. The `govern()` function handles edge protection automatically using raw score gaps, which works well for most use cases. You only need explicit calibration if you have historical data and want to map gaps to correctness probabilities.

### Why Raw Score Gaps Aren't Probabilities

When the base ranker scores item A at 0.82 and item B at 0.80, the gap is 0.02. But how confident is the ranker that A truly belongs above B? That depends on the ranker's calibration — a 0.02 gap from a well-calibrated model means something different than 0.02 from a poorly-calibrated one.

Gap calibration learns the mapping from score gaps to correctness probabilities: `P(A > B | gap)`.

### Building Training Pairs

You need historical baskets (sets of items a user interacted with) and a base scoring function:

```python
from mosaic.gap_calibration import extract_pairs_pos_neg

def my_scorer(positives, negatives):
    """Score items given context. Returns {item_id: score}."""
    all_items = positives + negatives
    # ... your model scoring logic ...
    return {item: score for item, score in zip(all_items, scores)}

pairs = extract_pairs_pos_neg(
    baskets=historical_baskets,     # List[List[int]] — each basket is a set of interacted items
    base_scorer=my_scorer,          # Callable[[List[int], List[int]], Dict[int, float]]
    n_items=total_item_count,       # total number of items in catalog
    n_baskets=5000,                 # how many baskets to sample
    neg_per_pos=10,                 # negative samples per positive
    max_rank=50,                    # only consider items ranked in top-50
    rng_seed=42,
)
# pairs: List[Tuple[float, bool]] — (score_gap, was_correct_order)
```

### Learning the Calibration Curve

```python
from mosaic.gap_calibration import learn_gap_calibration, CalibrationConfig

calibration = learn_gap_calibration(
    pairs=pairs,
    config=CalibrationConfig(
        n_buckets=30,
        min_samples_per_bucket=50,
        monotonic=True,
    ),
)

calibration.print_summary()  # shows the gap → confidence curve
```

### CalibrationResult

```python
# Look up confidence for a specific gap
conf = calibration.gap_to_conf(0.05)
print(f"Gap 0.05 → {conf:.1%} confidence")

# Inverse: what gap gives 90% confidence?
gap = calibration.conf_to_gap(0.90)
print(f"90% confidence requires gap ≥ {gap:.4f}")

# Save/load for production
calibration.save("calibration_model.json")
loaded = CalibrationResult.load("calibration_model.json")
```

### Using Calibration with Edge Protection

**Threshold mode** — protect edges where confidence exceeds ρ:

```python
from mosaic.gap_calibration import get_protected_edges

protected = get_protected_edges(
    base_order=sorted_items,
    base_scores=scores,
    calibration=calibration,
    rho=0.90,  # protect edges where P(correct) ≥ 90%
)
```

**Budget mode** — protect a fixed fraction of edges, prioritized by gap size:

```python
from mosaic.gap_calibration import get_protected_edges_by_budget

protected = get_protected_edges_by_budget(
    base_order=sorted_items,
    base_scores=scores,
    calibration=calibration,     # optional — used to sort by confidence
    budget_pct=0.30,
    max_rank=50,
)
```

Budget mode is the default in `govern()`. Threshold mode is available when you have a trained calibration model and want confidence-based protection.

---

## Chapter 5 — Isotonic Projection

### What It Does

After orthogonalization gives you target scores and edge protection gives you constraints, the isotonic projection finds the **closest feasible ranking** — the one that maximizes policy effect while respecting every protected edge.

### Standalone Usage

```python
from mosaic.isotonic_projection import isotonic_project_on_runs

base_order = ["a", "b", "c", "d", "e"]
target_scores = {"a": 0.7, "b": 0.9, "c": 0.6, "d": 0.8, "e": 0.5}
protected_edges = [1, 3]  # protect edges b→c and d→e

proj = isotonic_project_on_runs(
    base_order=base_order,
    target_scores=target_scores,
    protected_edges=protected_edges,
)
```

### ProjectionResult Fields

```python
print(proj.z)                    # {item: final_score} after projection
print(proj.n_constraints)        # number of protected edges
print(proj.n_active_constraints) # how many constraints were binding (items wanted to swap)
print(proj.pooled_blocks)        # items that were pooled (given equal scores) to satisfy constraints
print(proj.n_pre_violations)     # how many protected edges the target scores violated
```

**Pooled blocks:** When the target scores want to reverse a protected edge, the PAV algorithm pools those items together and gives them the weighted average score. The `pooled_blocks` field shows which items were pooled. More pooling = more constraint activity = the steering signal was fighting the base ranking harder.

### Computing the Final Ranking

```python
from mosaic.isotonic_projection import compute_final_ranking

final_order = compute_final_ranking(z=proj.z, base_order=base_order)
# Sorts by descending z with stable tie-breaking by base order position
```

Stable tie-breaking means that when two items have the same projected score (because they were pooled), they keep their original base ordering. This ensures deterministic, reproducible rankings.

### The PAV Algorithm

The Pool Adjacent Violators (PAV) algorithm is a classic method from isotonic regression. It works in a single O(N) pass:

1. Walk left to right through items in base order.
2. If the current item's target score violates a protected constraint with its predecessor, pool them together and replace both scores with their weighted average.
3. Continue: the pooled block may now violate a constraint with *its* predecessor, so merge again if needed.
4. Result: a monotone sequence on every protected run, as close as possible (in squared error) to the original target scores.

---

## Chapter 6 — Worked Examples (Notebooks)

The repository includes 4 notebooks with complete, reproducible examples. Each demonstrates a different domain and includes budget sweeps, head-to-head comparisons, and diagnostic output.

### Fairness — COMPAS (`notebooks/demo.ipynb`)

Reduces racial bias in recidivism risk rankings. Steers a COMPAS-derived ranking toward demographic parity using a fairness boost signal. Adverse impact ratio improves from 0.773 to 0.916 (passing the 4/5ths rule) while retaining 95% ranking quality. Also includes a MovieLens genre-steering example and an orthogonalization walkthrough.

### Content Moderation (`notebooks/content_moderation.ipynb`)

200 synthetic posts with a realistic engagement-toxicity correlation (r = 0.424). Shows why naive toxicity penalties over-correct and break the ranking, while MOSAIC targets only the uncertain zone. Includes a 7-row budget sweep table demonstrating the smooth quality-safety tradeoff.

### Fraud Detection (`notebooks/fraud_detection.ipynb`)

300 simulated transactions with a log-normal amount distribution. Steers a fraud review queue toward high-value suspicious transactions. MOSAIC captures 4.1x more fraud value in the top-20 and 10.4x more in the auto-block tier — at the same precision. Fraud slipping through the allow tier drops 81%.

### Objective Discovery (`notebooks/objective_discovery.ipynb`)

500 synthetic articles across 7 categories. Tests 7 candidate policies to discover which objectives align with user preferences before deploying. Key finding: quality-based steering is the only policy that achieves both engagement lift AND diversity gain. Forced diversity fights user preference.

---

- **GitHub:** [github.com/rdoku/governed-rank](https://github.com/rdoku/governed-rank)
- **PyPI:** [pypi.org/project/governed-rank](https://pypi.org/project/governed-rank/)
- **License:** Apache 2.0
