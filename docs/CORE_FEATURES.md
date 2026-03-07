# governed-rank: Core Features Documentation

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [govern() — Zero-Config Reranking](#2-govern--zero-config-reranking)
3. [Orthogonalization — Interference Removal](#3-orthogonalization--interference-removal)
4. [Gap Calibration — Confidence Learning](#4-gap-calibration--confidence-learning)
5. [Isotonic Projection — Constrained Optimization](#5-isotonic-projection--constrained-optimization)
6. [Validated Results](#6-validated-results)

---

## 1. Architecture Overview

governed-rank implements a 3-stage reranking pipeline that steers any ranked list toward policy objectives while preserving the base ranker's confident decisions.

```
Step 1  Orthogonalization      →  u_⊥ = u − proj(u onto s)
Step 2  Protected Edges        →  top B% by gap confidence
Step 3  Constrained Projection →  isotonic regression on protected runs
```

**Core insight**: Naively adding a policy signal to a base ranker degrades accuracy because the policy signal correlates with the base scores, creating interference. governed-rank solves this by (1) removing the component of the policy signal that correlates with the base score, and (2) locking the base ranker's most confident ordering decisions via budget-controlled constraints. See `notebooks/content_moderation.ipynb` for a worked example.

**Entry point**: `govern()` — zero-config, 3 arguments, works with any key type.

**Complexity**: O(N log N) sorting + O(N) orthogonalization + O(N) projection = O(N log N) overall.

---

## 2. govern() — Zero-Config Reranking

**Module**: `mosaic/govern.py`

The simplest possible entry point. No config objects needed.

### Usage

```python
from mosaic import govern

result = govern(
    base_scores={"doc1": 0.9, "doc2": 0.8, "doc3": 0.7, "doc4": 0.6, "doc5": 0.5},
    steering_scores={"doc1": -0.5, "doc2": 0.3, "doc3": 0.8, "doc4": 0.1, "doc5": 0.6},
    budget=0.3,
)

result.ranked_items   # reranked order
result.scores         # final scores per item
result.receipts       # per-item audit trail
```

### Function Signature

```python
def govern(
    base_scores: Dict[Hashable, float],    # item → relevance score
    steering_scores: Dict[Hashable, float], # item → policy signal
    budget: float = 0.30,                   # fraction of edges to protect [0, 1]
) -> GovernResult
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_scores` | `Dict[Hashable, float]` | Your base ranker's scores. Keys can be strings, ints, or any hashable type. |
| `steering_scores` | `Dict[Hashable, float]` | Policy signal. Positive = promote, negative = demote. |
| `budget` | `float` | Fraction of base ordering edges to protect. `0.0` = full reorder allowed, `1.0` = base order locked entirely. Default `0.30` protects the top 30% most confident decisions. |

### Return Types

**GovernResult**:

| Field | Type | Description |
|-------|------|-------------|
| `ranked_items` | `List[Any]` | Final reranked order (best first) |
| `scores` | `Dict[Any, float]` | Final projected score per item |
| `receipts` | `List[GovernReceipt]` | Per-item audit trail |
| `n_protected_edges` | `int` | Number of ordering constraints enforced |
| `n_active_constraints` | `int` | Number of constraints that were actually binding |
| `projection_coeff` | `float` | How much of the steering signal was aligned with base scores (diagnostic) |

**GovernReceipt** (per-item audit):

| Field | Type | Description |
|-------|------|-------------|
| `item` | `Any` | Original item key |
| `base_score` | `float` | Score from the base ranker |
| `steering_score` | `float` | Raw policy signal |
| `orthogonalized_steering` | `float` | Steering after interference removal |
| `final_score` | `float` | Score after constrained projection |
| `base_rank` | `int` | Position in original ranking |
| `final_rank` | `int` | Position in final ranking |

### Internal Pipeline

1. Sort items by `base_scores` descending → establish base order
2. Map arbitrary keys to integer indices (core pipeline requires ints)
3. `orthogonalize_against_base()` → remove interference
4. `compute_target_scores()` → `t_i = s_i + u_⊥_i`
5. `get_protected_edges_by_budget()` → select edges to lock
6. `isotonic_project_on_runs()` → solve constrained optimization
7. `compute_final_ranking()` → final order with stable tie-breaking
8. Map back to original keys

### Domain Examples

**Content Moderation**:
```python
result = govern(
    base_scores=engagement_scores,       # predicted engagement
    steering_scores=toxicity_penalties,   # negative for toxic content
    budget=0.3,
)
```

**Fairness**:
```python
result = govern(
    base_scores=quality_scores,          # hiring model / credit scores
    steering_scores=fairness_boosts,     # positive for underrepresented candidates
    budget=0.3,
)
```

**RAG Safety**:
```python
result = govern(
    base_scores=retrieval_scores,        # embedding similarity
    steering_scores=groundedness_scores, # factuality / compliance signal
    budget=0.5,
)
```

---

## 3. Orthogonalization — Interference Removal

**Module**: `mosaic/orthogonalization.py`

### Problem

When you add a policy signal `u` to base scores `s`, the component of `u` that's aligned with `s` amplifies or suppresses the base ranker's existing beliefs. This "interference" degrades accuracy without providing any policy steering.

### Mathematical Formulation

```
Center both vectors:
  s̃ = s − mean(s)
  ũ = u − mean(u)

Compute projection coefficient:
  α = (s̃ᵀ ũ) / (‖s̃‖² + ε)

Remove the aligned component:
  u_⊥ = ũ − α · s̃

Property: Cov(s̃, u_⊥) = 0  over the candidate set
```

The orthogonalized signal `u_⊥` can only move items in directions that the base ranker is indifferent about. It cannot amplify or suppress what the base ranker already believes.

### Function Signature

```python
def orthogonalize_against_base(
    base_scores: Dict[int, float],
    steering_utilities: Dict[int, float],
    eps: float = 1e-8,
    rescale: bool = False,
    target_rms: float = None,
) -> OrthogonalizationResult
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `base_scores` | Item → base relevance score |
| `steering_utilities` | Item → raw steering utility |
| `eps` | Numerical stability constant. Prevents division by zero for constant base scores. |
| `rescale` | If True, rescale `u_⊥` to match the original RMS magnitude of `u`. Useful when you want the steering effect size to remain constant. |
| `target_rms` | If provided, rescale `u_⊥` to this specific RMS value. Overrides `rescale`. |

### Return: OrthogonalizationResult

| Field | Type | Description |
|-------|------|-------------|
| `u_perp` | `Dict[int, float]` | Orthogonalized steering utility per item |
| `projection_coeff` | `float` | How much of `u` was aligned with `s`. Positive = correlated, negative = anti-correlated. |
| `u_magnitude_before` | `float` | RMS of `u` before orthogonalization |
| `u_magnitude_after` | `float` | RMS of `u_⊥` after orthogonalization |
| `corr_before` | `float` | Correlation(s, u) before. Range [−1, 1]. |
| `corr_after` | `float` | Correlation(s, u_⊥) after. Should be ≈ 0. |

### Diagnostic Interpretation

- **Large `projection_coeff`**: The steering signal was heavily aligned with the base scores. Orthogonalization removed a lot of interference.
- **`u_magnitude_after` ≪ `u_magnitude_before`**: Most of the steering signal was redundant with the base ranker. Only a small orthogonal component remains.
- **`corr_after` ≈ 0**: Confirms successful orthogonalization.

### Helper: compute_target_scores

```python
def compute_target_scores(
    base_scores: Dict[int, float],
    u_perp: Dict[int, float],
) -> Dict[int, float]
```

Computes tentative target scores: `t_i = s_i + u_⊥_i`. These are the scores that would be served if no constraints existed. The projection stage adjusts them to preserve confident base decisions.

### Edge Cases

- **Empty candidate set**: Returns zero orthogonalization.
- **Constant base scores** (all equal): `s̃` is zero → `u_⊥ = ũ` (no projection, full steering).
- **Missing items**: Items in `base_scores` but not in `steering_utilities` get `u = 0.0`.

---

## 4. Gap Calibration — Confidence Learning

**Module**: `mosaic/gap_calibration.py`

### Core Idea

Adjacent items in the base ranking have a "gap" (score difference). This gap reliably predicts how confident the base ranker is about that particular ordering decision. Items with large gaps are confidently ordered; items with tiny gaps could go either way.

The calibration module learns the mapping: `gap → P(ordering is correct)`.

### Training the Calibrator

#### CalibrationConfig

```python
@dataclass
class CalibrationConfig:
    n_buckets: int = 30                    # Number of gap buckets
    min_samples_per_bucket: int = 50       # Minimum samples for reliable estimates
    smoothing_pseudocount: float = 1.0     # Laplace smoothing for small buckets
    use_quantile_buckets: bool = True      # Recommended: equal samples per bucket
    gap_range: Tuple[float, float] = (0.0, 1.0)  # Fallback if quantiles degenerate
    monotonic: bool = True                 # Enforce monotonicity via isotonic regression
```

#### learn_gap_calibration()

```python
def learn_gap_calibration(
    pairs: List[Tuple[float, bool]],  # (gap, was_ordering_correct)
    config: CalibrationConfig = None,
) -> CalibrationResult
```

**Algorithm**:

1. **Bucket creation**: Use quantile edges so each bucket has roughly equal samples. Falls back to linear edges if quantiles produce duplicates.
2. **Empirical P(correct)**: For each bucket, count how many pairs had the correct ordering.
3. **Laplace smoothing**: `P = (correct + α) / (total + 2α)` to prevent zero-probability buckets.
4. **Monotonic enforcement**: Weighted Pool-Adjacent-Violators ensures confidence is non-decreasing with gap. More heavily sampled buckets exert greater influence.

#### CalibrationResult

```python
@dataclass
class CalibrationResult:
    bucket_edges: np.ndarray           # (n_buckets + 1,) bucket boundaries
    bucket_confidences: np.ndarray     # (n_buckets,) P(correct) per bucket
    n_samples: int                     # Total training samples
    samples_per_bucket: np.ndarray     # Per-bucket sample counts
```

**Key Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `gap_to_conf` | `(gap: float) → float` | Look up confidence for a given score gap |
| `conf_to_gap` | `(conf: float) → float` | Inverse: find gap threshold for a confidence level |
| `save` | `(path: Path)` | Serialize to JSON |
| `load` | `(path: Path) → CalibrationResult` | Deserialize from JSON |

### Pair Extraction Methods

Three methods for generating training data:

#### extract_pairs_adjacent_boundary() — Recommended

```python
def extract_pairs_adjacent_boundary(
    baskets: List[List[int]],
    base_scorer: Callable,
    n_items: int,
    n_baskets: int = 5000,
    max_rank: int = 50,
    rng_seed: int = 42,
) -> List[Tuple[float, bool]]
```

For each adjacent pair (k, k+1) in the base ranking, checks if the engaged item ranks higher. Produces smooth calibration curves rising from ~0.5 to ~0.9+.

#### extract_pairs_pos_neg() — Deprecated

Creates pairs by sampling negative (non-engaged) items against positive (engaged) items. Can produce flat curves when positives are rare in top-K.

#### extract_pairs_from_logs() — Deprecated

Uses historical ranking logs. Issues with flat curves because most adjacent pairs have identical outcomes (both 0).

### Edge Protection Modes

Four strategies for deciding which ordering edges to protect:

#### get_protected_edges_by_budget() — Primary Mode

```python
def get_protected_edges_by_budget(
    base_order: List[int],
    base_scores: Dict[int, float],
    calibration: Optional[CalibrationResult] = None,
    budget_pct: float = 0.30,
    max_rank: int = 50,
    rank_bands: Optional[List[Tuple[int, int, float]]] = None,
    normalize_gaps: bool = False,
    sentinel_k: Optional[int] = None,
) -> List[int]
```

**Algorithm**:
1. Compute score gaps for all adjacent pairs in top-N
2. If calibration is available, map gaps to confidences; otherwise use raw gaps as proxy
3. Sort edges by confidence descending
4. Protect the top `budget_pct`% most confident edges

**Parameters**:

| Parameter | Description |
|-----------|-------------|
| `budget_pct` | Fraction of edges to protect. `0.30` = protect top 30% most confident. |
| `calibration` | If None, falls back to raw gap-based protection. |
| `rank_bands` | Per-region budgets, e.g., `[(0, 10, 0.5), (10, 30, 0.3), (30, 50, 0.2)]` for head/mid/tail. |
| `normalize_gaps` | Divide gaps by local standard deviation to handle score dispersion. |
| `sentinel_k` | Hard cap: items below position k cannot exceed the sentinel item's score. |

#### get_protected_edges() — Classic Threshold Mode

```python
def get_protected_edges(
    base_order: List[int],
    base_scores: Dict[int, float],
    calibration: CalibrationResult,
    rho: float = 0.90,
) -> List[int]
```

Protects edge k if `calibration.gap_to_conf(gap_k) >= rho`. Simple, but suffers from "cliff problem" when confidence values cluster (e.g., all edges at 0.995).

#### get_protected_edges_by_percentile() — Fallback

```python
def get_protected_edges_by_percentile(
    base_order: List[int],
    base_scores: Dict[int, float],
    protect_top_pct: float = 0.20,
    max_rank: int = 20,
) -> List[int]
```

Protects edges by gap percentile. No calibration needed.

#### get_protected_edges_adaptive() — Auto-Selection

```python
def get_protected_edges_adaptive(
    base_order, base_scores,
    calibration=None, rho=None,
    budget_pct=0.30, max_rank=50,
    mode="budget",
    sentinel_k=None,
) -> Tuple[List[int], str]
```

Automatically chooses strategy:
- `"threshold"`: Use `rho`-based protection (for well-calibrated rankers)
- `"budget"`: Use budget-based protection (for clustered confidences)
- `"auto"`: Use threshold if calibration has good spread (range > 0.3), else budget

Returns both the protected edges and a string indicating which mode was used.

### Design Decision: Budget Over Threshold

Empirical finding: Item-CF and many collaborative filtering rankers produce calibration curves with a cliff structure (all confidences > 0.55), making threshold-based protection binary. Budget-based protection provides smooth, monotonic control regardless of the calibration shape.

```
Budget=0.10 → protect 5 edges  → stability 0.79 → max steering
Budget=0.30 → protect 15 edges → stability 0.89 → sweet spot
Budget=0.50 → protect 25 edges → stability 0.94 → conservative
Budget=1.00 → protect all      → stability 0.97 → base order locked
```

---

## 5. Isotonic Projection — Constrained Optimization

**Module**: `mosaic/isotonic_projection.py`

### Problem Formulation

Given target scores `t` (base + orthogonalized steering) and a set of protected edges `E`, find final scores `z` that:

```
Minimize:  Σ_i  w_i (z_i − t_i)²
Subject to:  z_{π(k)} ≥ z_{π(k+1)}   ∀ k ∈ E_protected
```

where `π` is the base ranking order. The solution preserves protected orderings while staying as close as possible to the target scores.

### Algorithm: Pool-Adjacent-Violators (PAV)

#### isotonic_project_on_runs()

```python
def isotonic_project_on_runs(
    base_order: List[int],        # items in base rank order
    target_scores: Dict[int, float],  # t_i per item
    protected_edges: List[int],   # edge indices to protect
    weights: Optional[Dict[int, float]] = None,  # per-item weights
) -> ProjectionResult
```

**Steps**:

1. **Identify runs**: Find maximal sequences of consecutive protected edges. If edges {2, 3, 4, 7, 8} are protected, the runs are {2–4} and {7–8}. Each run is solved independently.

2. **Weighted PAV on each run**:
   - Initialize each element as its own block: `(sum = w_i·t_i, weight = w_i)`
   - Left-to-right scan: if `mean(block_i) < mean(block_{i+1})`, the monotonicity constraint is violated
   - Merge violating blocks: `new_sum = sum_a + sum_b`, `new_weight = weight_a + weight_b`, `new_mean = sum/weight`
   - Repeat until no violations remain

3. **Set final scores**: All items in a merged block get the block's weighted mean.

**Complexity**: O(N) per run — each element is merged at most once.

#### ProjectionResult

| Field | Type | Description |
|-------|------|-------------|
| `z` | `Dict[int, float]` | Final scores after projection |
| `n_constraints` | `int` | Number of protected edges |
| `n_active_constraints` | `int` | Constraints that were actually binding (blocks with > 1 element) |
| `pooled_blocks` | `List[List[int]]` | Items that were pooled together |
| `pooled_block_positions` | `List[Tuple[int, int]]` | Index ranges of pooled blocks |
| `n_pre_violations` | `int` | Protected edges violated before projection |

#### compute_final_ranking()

```python
def compute_final_ranking(
    z: Dict[int, float],
    base_order: List[int],
) -> List[int]
```

Sorts by `(-z_i, base_position_i)` for:
- **Primary**: descending final score
- **Tiebreaker**: ascending base position (stable tie-breaking — if two items get the same projected score, the one that ranked higher originally stays higher)

### Interpretation

- **`n_active_constraints` = 0**: The steering didn't violate any protected edges. All items moved freely.
- **`n_active_constraints` > 0**: Some items wanted to swap past protected edges but were constrained. The pooled blocks show which items were forced to share scores.
- **`pooled_blocks` with many items**: Heavy constraint binding — the steering signal strongly disagrees with the base ranker in that region.

---

## 6. Validated Results (Included Notebooks)

All results below are **reproducible** from the notebooks included in this repository. Run them yourself to verify.

### Content Moderation (`notebooks/content_moderation.ipynb`)

200 synthetic posts with a realistic engagement-toxicity correlation (r = 0.424). Toxic content is engaging — outrage drives clicks.

| Method | Toxic in top-10 | Mean toxicity (top-10) | Kendall tau vs base |
|--------|----------------|----------------------|-------------------|
| Base | 7 | 0.339 | 1.000 |
| Naive (subtract penalty) | 2 | 0.250 | 0.438 |
| MOSAIC (budget=0.30) | 5 | 0.280 | 0.510 |

At budget=0.00 (maximum steering), MOSAIC matches naive's toxicity reduction with better quality retention (tau 0.456 vs 0.438).

**Budget sweep** (top-10 metrics):

| Budget | Toxic/10 | Mean Toxicity | Tau | Quality |
|--------|---------|--------------|-----|---------|
| 0.00 | 4 | 0.268 | 0.456 | 72.8% |
| 0.10 | 4 | 0.268 | 0.456 | 72.8% |
| 0.20 | 5 | 0.280 | 0.510 | 75.5% |
| 0.30 | 5 | 0.280 | 0.510 | 75.5% |
| 0.50 | 6 | 0.308 | 0.624 | 81.2% |
| 0.70 | 7 | 0.339 | 0.827 | 91.3% |
| 1.00 | 7 | 0.339 | 1.000 | 100.0% |

### Fraud Detection (`notebooks/fraud_detection.ipynb`)

300 simulated transactions ($5–$17,660 range, 19.3% fraud rate). The base model ranks by fraud probability but ignores transaction value.

| Method | Fraud value in top-20 | Improvement |
|--------|----------------------|-------------|
| Base | $552 | — |
| MOSAIC (budget=0.30) | $2,271 | 4.1x |

**Tiered gating** (BLOCK = top 10%, REVIEW = next 20%, ALLOW = bottom 70%):

| Tier | Method | Precision | Fraud Value |
|------|--------|-----------|-------------|
| BLOCK | Base | 40% | $154 |
| BLOCK | MOSAIC | 40% | $1,603 |

MOSAIC captures 10.4x more fraud value in the BLOCK tier at the same precision. Fraud slipping through ALLOW: Base $4,088, MOSAIC $767 (81% reduction).

### Fairness — COMPAS (`notebooks/fairness_compas.ipynb`)

100 defendants sampled from the ProPublica COMPAS dataset (66 African-American, 34 Caucasian).

| Metric | Base | MOSAIC (budget=0.30) |
|--------|------|---------------------|
| Adverse Impact Ratio | 0.773 | 0.916 |
| 4/5ths rule | FAILS | PASSES |
| Kendall tau | — | 0.900 |
| Quality retained | — | 95.0% |

The budget sweep shows AIR stable above 0.80 from budget 0.00–0.70, with quality increasing monotonically.

### Objective Discovery (`notebooks/objective_discovery.ipynb`)

500 synthetic articles across 7 categories with 10,000 simulated reads. Tests 7 candidate policies to discover which objectives align with user preferences.

| Policy | Preference Lift | ΔEntropy | ΔReads |
|--------|----------------|----------|--------|
| quality_depth | 1.90x | +0.170 | +494 |
| trending | 2.71x | −0.175 | +686 |
| diversity_underexposed | 0.25x | +0.281 | −310 |

**Key finding**: `quality_depth` is the only policy that achieves BOTH preference alignment (lift > 1) AND diversity gain (positive entropy delta). Forcing diversity directly (underexposed categories) fights user preferences. Quality-based steering achieves diversity as a side effect.

**Portfolio frontier**: Pure quality steering (0% trending / 100% quality) is the only mix that achieves both more reads and higher diversity than the base ranking.
