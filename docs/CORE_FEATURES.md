# governed-rank: Core Features Documentation

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [govern() — Zero-Config Reranking](#2-govern--zero-config-reranking)
3. [Orthogonalization — Interference Removal](#3-orthogonalization--interference-removal)
4. [Gap Calibration — Confidence Learning](#4-gap-calibration--confidence-learning)
5. [Isotonic Projection — Constrained Optimization](#5-isotonic-projection--constrained-optimization)
6. [MOSAICScorer — Full Pipeline Orchestration](#6-mosaicscorer--full-pipeline-orchestration)
7. [Hybrid Moment Activation](#7-hybrid-moment-activation)
8. [Satiation — Diminishing Returns](#8-satiation--diminishing-returns)
9. [Steering Guardrails — Risk Management](#9-steering-guardrails--risk-management)
10. [Objective Discovery Engine](#10-objective-discovery-engine)
11. [LENS — Experience-Neutral Ad Insertion](#11-lens--experience-neutral-ad-insertion)
12. [Moment Discovery (NMF)](#12-moment-discovery-nmf)
13. [Evidence Graph — Feature-to-Moment Learning](#13-evidence-graph--feature-to-moment-learning)
14. [Moment Priors](#14-moment-priors)
15. [Moment Drift Detection](#15-moment-drift-detection)
16. [Exploration Pool](#16-exploration-pool)
17. [Counterfactual Evaluation](#17-counterfactual-evaluation)
18. [Rank Protection (Legacy)](#18-rank-protection-legacy)
19. [Validated Results](#19-validated-results)

---

## 1. Architecture Overview

governed-rank implements a 7-stage reranking pipeline that steers any ranked list toward policy objectives while preserving the base ranker's confident decisions.

```
Stage A  Moment Activation      →  p(m | context)
Stage B  Candidate Recall       →  accuracy pool + moment pool + exploration
Stage C  Base Scoring           →  s_i = S_base(i | context)
Stage D  Control Utility        →  u_i = λ_m · align_i · sat_i + policy_i
Stage E  Orthogonalization      →  u_⊥ = u − proj(u onto s)
Stage F  Protected Edges        →  where gap_to_conf(Δ) ≥ ρ  or  top B% by confidence
Stage G  Constrained Projection →  isotonic regression on protected runs
```

**Core insight**: Naively adding a policy signal to a base ranker degrades accuracy (e.g., −4.87% Recall@10 in experiments). governed-rank solves this by (1) removing the component of the policy signal that correlates with the base score, and (2) locking the base ranker's most confident ordering decisions via budget-controlled constraints.

**Two entry points**:
- `govern()` — zero-config, 3 arguments, works with any key type
- `MOSAICScorer` — full pipeline with moments, calibration, satiation, receipts

**Complexity**: O(N log N) sorting + O(N) orthogonalization + O(N) projection = O(N log N) overall.

---

## 2. govern() — Zero-Config Reranking

**Module**: `mosaic/govern.py`

The simplest possible entry point. No moments, no calibration, no config objects.

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
**Pipeline Stage**: E

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
| `steering_utilities` | Item → raw steering utility (mission + policy) |
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
**Pipeline Stage**: F

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
**Pipeline Stage**: G

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

## 6. MOSAICScorer — Full Pipeline Orchestration

**Module**: `mosaic/mosaic_scorer.py`
**Pipeline Stages**: A through G

For production systems that need moment activation, calibrated confidence, satiation, and full diagnostic receipts.

### MOSAICConfig

```python
@dataclass
class MOSAICConfig:
    # Activation
    evidence_beta: float = 1.0           # Evidence graph weight
    temperature_max: float = 2.0         # Max softening temperature

    # Steering
    lambda_m: float = 0.03               # Mission alignment weight
    satiation_floor: float = 0.25        # Min satiation multiplier
    satiation_rate: float = 0.6          # Satiation decay rate
    satiation_top_m: int = 2             # Only satiate top-M moments

    # Protection
    rho: float = 0.90                    # Confidence threshold (threshold mode)
    use_calibration: bool = True         # Use learned calibration?
    fallback_gap_threshold: float = 0.05 # Raw gap threshold if no calibration
    protection_mode: str = "threshold"   # "threshold" or "budget"
    budget_pct: float = 0.30             # Budget fraction (budget mode)
    max_rank: int = 50                   # Head band size

    # Auto-budget (target stability)
    target_stability: Optional[float] = None  # Desired top-K membership stability
    target_top_k: int = 10                    # K for stability measurement
    auto_budget_steps: Tuple = (0.30, 0.50, 0.70, 1.00)
    auto_budget_max_passes: int = 2
    auto_budget_use_sentinel: bool = False
    sentinel_k: Optional[int] = None

    # Risk guardrails (VaR/CVaR on displacement)
    risk_var_quantile: Optional[float] = None   # e.g., 0.95
    risk_max_displaced: Optional[float] = None  # Max top-K items displaced
    risk_cvar_quantile: Optional[float] = None
    risk_cvar_max: Optional[float] = None
    risk_target_top_k: int = 10
    risk_budget_table: Optional[List[Dict]] = None  # Pre-computed lookup
    risk_budget_default: Optional[float] = None

    # Orthogonalization
    rescale_after_ortho: bool = False

    # Diagnostics
    log_receipts: bool = True
```

### Usage

```python
from mosaic import MOSAICScorer, MOSAICConfig, CalibrationResult

calibration = CalibrationResult.load("models/gap_calibration.json")

scorer = MOSAICScorer(
    moment_affinities=moment2vec,        # (n_items, K) matrix
    calibration=calibration,
    config=MOSAICConfig(
        lambda_m=0.03,
        rho=0.90,
        protection_mode="budget",
        budget_pct=0.30,
    ),
)

result = scorer.rank(
    candidates=[1, 2, 3, 4, 5],
    base_scores={1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5},
    activation_p=np.array([0.7, 0.1, 0.1, 0.05, 0.05]),
    activation_confidence="high",
    cart_items=[10, 20],
    policies=[{"item_id": 3, "boost": 0.02}],
)
```

### rank() Method

```python
def rank(
    self,
    candidates: List[int],
    base_scores: Dict[int, float],
    activation_p: np.ndarray,                 # p(m|context) from activation
    activation_confidence: str = "medium",    # "high", "medium", "low"
    cart_items: List[int] = None,
    policies: List[Any] = None,
    fired_features: List[str] = None,
    feature_contributions: Dict[str, float] = None,
) -> MOSAICResult
```

**Internal pipeline**:

1. **Stage C — Base Order**: Sort candidates by `base_scores` descending.
2. **Stage D — Control Utility**:
   - Mission alignment: `A[i] · p` (dot product of item's moment affinities with activation)
   - Satiation factor from cart contents (top-M moments only)
   - Confidence scaling: high=1.0, medium=0.6, low=0.0
   - Combine: `u_mission = λ_m × alignment × satiation × confidence_scale`
   - Add policy boosts: `u_i = u_mission + u_policy`
3. **Stage E — Orthogonalization**: `orthogonalize_against_base(base_scores, u)` → `u_⊥`
4. **Stage F — Protected Edges**: Apply selected protection mode (budget or threshold)
5. **Stage G — Constrained Projection**: `isotonic_project_on_runs()` → final scores `z`
6. **Final Ranking**: `compute_final_ranking(z, base_order)`

### Auto-Budget

When `target_stability` is set, the scorer iteratively increases the budget until the desired top-K membership stability is achieved:

1. Start with `budget_pct`
2. Measure stability: `|top_K_base ∩ top_K_final| / K`
3. If below target, escalate to the next step in `auto_budget_steps`
4. Up to `auto_budget_max_passes` attempts
5. Optionally applies sentinel cap on pass 2

### Risk Guardrails

VaR/CVaR-based budget selection:
- Pre-compute a lookup table mapping `(budget, VaR_quantile) → max_displaced`
- At serving time, select the minimum budget that satisfies the displacement risk bound
- Example: VaR95 ≤ 2 displaced items → budget ≥ 0.50

### MOSAICReceipt

Every ranked item gets a receipt with full diagnostics:

| Field | Description |
|-------|-------------|
| `activation_p` | Moment distribution for this request |
| `activation_confidence` | Confidence tier |
| `fired_features` | Evidence features that fired |
| `base_score` | Original score |
| `mission_alignment` | `A[i] · p` |
| `satiation_factor` | Diminishing returns multiplier |
| `policy_boost` | Operator-specified boost |
| `raw_steering_utility` | `u_i` before orthogonalization |
| `orthogonalized_utility` | `u_⊥_i` after orthogonalization |
| `target_score` | `t_i = s_i + u_⊥_i` |
| `final_score` | `z_i` after projection |
| `was_constrained` | Was this item in a pooled block? |
| `base_rank` / `final_rank` | Position change |
| `primary_reason` | "policy" / "moment" / "history" / etc. |

### MOSAICResult

| Field | Description |
|-------|-------------|
| `ranked_items` | Final reranked order |
| `scores` | Final `z_i` per item |
| `receipts` | `Dict[int, MOSAICReceipt]` per item |
| `n_protected_edges` | Total constraints |
| `n_active_constraints` | Binding constraints |
| `projection_coeff` | Orthogonalization diagnostic |
| `target_stability` / `achieved_stability` | Auto-budget metrics |
| `budget_used` | Actual budget after auto-escalation |
| `risk_budget_used` | Budget from risk guardrail lookup |

---

## 7. Hybrid Moment Activation

**Module**: `mosaic/activation.py`
**Pipeline Stage**: A

Computes `p(m | context)` — the probability distribution over moments (latent intents) given the current context (cart, history, time, features).

### Pipeline

```
Prior (cart/history/population)
    ↓
Time Multipliers (time-of-day adjustments)
    ↓
Evidence Graph (feature → moment weights)
    ↓
Rule Posterior: p_rule ∝ prior · time · exp(β · evidence)
    ↓
Blending Gate (α): decides how much to trust rules vs model
    ↓
Mix: p_mix = (1−α) · p_rule + α · p_model
    ↓
Temperature Softening: T = 1 + (T_max − 1) · (1 − ev)^γ
    ↓
Final: p = normalize(p_mix^(1/T))
```

### ActivationConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 1.0 | Weight for evidence graph contributions |
| `T_max` | 2.5 | Maximum temperature (lower evidence → flatter distribution) |
| `gamma` | 2.0 | Temperature scaling exponent |
| `ev_high` | 0.6 | Evidence volume threshold for "high evidence" |
| `ev_low` | 0.25 | Evidence volume threshold for "low evidence" |
| `agree_high` | 0.7 | Agreement threshold |
| `alpha_strong_rules` | 0.35 | Blend weight when evidence is strong + features agree |
| `alpha_weak_evidence` | 0.75 | Blend weight when evidence is weak (trust model) |
| `alpha_strong_model` | 0.70 | Blend weight when model is confident + has history |
| `alpha_default` | 0.50 | Default blend weight |

### Blending Gate

The gate decides how much to trust rules vs. a learned model:

| Condition | α | Reason |
|-----------|---|--------|
| High evidence + high agreement | 0.35 | Strong rules → mostly rule-based |
| Weak evidence | 0.75 | Little signal → trust the model |
| Model confident + has history | 0.70 | Model has a strong opinion → lean model |
| Default | 0.50 | Even split |

### Evidence Volume

```
ev = sqrt(cart_signal × feature_signal)
```

- **cart_signal**: From how informative the cart items are
- **feature_signal**: From how many features fired and their agreement

### Confidence Tiering

Four signals are checked: `margin ≥ 0.25`, `peakedness ≥ 0.35`, `ev ≥ 0.4`, `agreement ≥ 0.6`.

| Tier | Signals True | Steering Effect |
|------|-------------|-----------------|
| `high` | ≥ 3 | Full steering: `conf_scale = 1.0` |
| `medium` | 2 | Partial: `conf_scale = 0.6` |
| `low` | < 2 | No steering: `conf_scale = 0.0` |

### Temperature Softening

Low-evidence contexts produce flatter distributions (more uncertainty):

```
T = 1 + (T_max − 1) × (1 − ev)^γ

When ev = 0 → T = T_max (very flat)
When ev = 1 → T = 1.0 (no softening)
```

### ActivationResult

Contains full diagnostics: `p` (final distribution), `confidence_tier`, `prior_source`, `alpha`, `gate_reason`, `evidence_volume`, `agreement`, `temperature`, `fired_features`, `evidence_contributions`, `margin`, `peakedness`.

---

## 8. Satiation — Diminishing Returns

**Module**: `mosaic/satiation.py`

### Problem

If a user's cart already contains 3 breakfast items, recommending a 4th has diminishing value. Satiation models this: as a moment becomes "filled" by cart contents, its contribution to steering is reduced.

### Mathematical Model

```
fill_m = mean(A[item, m] for item in cart)    # how filled moment m is
sat_m = floor + (1 − floor) × exp(−rate × fill_m)   for m ∈ top-M active moments
sat_m = 1.0                                          for other moments
```

### Functions

#### compute_moment_fill()

```python
def compute_moment_fill(
    cart_affinities: List[np.ndarray],  # (K,) vector per cart item
    K: Optional[int] = None,
    normalize: str = "mean",             # "mean", "sqrt", or "sum"
) -> np.ndarray  # (K,) fill vector
```

| Normalize | Behavior |
|-----------|----------|
| `"mean"` | Bounded [0, 1]. Recommended for stability. |
| `"sqrt"` | Scales with √(cart_size). Intermediate. |
| `"sum"` | Raw accumulation. Can grow unbounded with large carts. |

#### compute_satiation()

```python
def compute_satiation(
    fill: np.ndarray,                  # (K,) how filled each moment is
    activation: np.ndarray,            # (K,) p(m|context)
    rate: float = 0.6,                 # decay rate
    top_m: int = 2,                    # only satiate top-M active moments
    floor: float = 0.25,               # minimum multiplier
) -> np.ndarray  # (K,) satiation factors
```

**Key properties**:
- `floor = 0.25` means even a fully-saturated moment still contributes 25% of its weight
- Only top-M moments by activation are satiated — prevents "diversifying away" from actual intent
- Other moments have satiation factor = 1.0 (no reduction)

#### satiated_moment_score()

```python
def satiated_moment_score(
    item_affinity: np.ndarray,     # A[item] shape (K,)
    activation: np.ndarray,        # p(m|ctx) shape (K,)
    cart_affinities: Optional[List[np.ndarray]] = None,
    rate: float = 0.6,
    top_m: int = 2,
    floor: float = 0.25,
) -> float
```

Returns: `Σ_m A[item, m] × p[m] × sat[m]`

Batch version `satiated_moment_scores_batch()` handles `(N, K)` item affinities efficiently.

---

## 9. Steering Guardrails — Risk Management

**Module**: `mosaic/steering_guardrails.py`

Automatically throttles policies that hurt key metrics.

### GuardrailConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_impressions_for_full_effect` | 100 | Ramp-up threshold |
| `min_clicks_for_evaluation` | 20 | Minimum data for stop-loss |
| `ctr_drop_threshold` | 0.20 | 20% CTR drop → throttle |
| `atc_drop_threshold` | 0.25 | 25% ATC drop → throttle |
| `conversion_drop_threshold` | 0.30 | 30% conversion drop → **STOP** |
| `throttle_step` | 0.25 | Reduce throttle by 25% each time |
| `min_throttle` | 0.25 | Floor (never go below 25%) |
| `auto_recover` | True | Allow recovery when metrics improve |
| `warmup_hours` | 6 | No evaluation during warmup |

### Throttle Lifecycle

```
Policy deployed → Warmup (6h, no evaluation)
    ↓
Ramp-up → confidence = min(impressions / 100, 1.0)
    ↓
Monitoring → check stop-loss each evaluation cycle
    ↓
If conversion drops ≥ 30% → STOP (throttle = 0.0)
If CTR drops ≥ 20% → throttle down 25%
If ATC drops ≥ 25% → throttle down 25%
    ↓
If auto_recover + metrics improved → throttle up
```

### Effective Boost Computation

```python
effective_boost = requested_boost × confidence_factor × throttle_factor
```

This means a policy's real-world impact ramps up gradually and can be automatically attenuated if it hurts performance.

### PolicyMetrics

Tracks per-policy: impressions, clicks, add-to-carts, purchases (both lifetime and windowed 24h), baseline rates, throttle state, stop status.

---

## 10. Objective Discovery Engine

**Module**: `mosaic/discovery/`

Discovers which policies are naturally aligned with user behavior before you deploy them.

### Core Insight

Instead of guessing which policy to steer toward, run all candidates through a scorecard and measure **Preference Lift**:

```
                 P(item ∈ Policy | user engages with item)
Preference Lift = ─────────────────────────────────────────
                      P(item ∈ Policy | catalog)
```

- **Lift > 1.0**: Users disproportionately engage with policy items (aligned)
- **Lift < 1.0**: Users avoid policy items (misaligned)

### DiscoveryConfig

```python
@dataclass
class DiscoveryConfig:
    min_lift: float = 1.2               # Report as opportunity
    max_lift: float = 0.7               # Report as oversupply
    min_catalog_rate: float = 0.01      # Ignore < 1% categories
    min_reads: int = 30                 # Significance threshold
    enable_time_segments: bool = True   # Morning/evening segments?
    hour_cutoff: int = 12               # Morning if hour < 12
    compute_confidence: bool = False    # Bootstrap CIs?
    n_bootstrap: int = 200
    alpha: float = 0.05
    category_field: str = "category"
```

### Usage

```python
from mosaic.discovery import DiscoveryEngine, DiscoveryConfig

engine = DiscoveryEngine(config=DiscoveryConfig(min_lift=1.2))
report = engine.discover(sessions, catalog, dataset_name="my_data")

# Opportunities: categories users want more of
for opp in report.top_opportunities(5):
    print(f"{opp.category}: {opp.preference_lift:.1f}x lift — {opp.action.value}")

# Oversupply: categories you're over-serving
for over in report.top_oversupply(5):
    print(f"{over.category}: {over.preference_lift:.2f}x lift — {over.action.value}")

# Segment comparison: which categories differ by time of day
diffs = report.segment_comparison("sports")
```

### ActionType Classification

| Action | Lift Range | Description |
|--------|-----------|-------------|
| `PROMOTE` | > 1.5 | Strong opportunity — users want significantly more |
| `BOOST` | 1.2 – 1.5 | Mild opportunity |
| `NEUTRAL` | 0.8 – 1.2 | Balanced supply/demand |
| `REDUCE` | 0.5 – 0.8 | Oversupplied — consider reducing exposure |
| `CUT` | < 0.5 | Severely misaligned — actively avoided by users |

### DiscoveredObjective

Each discovery contains:

| Field | Description |
|-------|-------------|
| `category` | The content category |
| `segment` | "all", "morning", "evening" |
| `preference_lift` | `user_rate / catalog_rate` |
| `catalog_rate` | Base rate in catalog |
| `user_rate` | Observed rate in user behavior |
| `n_reads` | Sample size |
| `action` | Auto-derived `ActionType` |
| `confidence` | HIGH / MEDIUM / LOW |
| `business_insight` | Human-readable summary |

### DiscoveryReport

| Property | Description |
|----------|-------------|
| `opportunities` | All discoveries with lift > `min_lift` |
| `oversupply` | All discoveries with lift < `max_lift` |
| `by_segment` | Grouped by time segment |
| `top_opportunities(n)` | Top-N by lift |
| `segment_comparison(category)` | Compare one category across segments |

### Validated Findings

On Adressa (34 policies tested):

| Category | Result |
|----------|--------|
| Trending | 9.32x lift — strongest alignment |
| Quality | 2.00x lift AND +47% click diversity |
| Business news | 3.00x lift — significantly underexposed |
| Diversity-forcing | 0.25x lift — users actively avoid it |
| Long-tail | 0.01x lift — severely misaligned |

**Key insight**: "Don't optimize diversity directly. Optimize quality and you get diversity for free."

---

## 11. LENS — Experience-Neutral Ad Insertion

**Module**: `mosaic/lens_ads.py`

Inserts sponsored content only at low-confidence recommendation boundaries, preserving user experience.

### Core Idea

MOSAIC's calibration reveals where the ranker is confident and where it's uncertain. Ads are inserted only at uncertain boundaries, where displacing an organic item costs the least in user experience.

### LENSConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_ad_load` | 0.10 | Max 10% of feed can be ads |
| `min_spacing` | 5 | Minimum items between ads |
| `rho_insert` | 0.80 | Only insert where confidence < 0.80 |
| `lambda_ad` | 0.1 | Mission alignment weight for ad selection |
| `boundary_cost_weight` | 0.2 | Penalty for inserting at high-confidence boundaries |
| `position_decay` | 0.95 | CTR decay per position |

### Algorithm

1. Compute boundary confidences using calibration
2. Identify eligible slots where confidence < `rho_insert`
3. Score each ad: `utility = expected_value + λ_ad × alignment`
4. Greedy assignment by `gain = utility × position_decay − boundary_cost`
5. Enforce `max_ad_load` and `min_spacing` constraints
6. Build final feed with interspersed ads

### AdCandidate

| Field | Description |
|-------|-------------|
| `ad_id` | Unique identifier |
| `bid` | Cost per event |
| `p_click` | Predicted CTR |
| `quality_score` | User experience score [0, 1] |
| `throttle` | Pacing throttle |
| `moment_affinities` | Optional moment alignment for targeted insertion |

### Validated Results (Adressa)

| Method | Recall@10 | NDCG@10 | Ad Value | Violations/Session |
|--------|-----------|---------|----------|-------------------|
| No ads | 38.0% | 23.0% | — | — |
| Fixed-slot | 32.9% | 20.6% | 0.100 | 0.83 |
| **LENS** | **34.7%** | **21.6%** | 0.081 | **0.0** |

LENS achieves zero policy violations while recovering 35% of the recall lost to fixed-slot insertion.

---

## 12. Moment Discovery (NMF)

**Module**: `mosaic/moment_nmf.py`

Discovers latent "moments" (shopping intents) from co-purchase patterns using Non-negative Matrix Factorization.

### Pipeline

```
Baskets → Co-occurrence Matrix → NMF → moment2vec
```

### discover_moments_nmf()

```python
def discover_moments_nmf(
    baskets: List[List[int]],
    products: Dict[int, dict],
    k: int = 8,                    # number of moments
    max_iter: int = 300,
    random_state: int = 42,
    init: str = "random",
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[int, int]]
```

1. Build item co-occurrence matrix from baskets
2. NMF factorization: `W @ H ≈ co-occurrence`
3. Normalize W: `moment2vec = W / row_sums` → (n_items, K) affinity matrix

**Returns**: `(moment2vec, H, idx_to_product_id, product_id_to_idx)`

### auto_label_moments()

Heuristic labels by finding the most common aisles/departments for each moment's top items:

```python
labels = auto_label_moments(H, products, idx_to_product_id, top_n=10)
# e.g., ["Party Snacks", "Breakfast Staples", "Fresh Produce", ...]
```

### build_and_save_moment2vec()

Full pipeline that outputs:
```
output/
├── moment2vec.npy          # (n_items, K) affinity matrix
├── H_components.npy        # (K, n_items) moment weights
├── product_id_mapping.json # ID ↔ index mappings
├── moment_labels.json      # Auto-generated labels
└── meta.json               # Stats and checksums
```

---

## 13. Evidence Graph — Feature-to-Moment Learning

**Module**: `mosaic/evidence_graph.py`

Learns sparse weights from contextual features (time of day, device, etc.) to moments using Pointwise Mutual Information.

### Mathematical Formulation

```
P(m) = (moment_count[m] + α) / (n_orders + K·α)
P(m|f) = (feature_moment_count[f, m] + α) / (feature_count[f] + K·α)
w[f, m] = log(P(m|f) / P(m))
```

Positive weight: feature f makes moment m more likely.
Negative weight: feature f makes moment m less likely.

### EvidenceGraphConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weight_threshold` | 0.1 | Minimum |weight| to keep |
| `max_edges_per_feature` | 3 | Sparsity limit per feature |
| `min_feature_support` | 50 | Minimum feature occurrences |
| `smoothing_alpha` | 1.0 | Laplace smoothing |
| `max_weight` / `min_weight` | 3.0 / −3.0 | Clamping bounds |

### build_evidence_graph()

```python
def build_evidence_graph(
    orders: List[Tuple[List[int], FrozenSet[str]]],  # (items, features) per order
    moment2vec: np.ndarray,                           # (n_items, K)
    config: Optional[EvidenceGraphConfig] = None,
) -> EvidenceGraph
```

1. For each order: compute order profile = mean(A[item] for item in order)
2. Accumulate profiles to moment counts and feature-moment counts
3. Compute PMI weights
4. Filter by threshold, support, max edges
5. Clamp to [min_weight, max_weight]

### EvidenceGraph

| Method | Description |
|--------|-------------|
| `get_weight(feature, moment)` | Single weight lookup |
| `get_feature_weights(feature)` | All moment weights for a feature |
| `get_moment_features(moment)` | All features that affect a moment |
| `to_dict()` / `from_dict()` | Serialization |

### Default Evidence Graph

`create_default_evidence_graph(K)` provides hand-coded bootstrap rules:

```python
("chips_present", {0: 0.5})        # Party moment
("breakfast_present", {1: 0.7})    # Breakfast moment
("hour_morning", {1: 0.3})        # Morning features boost breakfast
```

---

## 14. Moment Priors

**Module**: `mosaic/moment_priors.py`

### Prior Hierarchy

| Priority | Prior | Condition |
|----------|-------|-----------|
| 1 | **Cart** | Cart is non-empty |
| 2 | **History** | User has browsing/purchase history |
| 3 | **Population** | Fallback — average across all orders |

### Functions

**compute_population_prior()**
```python
P_m = mean(mean(A[item] for item in order) for order in all_orders)
```

**compute_cart_prior()**
```python
p_cart = mean(A[item] for item in cart)
```

**compute_history_prior()**
```python
weight[j] = decay^(H − j − 1)   # most recent items weighted highest
p_history = normalize(Σ weight[j] × A[history[j]])
```

**compute_combined_prior()**

When all three are available:
- 60% cart + 30% history + 10% population (cart present)
- 70% history + 30% population (no cart)
- 100% population (no history or cart)

---

## 15. Moment Drift Detection

**Module**: `mosaic/moment_drift.py`

Detects when moments silently change meaning between NMF rebuilds.

### MomentSignature

Captures the semantic fingerprint of a moment:
- Top departments/aisles distribution
- Centroid checksum (hash)
- Number of items with affinity ≥ 0.3
- Mean affinity

### Drift Computation

| Metric | Method |
|--------|--------|
| `department_drift` | KL-divergence-like between old/new department distributions |
| `aisle_drift` | Same for aisles |
| `centroid_drift` | 0 if checksum matches, 1 if different |
| `overall_drift` | Weighted average of above |

### Status Classification

| Status | Threshold | Action |
|--------|-----------|--------|
| `stable` | < 0.25 | No action needed |
| `drifted` | 0.25 – 0.50 | Review and potentially relabel |
| `quarantined` | ≥ 0.50 | Do not serve — moment meaning has changed substantially |

### check_drift_on_rebuild()

Compares signatures of old vs. new moment2vec after an NMF rebuild. Reports per-moment drift status.

---

## 16. Exploration Pool

**Module**: `mosaic/exploration_pool.py`

Adds cold-start items and under-explored moments to the candidate set.

### ExplorationConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `explore_rate` | 0.05 | Fraction of total pool size |
| `cold_start_impressions` | 100 | Items below this → cold-start |
| `cold_start_fraction` | 0.5 | Fraction of explore pool for cold-start |
| `moment_explore_fraction` | 0.5 | Fraction for under-explored moments |
| `low_confidence_boost` | 1.5 | Increase explore rate when confidence is low |

### Algorithm

1. Pool size = `explore_rate × (|history_pool| + |collaborative_pool|)`
2. Boost explore rate by 1.5x if confidence tier is "low"
3. Cold-start items: sampled with weight ∝ `1/(impressions + 1)` — items with fewer impressions get more chances
4. Under-explored moments: items with high affinity to the bottom-third moments by activation get boosted

---

## 17. Counterfactual Evaluation

**Module**: `mosaic/counterfactual.py`

Logs alternative rankings alongside the served ranking for offline impact measurement.

### Counterfactuals Generated

| Variant | Description |
|---------|-------------|
| `pure_model` | No steering at all — `z = s` |
| `no_mission` | Policy boosts only — `z = s + u_policy` |
| `no_policy` | Mission steering only — `z = s + u_mission` |
| `alpha_variants` | Different blend weights between mission and policy |

### Metrics Computed

| Metric | Description |
|--------|-------------|
| `rank_correlation` | Kendall-tau-like correlation between served and counterfactual |
| `rank_overlap` | Jaccard similarity of top-N sets |
| `mean_abs_change` | Average absolute rank change |
| `mission_impact` | Rank changes attributable to mission steering |
| `policy_impact` | Rank changes attributable to policy boosts |
| `head_stability` | How much the top-10 changed |

---

## 18. Rank Protection (Legacy)

**Module**: `mosaic/rank_protection.py`

Position-based protection gate from the SRG-CC approach. Superseded by gap-based protection (Stages F/G) but retained as a fallback.

### Smoothstep Gate

```
For rank ≤ decay_start:         weight = 0.0  (fully protected)
For decay_start < rank < decay_end:  weight = smoothstep((rank − start) / (end − start))
For rank ≥ decay_end:           weight = 1.0  (fully exposed)

smoothstep(x) = x² × (3 − 2x)    (smooth S-curve, no discontinuities)
```

Default: ranks 1–5 fully protected, ranks 5–12 gradual transition, ranks 12+ fully exposed.

### Limitation

This approach protects by position, not by confidence. If rank 3 has a tiny gap to rank 4, it's still fully protected. Gap-based protection (the primary mode) is more principled.

---

## 19. Validated Results

### Cross-Dataset Summary (17 datasets, 6 domains)

| Domain | Datasets | Key Metric |
|--------|----------|------------|
| Recommendations | Ta Feng, Instacart, MovieLens, H&M, LastFM, Adressa, Yoochoose | 0.890 stability @ 0.344 exposure |
| Fairness | COMPAS, Adult Income, German Credit | adverse_impact_ratio = 0.963 |
| Healthcare | MIMIC-IV, SynPUF | 71.6% HIGH tier, 5.0x NMF lift |
| Content Moderation | Civil Comments | 3.4x more efficient on borderline content |
| Fraud | IEEE-CIS, PaySim | 3x higher precision on hard blocks |
| Trust-Gated RAG | RAGWall, BEIR NQ, Tensor Trust | 95–100% attack reduction |

### Policy Exposure Lifts

| Dataset | Best Lift | Policy |
|---------|-----------|--------|
| H&M | 41.7x | Product group steering |
| Instacart | 15.4x | Category (produce) |
| Adressa | 3.58x | Temporal (morning) |
| Ta Feng | 2.94x | Moment alignment |
| MovieLens | 1.15x | Genre (documentary) |

### Budget Monotonicity (5 seeds, 1000 baskets each)

| Budget | Stability | PolicyExp@50 | Recall@10 |
|--------|-----------|-------------|-----------|
| 0.0 | 0.787 | 0.351 | 0.165 |
| 0.3 | 0.890 | 0.349 | 0.164 |
| 0.5 | 0.938 | 0.340 | 0.165 |
| 0.7 | 0.958 | 0.327 | 0.164 |
| 1.0 | 0.973 | 0.312 | 0.165 |

Spearman(budget, stability) = 1.00 ± 0.00 across seeds.

### Content Moderation (Civil Comments, 3.8M)

- 36–59% toxicity reduction while retaining 97–99% engagement
- 3.4x more efficient than naive safety on borderline content
- 30x more efficient on high-correlation platforms

### Fairness (COMPAS)

- Adverse impact ratio: 0.963 (above 4/5ths rule threshold)
- Quality retained: 98.65%
- 1.5pp more quality than group thresholding at parity

### Key Experimental Conclusions

1. Budget cleanly controls stability — λ_m only affects alignment magnitude
2. Recall is preserved across configurations (orthogonalization prevents interference)
3. Stronger rankers need less protection (larger score gaps → fewer violations)
4. Coverage bonus: protection constraints increase item diversity as a side effect
5. Quality > diversity for diversity: quality-based steering achieves 2.00x lift AND +47% click diversity
6. Don't force diversity directly: diversity-forcing policies achieve only 0.25x lift
