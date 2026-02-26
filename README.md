# MOSAIC (governed-rank)

**Margin-Orthogonal Mission Steering for Accuracy-Preserving, Controllable Recommendations**

## Overview

MOSAIC is a recommendation algorithm that treats shopping missions as a **control layer**, not an accuracy layer. The base ranker owns predictive relevance; missions provide interpretable steering, diversity/satiation, and operator controls—without regressing accuracy.

### Key Insight

Naively adding mission scores to a strong collaborative ranker hurts accuracy (e.g., -4.87% Recall@10 in our experiments). MOSAIC solves this with two innovations:

1. **Interference Removal (Orthogonalization):** Remove the component of steering utility that aligns with the base score direction
2. **Confidence-Preserving Constraints:** Protect only confident base decisions using a calibrated gap-to-confidence mapping

## Installation

```bash
pip install -e .
```

Or from PyPI (once published):

```bash
pip install governed-rank
```

## Algorithm Stages

```
A) Moment Activation     -> p(m | context)
B) Candidate Recall      -> accuracy pool + moment pool + exploration
C) Base Scoring          -> s_i = S_base(i | ctx)
D) Control Utility       -> u_i = lambda_m * align_i * sat_i + policy_i
E) Orthogonalization     -> u_perp = u - proj(u onto s)
F) Protected Edges       -> where gap_to_conf(delta) >= rho
G) Constrained Projection -> isotonic regression on protected runs
```

## Quick Start

```python
from mosaic import MOSAICScorer, MOSAICConfig, CalibrationResult
import numpy as np

# Load moment affinities (item -> moment)
A = np.load("models/moment2vec.npy")

# Load calibration (or None for fallback mode)
calibration = CalibrationResult.load("models/gap_calibration.json")

# Create scorer
scorer = MOSAICScorer(
    moment_affinities=A,
    calibration=calibration,
    config=MOSAICConfig(
        lambda_m=0.03,
        rho=0.90,
    ),
)

# Rank candidates
result = scorer.rank(
    candidates=[1, 2, 3, 4, 5],
    base_scores={1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5},
    activation_p=np.array([0.7, 0.1, 0.1, 0.05, 0.05]),
    activation_confidence="high",
    cart_items=[10, 20],
)

print(result.ranked_items)
print(f"Protected {result.n_protected_edges} edges, {result.n_active_constraints} active")
```

## Project Structure

```
mosaic/
├── src/mosaic/                    # Installable Python package
│   ├── __init__.py                # Public API exports
│   ├── mosaic_scorer.py           # Full pipeline orchestration
│   ├── orthogonalization.py       # Score-space interference removal
│   ├── gap_calibration.py         # Learn gap->confidence mapping
│   ├── isotonic_projection.py     # PAV on protected runs
│   ├── activation.py              # Hybrid moment activation pipeline
│   ├── satiation.py               # Diminishing returns
│   ├── rank_protection.py         # Head protection
│   ├── steering_guardrails.py     # Policy safety
│   ├── moment_nmf.py              # NMF moment discovery
│   ├── lens_ads.py                # LENS ad insertion
│   ├── discovery/                 # Objective discovery engine
│   │   ├── api.py
│   │   ├── engine.py
│   │   └── models.py
│   └── _compat/                   # Patterna-dependent modules
│       ├── loaders.py
│       └── state.py
├── scripts/                       # Experiment and evaluation scripts
├── tests/                         # Unit tests
├── models/                        # Result JSON files
├── paper/                         # LaTeX sources, figures
├── docs/                          # Patent, algorithm docs
├── pyproject.toml                 # Package configuration
├── requirements.txt               # Pinned dependencies
├── LICENSE                        # Apache 2.0
└── CHANGELOG.md
```

## Evaluation Results (Ta Feng)

**Dataset:** Ta Feng (Taiwan grocery), 11,208 items, 99,485 baskets
**Evaluation:** 500 baskets, hold-out prediction task
**Baseline:** Item-CF (co-occurrence similarity)

| Method | Recall@10 | NDCG@10 | Displaced | Alignment |
|--------|-----------|---------|-----------|-----------|
| Item-CF (baseline) | 16.20% | 13.94% | 0.00 | 0.207 |
| Naive (base + moment) | 16.20% | 13.91% | 0.39 | 0.210 |
| Rank Protected (Patterna v1) | 15.95% | 13.84% | 0.69 | 0.209 |
| **MOSAIC** | **16.20%** | 13.91% | 0.39 | 0.210 |

**Key Finding:** MOSAIC preserves accuracy (matches Item-CF baseline at 16.20% Recall@10) while providing interpretable mission steering and operator controls.

## License

Apache 2.0 — see [LICENSE](LICENSE).
