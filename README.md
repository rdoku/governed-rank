# governed-rank

**Governed reranking for any domain — steer ranked lists toward policy objectives without breaking accuracy.**

## The Problem

You have a ranked list (search results, recommendations, content feed) and a policy objective (reduce toxicity, increase fairness, promote margin). Naively injecting policy scores breaks your base ranker's accuracy. You need a principled way to steer without regressing.

## Install

```bash
pip install governed-rank
```

## Quick Start

```python
from mosaic import govern

result = govern(
    base_scores={"doc1": 0.9, "doc2": 0.8, "doc3": 0.7, "doc4": 0.6, "doc5": 0.5},
    steering_scores={"doc1": -0.5, "doc2": 0.3, "doc3": 0.8, "doc4": 0.1, "doc5": 0.6},
    budget=0.3,
)

print(result.ranked_items)   # reranked order
print(result.receipts)       # per-item audit trail
```

`budget` controls how much of the original ordering is protected. `0.3` means the top 30% most confident base decisions are locked — steering can only move items around where the base ranker isn't sure.

## Domain Examples

### Content Moderation — demote toxic content without hurting engagement

```python
result = govern(
    base_scores=engagement_scores,       # relevance / predicted engagement
    steering_scores=toxicity_penalties,   # negative = toxic, from your classifier
    budget=0.3,
)
# Toxic items drop in ranking. Engagement-critical ordering preserved.
```

### Fairness — boost underrepresented groups without sacrificing quality

```python
result = govern(
    base_scores=quality_scores,          # hiring model / credit scores / quality
    steering_scores=fairness_boosts,     # positive for underrepresented candidates
    budget=0.3,
)
# Fair reranking with auditable receipts. See notebooks/fairness_compas.ipynb.
```

### RAG Safety — steer retrieval toward grounded, policy-safe documents

```python
result = govern(
    base_scores=retrieval_scores,        # embedding similarity from your vector DB
    steering_scores=groundedness_scores,  # factuality / policy compliance signal
    budget=0.5,
)
# Grounded docs promoted. Top retrieval results still relevant.
```

## How It Works

Three steps, fully automatic:

```
1. Orthogonalize    Remove the component of your policy signal that correlates
                    with the base ranker (so steering can't accidentally hurt accuracy)

2. Protect edges    Lock the most confident base ordering decisions (controlled by budget)

3. Project          Isotonic regression on the remaining items — maximize policy
                    effect while respecting constraints
```

Mathematically: the steering signal is projected into the null space of the base score direction, then a constrained isotonic projection enforces protected ordering decisions. The result is Pareto-optimal — you cannot get more policy effect without giving up more accuracy.

## Validated On

Reproducible results from the included notebooks:

| Notebook | Domain | Key Result |
|----------|--------|------------|
| [`fairness_compas.ipynb`](notebooks/fairness_compas.ipynb) | Fairness (COMPAS) | AIR 0.773 → 0.916, quality 95% |
| [`content_moderation.ipynb`](notebooks/content_moderation.ipynb) | Content feeds | Toxicity reduction with smooth budget tradeoff |
| [`fraud_detection.ipynb`](notebooks/fraud_detection.ipynb) | Fraud review queues | 4.1x fraud value captured, 81% less slippage |
| [`rag_safety.ipynb`](notebooks/rag_safety.ipynb) | RAG safety | Injected docs removed from top-3, quality 65% |
| [`objective_discovery.ipynb`](notebooks/objective_discovery.ipynb) | Policy selection | Quality steering → engagement AND diversity |

Run any notebook to verify. All use synthetic or public data — no external dependencies.

## Core Modules

| Module | Purpose |
|--------|---------|
| `mosaic.govern` | `govern()` entry point — orthogonalize, protect, project |
| `mosaic.orthogonalization` | Score-space interference removal |
| `mosaic.gap_calibration` | Gap → confidence mapping, edge protection |
| `mosaic.isotonic_projection` | Constrained isotonic regression (PAV) |

## License

Apache 2.0 — see [LICENSE](LICENSE).
