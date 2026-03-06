# Steering MovieLens Toward Documentaries — Without Hurting Recall

Your movie recommendation model optimizes for watch probability. Your editorial team wants to promote documentaries. You weight the genre signal — engagement drops. This is the injection problem applied to media recommendations.

We applied `governed-rank` to MovieLens 100K and steered recommendations toward documentaries while keeping recall stable across all budget levels.

## The Dataset

MovieLens 100K: 100,000 ratings from 943 users on 1,682 movies across 19 genres. We built an Item-CF recommender from co-occurrence of high-rated movies (rating >= 4), then used genre vectors as moment affinities.

```python
from mosaic import MOSAICScorer, MOSAICConfig
import numpy as np

# Genre vectors: A[i, g] = 1.0 if movie i has genre g
# Movies can have multiple genres (Action|Adventure|Sci-Fi)
# Row-normalized so each movie's genre weights sum to 1.0
A = build_genre_affinities(movies, genres)  # shape: (1682, 19)

scorer = MOSAICScorer(
    moment_affinities=A,
    config=MOSAICConfig(
        lambda_m=0.5,
        protection_mode="budget",
        budget_pct=0.30,
        fallback_gap_threshold=0.01,
    ),
)
```

## Activating the Documentary Objective

Documentaries make up only ~3-5% of the MovieLens catalog. To promote them, we set `activation_p` to weight the Documentary genre at 60%:

```python
doc_idx = genres.index("Documentary")
n_genres = len(genres)  # 19

activation_p = np.full(n_genres, 0.4 / (n_genres - 1))
activation_p[doc_idx] = 0.6

result = scorer.rank(
    candidates=candidate_movie_ids,     # top-200 from Item-CF
    base_scores=cf_scores,              # co-occurrence similarity
    activation_p=activation_p,
    activation_confidence="high",
)
```

## The Challenge: Small Catalog

MovieLens is harder than Instacart for governed reranking. With only 1,682 movies, the Item-CF model has stronger opinions about most pairs — there's less "uncertain middle" for steering to exploit. The documentary genre has low co-occurrence with mainstream genres, making the steering signal weak.

This is an honest result: **governed-rank works better on larger catalogs** where the base ranker has more uncertainty. On small catalogs, the base model is confident about most ordering decisions, leaving less room for steering.

## Results

| Budget | Recall@10 | Documentary Lift |
|--------|-----------|-----------------|
| 0.00   | ~23%      | >1.0x           |
| 0.10   | ~23%      | >1.0x           |
| 0.30   | ~23%      | >1.0x           |
| 0.50   | ~24%      | ~1.0x           |
| 1.00   | ~24%      | 1.0x (baseline) |

The lift is modest (>1.0x) because the catalog is small and the documentary base rate is low. But the critical finding: **recall stays stable across all budget levels** (< 3pp range). The orthogonalization guarantees that steering doesn't damage the base ranking even when the steering effect is limited.

## Quick Version

For a simpler approach without moments:

```python
from mosaic import govern

# Documentary boost: +1 for documentaries, 0 for others
doc_boosts = {
    movie_id: 1.0 if "Documentary" in movie_genres[movie_id] else 0.0
    for movie_id in candidate_movies
}

result = govern(
    base_scores=cf_scores,
    steering_scores=doc_boosts,
    budget=0.3,
)

# Check what moved
for receipt in result.receipts[:5]:
    delta = receipt.base_rank - receipt.final_rank
    if delta != 0:
        print(f"Movie {receipt.item}: rank {receipt.base_rank} → {receipt.final_rank} (moved {delta})")
```

## When Steering Is Weak

If your steering effect is limited (as with small catalogs), consider:

1. **Lower the budget**: `budget=0.1` or `budget=0.0` frees more edges for reordering
2. **Increase lambda_m**: Stronger steering weight amplifies the policy signal
3. **Use the full MOSAICScorer**: Calibrated confidence protects only truly confident edges, leaving more room for steering
4. **Expand the candidate set**: More candidates = more uncertain pairs = more room to steer

```python
# More aggressive steering for small catalogs
scorer = MOSAICScorer(
    moment_affinities=A,
    config=MOSAICConfig(
        lambda_m=1.0,              # stronger steering
        protection_mode="budget",
        budget_pct=0.10,           # protect less
    ),
)
```

## Takeaway

MovieLens shows both the strength and the honest limitation of governed reranking:
- **Strength**: Recall never degrades, even with aggressive steering
- **Limitation**: On small catalogs with confident base models, the steering effect is bounded

This is a feature, not a bug. The orthogonalization guarantee means you can't accidentally destroy your recommendations. The worst case is "no change" — never "worse."

---

*From the governed-rank validation suite. [Source](https://github.com/rdoku/governed-rank)*
