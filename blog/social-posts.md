# Social Posts — governed-rank Launch

Ready to copy-paste. Adjust links once the GitHub repo and blog post are live.

---

## Hacker News

**Title:** governed-rank: Steer any ranked list toward policy objectives without breaking accuracy

**URL:** https://github.com/PatternaAi/governed-rank

**Comment:**

Hi HN — I built this after watching three teams at different companies hit the same problem: you add a safety/fairness/policy signal to your ranker, and accuracy drops. The naive approach (weighted combination) fails because the signals correlate, creating interference.

governed-rank solves this with orthogonalization + constrained isotonic projection. The math guarantees Pareto optimality — you literally cannot get more policy effect without giving up more accuracy.

Three lines to use:

    from mosaic import govern
    result = govern(base_scores, steering_scores, budget=0.3)

Works for content moderation, fairness, fraud, RAG safety — anything with a ranked list and a policy objective. Validated on 17 datasets across 6 domains. 122 tests, 99 KB wheel, Apache 2.0.

Happy to answer questions about the math or implementation.

---

## Twitter/X

**Thread (3 tweets):**

**Tweet 1:**
You add a safety score to your ranker. Accuracy drops 5%. Your PM asks why.

This is the injection problem — and it happens in every domain.

We built governed-rank to fix it. Three lines of Python. Pareto-optimal reranking. Full audit trail.

github.com/PatternaAi/governed-rank

**Tweet 2:**
How it works:
1. Orthogonalize your policy signal against base scores (remove interference)
2. Lock confident ordering decisions (budget parameter)
3. Isotonic projection maximizes policy effect under constraints

Result: you can't get more policy effect without giving up more accuracy.

**Tweet 3:**
Validated on 17 datasets:
- COMPAS fairness: 0.963 adverse impact ratio, 98.65% quality retained
- Recommendations: 0.890 stability @ 0.344 exposure
- Healthcare (MIMIC-IV): 71.6% HIGH tier
- Cookieless: 4.65x CVR lift

pip install governed-rank | Apache 2.0

---

## Reddit (r/MachineLearning)

**Title:** [P] governed-rank — Pareto-optimal reranking for policy objectives (content safety, fairness, RAG, fraud)

**Body:**

**Problem:** You have a ranked list and a policy objective (reduce toxicity, increase fairness, promote grounded docs). Adding the policy score to your base scores breaks accuracy because the signals correlate.

**Solution:** `governed-rank` orthogonalizes the steering signal, protects confident ordering decisions, and finds the Pareto-optimal reranking via constrained isotonic projection.

```python
from mosaic import govern
result = govern(base_scores, steering_scores, budget=0.3)
```

**Key results (17 datasets, 6 domains):**
- COMPAS fairness: adverse_impact_ratio=0.963, quality_retained=98.65%
- Recommendations: 0.890 rank stability @ 0.344 policy exposure
- MIMIC-IV healthcare: 71.6% HIGH tier
- Cookieless targeting: 4.65x CVR lift

**What it's not:** This isn't a model or a framework. It's a single function that wraps around any existing ranker. 99 KB wheel, no heavy dependencies.

Apache 2.0: https://github.com/PatternaAi/governed-rank

Paper with proofs forthcoming — happy to discuss the math.

---

## LinkedIn

**Post:**

Releasing governed-rank today -- an open-source Python library for governed reranking.

The problem it solves: every team that adds a policy signal (safety, fairness, compliance) to a ranked system sees accuracy drop. The signals interfere because they correlate with the base scores.

governed-rank fixes this with three steps:
1. Orthogonalize the policy signal against base scores
2. Lock the most confident ordering decisions
3. Constrained isotonic projection finds the optimal reranking

The result is provably Pareto-optimal. You cannot get more policy effect without giving up accuracy. One parameter (budget) controls the tradeoff, and it's interpretable: budget=0.3 means "protect your 30% most confident ordering decisions."

Validated across 17 datasets and 6 domains — recommendations, fairness (COMPAS), healthcare (MIMIC-IV), content moderation, fraud, and cookieless targeting.

Three lines to use:

from mosaic import govern
result = govern(base_scores, steering_scores, budget=0.3)

99 KB. Apache 2.0. pip install governed-rank.

If your team is wrestling with the accuracy-vs-policy tradeoff in ranked systems, I'd love to hear how it works for you.

GitHub: https://github.com/PatternaAi/governed-rank

#MachineLearning #AI #OpenSource #Fairness #ContentSafety #Ranking
