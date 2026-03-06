# RAG Safety: Steering Retrieval Toward Grounded Documents

Your RAG pipeline retrieves the top-K documents by embedding similarity. But similarity doesn't mean factual. A document can be semantically close to the query while containing hallucination-prone content, outdated facts, or policy-violating material.

You need to steer retrieval toward grounded, policy-safe documents — without destroying the semantic relevance that makes RAG work.

## The Problem

Standard RAG retrieval optimizes a single objective: `similarity(query_embedding, doc_embedding)`. Adding a second signal (groundedness, factuality, policy compliance) creates the injection problem:

```python
# Naive approach — breaks retrieval quality
final_score = similarity_score + weight * groundedness_score
```

If groundedness correlates with similarity (and it usually does — well-written authoritative documents tend to be both grounded AND semantically rich), the weighted combination creates interference. Documents that are moderately relevant but very grounded get boosted above documents that are highly relevant.

## The Fix

```python
from mosaic import govern

result = govern(
    base_scores=retrieval_scores,       # embedding similarity from vector DB
    steering_scores=groundedness_scores, # factuality / policy compliance signal
    budget=0.5,
)

# Feed reranked documents to the LLM
context_docs = [documents[doc_id] for doc_id in result.ranked_items[:5]]
```

At `budget=0.5`, the top 50% of similarity-based ordering decisions are locked. The LLM still gets the most relevant documents first. Groundedness steering only affects pairs where the vector DB wasn't confident about the ordering.

## Building the Groundedness Signal

The groundedness signal can come from multiple sources:

```python
# Option 1: Classifier-based
# A model that scores documents on factual grounding
groundedness_scores = {
    doc_id: grounding_classifier.predict(documents[doc_id])
    for doc_id in candidate_docs
}

# Option 2: Source authority
# Known-reliable sources get higher scores
groundedness_scores = {
    doc_id: 1.0 if documents[doc_id].source in trusted_sources else 0.0
    for doc_id in candidate_docs
}

# Option 3: Recency + citation density
# Newer documents with more citations are more likely grounded
groundedness_scores = {
    doc_id: recency_score(doc) * citation_density(doc)
    for doc_id, doc in documents.items()
}

# Option 4: Multi-signal composite
# Combine multiple safety signals
groundedness_scores = {
    doc_id: (
        0.4 * factuality_score[doc_id] +
        0.3 * policy_compliance[doc_id] +
        0.3 * source_authority[doc_id]
    )
    for doc_id in candidate_docs
}
```

## Why Budget=0.5 for RAG

RAG retrieval is more sensitive to relevance degradation than recommendation systems. If your top-1 document becomes irrelevant, the LLM's response quality drops sharply. A higher budget (0.5) protects more ordering decisions:

| Budget | Use Case |
|--------|----------|
| 0.3    | General-purpose RAG — some relevance flexibility OK |
| 0.5    | Production RAG — conservative, protect top results |
| 0.7    | High-stakes RAG (legal, medical) — minimal disruption |
| 0.8    | Critical systems — only steer where retrieval is uncertain |

```python
# Conservative RAG safety
result = govern(
    base_scores=retrieval_scores,
    steering_scores=groundedness_scores,
    budget=0.5,   # protect top-half of ordering decisions
)

# Aggressive RAG safety (compliance-critical)
result = govern(
    base_scores=retrieval_scores,
    steering_scores=groundedness_scores,
    budget=0.2,   # allow more reordering for safety
)
```

## The Audit Trail

For RAG, the audit trail is especially valuable. When the LLM generates a response, you can trace which documents influenced it and why they were selected:

```python
for receipt in result.receipts[:5]:
    print(f"Document: {receipt.item}")
    print(f"  Similarity rank: {receipt.base_rank}")
    print(f"  Final rank: {receipt.final_rank}")
    print(f"  Similarity score: {receipt.base_score:.4f}")
    print(f"  Groundedness: {receipt.steering_score:.4f}")
    print(f"  Orthogonalized safety: {receipt.orthogonalized_steering:.4f}")
    print()
```

If a user questions the LLM's output, you can show:
- Which documents were in the context window
- Why each document was ranked where it was
- Whether any safety steering moved it up or down
- That the top similarity-based decisions were protected

## Integration with Vector DBs

```python
# Pinecone example
import pinecone
from mosaic import govern

# Step 1: Standard retrieval
results = index.query(query_embedding, top_k=50)
retrieval_scores = {r.id: r.score for r in results.matches}

# Step 2: Score groundedness
groundedness = score_groundedness(results.matches)

# Step 3: Governed reranking
governed = govern(
    base_scores=retrieval_scores,
    steering_scores=groundedness,
    budget=0.5,
)

# Step 4: Feed top-5 to LLM
context = [fetch_doc(doc_id) for doc_id in governed.ranked_items[:5]]
```

## What This Prevents

Without governed reranking, your RAG pipeline can surface:
- **Hallucination-prone documents**: Semantically similar but factually unreliable
- **Outdated content**: High similarity but stale information
- **Policy violations**: Content that matches the query but violates organizational rules
- **Adversarial injections**: Documents crafted to be similar but containing prompt injection

By steering at the ranking layer — before the LLM sees the documents — you reduce these risks without modifying the retrieval model or the LLM.

## Install

```bash
pip install governed-rank
```

Three lines between your vector DB and your LLM. No model retraining. Full audit trail.

---

*From the governed-rank documentation. [Source](https://github.com/rdoku/governed-rank)*
