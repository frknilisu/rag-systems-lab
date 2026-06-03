# Reranking

Your dense retriever runs in milliseconds. It embeds the query, scans the vector index, and returns the top 5 chunks. Fast — but there's a hidden trade-off buried in how it works.

When you embed a query, you produce one vector. When you embed a chunk, you produce another. You compare them with a dot product. The model encoded each one *separately* — the query never "sees" the chunk during encoding. That's called a **bi-encoder**, and the separation is what makes it fast.

The problem is that a lot of relevance signal only appears when you read query and chunk *together*. Consider the query *"How does attention work in transformers?"* and a chunk that says *"The attention mechanism, first described by Vaswani et al., uses query, key, and value matrices…"* A bi-encoder might miss that the word "attention" in the chunk is about neural networks, not human focus.

**Reranking** runs a second, more accurate model *after* retrieval to fix the order.

## The cross-encoder

A **cross-encoder** takes the query and a document chunk concatenated together as a single input, runs a full transformer forward pass over both, and outputs a relevance score. Attention flows freely between query tokens and chunk tokens — the model can notice that "attention" in the chunk refers to the same concept as in the query.

```
Input:  [CLS] How does attention work in transformers? [SEP] The attention mechanism... [SEP]
Output: 0.94  ← relevance score
```

This is much more accurate. It's also much slower — you run one forward pass *per candidate chunk*, so you can't use it for a million chunks. The standard pattern is a two-stage pipeline:

```
Stage 1: Dense retriever → top 50 candidates   (fast, uses bi-encoder)
Stage 2: Cross-encoder  → rerank to top 5      (accurate, expensive)
```

You send only 5 chunks to the LLM, but you chose them from a much better-ranked pool.

## Using it in RAG-Lab

```python
from rag_lab.retrieval.dense import DenseRetriever
from rag_lab.retrieval.rerank import CrossEncoderReranker

retriever = DenseRetriever(embedder, store)
reranker  = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")

candidates = retriever.retrieve(query, top_k=20)   # broad first-pass
final      = reranker.rerank(query, candidates, top_k=5)  # accurate second-pass
```

The `ms-marco-MiniLM-L-6-v2` model (~85 MB) is trained specifically to judge search relevance. It's fast for a cross-encoder and good enough for most RAG use cases.

## The latency trade-off

Reranking 20 candidates with a CPU cross-encoder takes roughly 200–500ms depending on chunk length. On a GPU it drops to 10–50ms. Whether that's acceptable depends on your use case:

| Use case | Reranking verdict |
|----------|------------------|
| Interactive chat (user waits) | Marginal if CPU; fine on GPU |
| Batch document processing | Always worth it |
| High-stakes (medical, legal) | Almost always required |
| Simple FAQ bot on small corpus | Probably overkill |

## Bi-encoder vs cross-encoder: the summary

| | Bi-encoder | Cross-encoder |
|--|-----------|--------------|
| **Encodes** | Query and doc separately | Query + doc together |
| **Speed** | Fast (pre-computed embeddings) | Slow (one pass per pair) |
| **Accuracy** | Good | Excellent |
| **Use for** | First-pass retrieval | Reranking top candidates |

A common mistake is trying to use a cross-encoder for all-pairs retrieval — it'll work, but at 1000× the cost. Use bi-encoders to get a shortlist, then cross-encoders to polish it.

---

**When it matters most:** reranking gives the biggest gains when your initial retrieval is broad (top-20 or more) and accuracy matters more than latency. It's the easiest single upgrade to a Standard RAG system — often boosting NDCG@5 by 5–15 percentage points with no other changes.
