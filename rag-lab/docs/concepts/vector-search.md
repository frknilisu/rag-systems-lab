# Vector Search

You have a million document chunks, each represented as a 384-dimensional vector. A query arrives. You want the 5 chunks closest to the query vector in that 384-dimensional space.

The naive approach — compute cosine similarity against every single chunk — is called **exact nearest-neighbor search**. For a million vectors it's doable but slow: ~100–500ms at inference time. At a billion vectors it's unusable.

**Vector search** (also called approximate nearest-neighbor, or ANN) trades a tiny amount of accuracy for orders-of-magnitude speed improvements using smart indexing structures. In practice, the "approximation" loses almost nothing you'd notice.

## How HNSW works (the intuition)

The most widely used ANN algorithm is **HNSW** (Hierarchical Navigable Small World graphs). Imagine it like a city transit network:

- At the top level, a sparse "express" network connects landmarks far apart.
- At the bottom level, a dense "local" network connects nearby stops.
- To find the nearest point to your query, you navigate top-down: take express routes to get roughly in the right area, then local routes to zero in on the nearest neighbor.

This greedy graph traversal finds near-optimal neighbors in O(log n) time instead of O(n) — that's the difference between 1ms and 500ms at a million vectors.

You don't need to implement this yourself. Every major vector database handles it under the hood.

## Distance metrics

Vector stores support several distance metrics. The right one depends on how your embedding model was trained:

| Metric | Formula | Use when |
|--------|---------|----------|
| **Cosine similarity** | `dot(a,b) / (‖a‖ · ‖b‖)` | Most embedding models (default) |
| **Dot product** | `dot(a, b)` | Models trained with dot-product loss (e.g., OpenAI) |
| **Euclidean (L2)** | `‖a − b‖` | Rarely used for text embeddings |

In RAG-Lab, Chroma defaults to cosine distance. Using the wrong metric can silently degrade retrieval quality — always match the metric to the model's training objective.

## What a search call looks like

```python
from rag_lab.retrieval.dense import DenseRetriever

retriever = DenseRetriever(embedder=embedder, store=store)
results = retriever.retrieve("What is chunking overlap?", top_k=5)

for r in results:
    print(f"[{r.rank}] score={r.score:.3f}  {r.chunk.text[:80]}…")
```

```
[0] score=0.891  Chunk overlap is the number of characters shared between adjacent chunks...
[1] score=0.847  When you cut text at a boundary, information near the cut is lost...
[2] score=0.801  A typical overlap is 10–15% of the chunk size...
```

The `score` is a cosine similarity in [0, 1]. Higher is more similar.

## The vector store options in RAG-Lab

| Store | Persistence | Notes |
|-------|-------------|-------|
| **Chroma** | Local on-disk | Default; zero setup, great for development |
| **Qdrant** | Docker / cloud | Production-grade; supports hybrid (dense + sparse) natively |
| **FAISS** | In-memory | Ultra-fast; no persistence; good for batch eval |

All three implement the same `VectorStore` protocol. Switching is a config change:

```yaml
# config/default.yaml
vectordb:
  profile: qdrant   # was: chroma
```

---

**When it matters most:** vector search is the performance bottleneck in any RAG system at scale. For development and datasets under ~100k chunks, Chroma's exact search is fast enough. For production, Qdrant with HNSW shines — and it also unlocks native hybrid search, which is the foundation of [Hybrid RAG](../architectures/hybrid.md).
