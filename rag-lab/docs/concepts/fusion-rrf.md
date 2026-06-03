# Fusion & RRF

You've run two retrievers — dense search and BM25 — and you have two ranked lists of chunks. Now what?

You can't just concatenate them: the same chunk might appear in both lists, you'd have duplicates, and the scores aren't on the same scale. Dense returns cosine similarities in [0, 1]. BM25 returns unbounded floats that could be 0.3 or 47.8 depending on the corpus. Adding them directly would be nonsense.

**Reciprocal Rank Fusion (RRF)** sidesteps all of this. It ignores scores entirely and uses only *rank position*. A chunk at rank 3 in the dense list gets a contribution of `1/(60+3)`, no matter what its cosine similarity was. This makes it completely robust to score-scale mismatches.

## The formula

For each chunk, sum its RRF contribution across every list it appears in:

```
RRF_score(chunk) = Σ   1 / (k + rank_i)
                  lists
```

Where `k = 60` is a smoothing constant from the original paper. The `+1` ensures the top-ranked item doesn't get an infinite score.

A concrete example — three chunks, two lists:

| Chunk | Dense rank | BM25 rank | RRF score |
|-------|-----------|-----------|-----------|
| A | 1 | 3 | 1/61 + 1/63 = 0.0322 |
| B | 2 | 1 | 1/62 + 1/61 = 0.0325 |
| C | 3 | — | 1/63 = 0.0159 |

Chunk B wins — it placed well in *both* lists, which is exactly the right signal. Chunk A was top in one list. Chunk C only appeared in one list and scores much lower.

## Using it in RAG-Lab

```python
from rag_lab.retrieval.fusion import reciprocal_rank_fusion

dense_results  = dense_retriever.retrieve(query, top_k=15)
sparse_results = bm25_retriever.retrieve(query, top_k=15)

fused = reciprocal_rank_fusion(
    [dense_results, sparse_results],
    k=60,
    top_k=5,
)
```

You can pass any number of ranked lists — not just two. If you add a third retriever (say, a title-only search), include it in the list and RRF handles it seamlessly.

## Why k=60?

The constant `k = 60` comes from the original Cormack et al. (2009) paper, which found it worked well across dozens of retrieval benchmarks. Intuitively:

- **k → 0**: Only the top-1 result in each list matters (winner-takes-all)
- **k → ∞**: All ranks contribute equally (no preference for top results)
- **k = 60**: A smooth curve — top results matter more, but positions 5–20 still contribute meaningfully

In practice, you rarely need to tune k. The default is robust.

## Visualising the rank contribution

```
Rank position →  1     5    10    20    50
Contribution →  1/61  1/65  1/70  1/80  1/110
               0.016  0.015  0.014  0.012  0.009
```

You'll notice the curve is quite flat — the difference between rank 1 and rank 20 is only about 30%. This is by design: RRF is conservative. It rewards consistent appearance in multiple lists over a single very-high rank in one list.

## RRF vs score normalisation

An alternative to RRF is normalising all scores to [0, 1] and adding them with weights. This can work, but it's brittle:

- You need to know the score distribution in advance (or estimate it per query)
- Different query lengths produce different BM25 score ranges
- The optimal weights depend on the corpus

RRF needs none of that. It's the reason [Hybrid RAG](../architectures/hybrid.md) uses it as the default fusion strategy.

---

**When it matters most:** any time you combine two or more retrievers with different score scales. RRF is the default fusion method for Hybrid RAG and appears in Self-RAG, Recursive RAG, and anywhere multiple evidence sources need to be merged.

> **Reference:** Cormack, Clarke, Buettcher. "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods." SIGIR 2009.
