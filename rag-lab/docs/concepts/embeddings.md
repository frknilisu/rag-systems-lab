# Embeddings

Imagine you have 10,000 documents about cooking. A user asks: *"How do I make pasta without gluten?"* None of your documents use those exact words together. One of them says *"wheat-free noodle preparation"* and another says *"celiac-safe pasta dishes"*. A keyword search returns nothing — but both documents are highly relevant.

Embeddings solve this. They convert text into a list of numbers — a *vector* — that captures **meaning**, not just characters. Two pieces of text with similar meaning end up with vectors that are numerically close, even if they share no words.

## The intuition: a map of meaning

Think of each embedding as coordinates in a high-dimensional space. You can't visualise 384 dimensions, but the geometry is real: "king" and "queen" are close together. "Paris" and "France" are close. "Python" (the language) is close to "Java" and far from "snake".

The embedding model learns these positions during training on billions of text samples. By the time it's done, nearby coordinates mean nearby meaning — which is exactly what you need for search.

## What an embedding looks like

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

vec = model.encode("How do I make pasta without gluten?")
print(vec.shape)   # (384,)
print(vec[:5])     # [ 0.032 -0.118  0.041  0.205 -0.073 ]
```

Those 384 numbers are the embedding. Each number is a coordinate in a 384-dimensional space. By itself, a single embedding is meaningless — its power comes from *comparing it to others*.

## Measuring similarity: cosine distance

The standard way to compare two embedding vectors is **cosine similarity** — the cosine of the angle between them. It's 1.0 for identical vectors, 0.0 for unrelated text, and −1.0 for opposites.

```python
from numpy import dot
from numpy.linalg import norm

def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))

v1 = model.encode("wheat-free noodle preparation")
v2 = model.encode("How do I make pasta without gluten?")
print(cosine(v1, v2))   # ~0.78 — pretty similar!
```

In RAG, you embed the user's query and every document chunk, then find the chunks whose vectors are closest to the query vector. That's semantic search.

## Bi-encoder vs cross-encoder

The embedding model you use for retrieval is called a **bi-encoder**: it encodes query and document *separately*, then compares the resulting vectors. This is fast — you pre-compute document embeddings once and store them; at query time you only embed the query.

The trade-off is precision: encoding separately loses some context. A **cross-encoder** reads query and document *together*, giving much better relevance scores but at much higher cost. That's what [reranking](reranking.md) uses.

## Choosing a model

| Model | Size | Dimensions | Notes |
|-------|------|-----------|-------|
| `all-MiniLM-L6-v2` | 90 MB | 384 | Default; great speed/quality balance |
| `BAAI/bge-small-en-v1.5` | 130 MB | 384 | Stronger retrieval performance |
| `all-mpnet-base-v2` | 420 MB | 768 | Higher quality, slower |

In RAG-Lab, the embedding model is set in `config/embeddings/` and never imported inside architecture code. Swap models with a config change.

---

**When it matters most:** every RAG architecture depends on embeddings for the retrieval step. The quality of your embeddings is usually the biggest lever on retrieval recall — much more than tuning top-k or prompt wording.
