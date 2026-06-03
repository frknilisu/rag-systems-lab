# Chunking

Here's the problem: your embedding model can only read about 512 tokens at a time (roughly 400 words). Your documents are often thousands of words long. And even if the model *could* read the whole thing, you wouldn't want it to — a single embedding for a 10-page PDF would average over every topic in the document, making it vague and hard to match precisely.

The solution is **chunking**: split each document into smaller pieces before embedding. Each chunk gets its own vector. When a query arrives, you find the *specific* chunk that answers it, not just the document it came from.

## Size matters — in both directions

If your chunks are **too large**, the embedding captures too many ideas at once. A 2,000-character chunk about "machine learning in healthcare" contains sub-topics about diagnosis, drug discovery, patient monitoring, and billing. A query about drug discovery will retrieve it, but the LLM gets a lot of noise alongside the relevant part.

If your chunks are **too small**, each chunk may lack the context to be useful on its own. A single sentence like *"The threshold is 0.6."* has no meaning without the surrounding paragraph.

The sweet spot for most prose is 300–600 characters (roughly 50–100 words) with some overlap between adjacent chunks.

## Overlap: stitching chunks back together

When you cut text at a boundary, information near the cut is lost. **Chunk overlap** solves this by repeating the last N characters of one chunk at the start of the next. A typical overlap is 10–15% of the chunk size.

```
Chunk 1: "...the enzyme binds to the receptor and triggers a cascade..."
                                              ↑ overlap zone ↑
Chunk 2: "...triggers a cascade that releases calcium ions into the cell..."
```

Both chunks now contain the phrase "triggers a cascade", so a query about calcium release will still find Chunk 1 — even though the relevant detail is in Chunk 2.

## Two chunking strategies

**Fixed-size chunking** cuts at exactly `chunk_size` characters, stepping forward by `chunk_size - chunk_overlap` each time. Fast, predictable, but it slices through sentence and paragraph boundaries:

```python
from rag_lab.ingestion.chunkers import FixedSizeChunker
chunker = FixedSizeChunker(chunk_size=400, chunk_overlap=50)
chunks = chunker.chunk(document)
```

**Recursive chunking** tries natural text boundaries in order: paragraph breaks (`\n\n`) first, then line breaks (`\n`), then sentence endings (`. `), then word boundaries. It only falls back to a harder split when a segment is still larger than `chunk_size`. This preserves logical structure:

```python
from rag_lab.ingestion.chunkers import RecursiveChunker
chunker = RecursiveChunker(chunk_size=400, chunk_overlap=50)
chunks = chunker.chunk(document)
```

Recursive chunking is the default in RAG-Lab and works well for almost all prose. Use fixed-size when you need perfectly predictable chunk lengths (e.g., for token-budget reasons).

## What a chunk looks like

Every chunk is a `Chunk` object with a stable ID, a reference back to its source document, the text, and any inherited metadata:

```python
Chunk(
    id="intro_to_rag__chunk_0003",
    doc_id="intro_to_rag",
    text="BM25 (Best Match 25) is a classical sparse retrieval algorithm...",
    metadata={"source": "data/corpora/intro_to_rag.txt", "chunk_index": 3},
)
```

The ID format `{doc_id}__chunk_{index:04d}` is stable across re-indexing of the same document, so upserts safely overwrite rather than duplicate.

---

**When it matters most:** chunking decisions have an outsized effect on retrieval precision. A well-tuned chunk size can improve recall@5 by 15–20% compared to naive fixed-size splitting — and no amount of prompt engineering fixes a context window full of the wrong chunks.
