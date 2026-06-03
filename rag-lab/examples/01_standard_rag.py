#!/usr/bin/env python
"""Standard RAG — the foundational "retrieve then answer" architecture.

What this script teaches
------------------------
- How the index() → query() loop works end-to-end
- What a pipeline trace looks like and what each step reports
- Where retrieval quality matters most (and where it doesn't yet)

Run it
------
With Ollama (fully local, no API key needed):

    ollama pull qwen2.5:3b                    # or llama3.2, mistral, etc.
    RAGLAB_LLM__PROFILE=ollama \\
    RAGLAB_LLM__MODEL="ollama/qwen2.5:3b" \\
    python examples/01_standard_rag.py

With an OpenRouter key (free tier available):

    RAGLAB_LLM__PROFILE=openrouter_free \\
    OPENROUTER_API_KEY=sk-or-... \\
    python examples/01_standard_rag.py

What you'll see
---------------
1. Indexing stats: N documents, M chunks, time taken
2. The grounded answer to the demo question
3. A pipeline trace: per-step latency + what each step retrieved/generated
4. The top retrieved chunks with similarity scores
"""

from __future__ import annotations

import os
from pathlib import Path

# ── 1. Load config ────────────────────────────────────────────────────────────
# Config is loaded from config/default.yaml, then profile YAMLs, then env vars.
# The LLM profile controls which model is used — see config/llm/ for options.

from rag_lab.config import load_config
from rag_lab.observability import configure_logging

cfg = load_config()
configure_logging("WARNING")  # suppress debug noise; use INFO to see structlog events

print("\n── RAG-Lab · Standard RAG Example ──────────────────────────────")
print(f"   LLM   : {cfg.llm.model or '(from profile)'}")
print(f"   Embed : {cfg.embeddings.model}")
print(f"   Store : {cfg.vectordb.provider} @ {cfg.vectordb.path}")
print()

# ── 2. Build the pipeline ─────────────────────────────────────────────────────
# StandardRAGPipeline wraps the two-step loop (retrieve → generate) and owns
# all provider instances. Everything is config-driven — no magic numbers here.

from rag_lab.architectures.standard import StandardRAGPipeline

pipeline = StandardRAGPipeline(config=cfg)

# ── 3. Index the sample corpus (skip if already done) ─────────────────────────
# The corpus is ~1 800 words about RAG — big enough to demonstrate chunking and
# retrieval, small enough to index in a few seconds without a GPU.

CORPUS = Path(__file__).parent.parent / "data" / "corpora" / "intro_to_rag.txt"

if pipeline.store_count == 0:
    print(f"Indexing corpus: {CORPUS.name}")
    stats = pipeline.index_file(str(CORPUS))
    print(
        f"  ✓ {stats.num_documents} doc(s) → {stats.num_chunks} chunks "
        f"({stats.duration_ms:.0f} ms)\n"
    )
else:
    print(f"Corpus already indexed ({pipeline.store_count} chunks in store). Skipping.\n")

# ── 4. Ask a question ─────────────────────────────────────────────────────────
# Change this to anything about RAG — the pipeline will retrieve the relevant
# chunks and ground the LLM's answer in them.

QUESTION = "What is the difference between BM25 and dense retrieval?"

print(f"Question: {QUESTION}\n")
print("Retrieving and generating… (this may take a moment with a local model)\n")

result = pipeline.query(QUESTION)

# ── 5. Print the answer ───────────────────────────────────────────────────────

print("─" * 60)
print("ANSWER")
print("─" * 60)
print(result.answer)
print()

# ── 6. Print the trace ────────────────────────────────────────────────────────
# The trace shows exactly what happened inside the pipeline: how long each
# step took and what it produced. This is the *teaching artifact* — when you
# read the docs, real traces like this one back up every claim about latency
# and what each step does.

print("─" * 60)
print("PIPELINE TRACE")
print("─" * 60)
total_ms = sum(t.latency_ms for t in result.trace)
for t in result.trace:
    print(f"  {t.step:<20} {t.latency_ms:>8.1f} ms")
    for k, v in (t.outputs or {}).items():
        print(f"    {'':20} {k}: {v}")
print(f"  {'TOTAL':<20} {total_ms:>8.1f} ms")
print()

# ── 7. Print the top retrieved contexts ──────────────────────────────────────
# Seeing the retrieved chunks explains *why* the answer looks like it does.
# High-scoring chunks → confident, specific answers. Low or off-topic scores
# → vague or evasive answers. This is the first lever to pull when quality
# is poor: fix retrieval before tuning the prompt.

if result.contexts:
    print("─" * 60)
    print(f"TOP {len(result.contexts)} RETRIEVED CHUNKS")
    print("─" * 60)
    for r in result.contexts:
        snippet = r.chunk.text[:200].replace("\n", " ")
        print(f"  [{r.rank}] score={r.score:.4f}  {snippet}…")
    print()

# ── What to try next ─────────────────────────────────────────────────────────
print("─" * 60)
print("WHAT TO TRY NEXT")
print("─" * 60)
print("  1. Change QUESTION above to something that isn't in the corpus.")
print("     → The scores will drop and the answer will hedge or be wrong.")
print("     → This is Standard RAG's honest failure mode.")
print()
print("  2. Run with --top-k 1 (via config) and watch answer quality drop")
print("     as you give the LLM fewer chunks to work with.")
print()
print("  3. Chapter 2 (Hybrid RAG) adds keyword search so that exact terms")
print("     like 'BM25' or 'HNSW' stop slipping through dense-only retrieval.")
print("─" * 60)
print()
