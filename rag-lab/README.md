<!--
This is the front door. A stranger decides in ~30 seconds whether to stay — earn it.
Order matters (see CLAUDE.md §10 and the rag-tutorial-doc skill). Replace the TODOs with
real content as the project lands. Keep it skimmable; personality welcome; no filler.
-->

# RAG-Lab — learn modern RAG by building all of it

Ask an LLM about *your* documents and it confidently makes things up. Retrieval-Augmented Generation (RAG) fixes that by fetching the right context first — but "RAG" isn't one thing anymore. It's a whole family of designs, each invented to fix where the previous one breaks.

**RAG-Lab teaches that family as a tutorial in code.** Ten architectures, built from scratch in plain Python, each with a runnable example and a chapter that explains *why it exists* — not just how it works. Read it start to finish like a course, or jump to the one you need.

> New to RAG? Start at the [learning path](docs/learning-path.md) and read the chapters in order — each one ends by exposing the problem the next one solves.

## What you'll learn

How retrieval actually works (embeddings, keyword search, hybrid fusion, reranking), and how the major RAG architectures build on each other:

| # | Architecture | The problem it solves | Family |
|---|--------------|-----------------------|--------|
| 1 | Standard RAG | "answer from my docs, not your imagination" | Core |
| 2 | Hybrid RAG | dense search misses exact keywords/IDs | Core |
| 3 | Recursive / Multi-Step | questions that need several hops | Core |
| 4 | Self-RAG | the model retrieves junk and trusts it | Core |
| 5 | Graph-Augmented | reasoning over entities and relationships | Knowledge-structured |
| 6 | Knowledge-Enhanced | enforcing schemas / ontology constraints | Knowledge-structured |
| 7 | HyDE | short, vague queries retrieve poorly | Query optimization |
| 8 | Query-Transforming | one bad query → rewrite, split, route it | Query optimization |
| 9 | Memory-Augmented | remembering across turns and sessions | Memory & temporal |
| 10 | Streaming / Real-Time | data that changes by the minute | Memory & temporal |

Every architecture is swappable across LLM providers (OpenAI, Anthropic, local Ollama, …) and vector DBs (Chroma, Qdrant, FAISS) with a **config change, no code change**. Each is also designed to be rebuilt as an [n8n workflow](docs/n8n/) — the same steps, as nodes.

## 60-second quickstart

```bash
git clone <repo-url> && cd rag-lab
make setup                 # uv sync + dev tools
cp .env.example .env        # add one LLM key (or run fully local via Ollama)
make example A=standard     # index a tiny corpus, ask a question, SEE the answer + trace
```

You'll get a real grounded answer and a trace showing exactly what was retrieved and why — before reading a line of theory. <!-- TODO: paste real sample output here once Phase 2 lands -->

## How to read this repo

Three views of the same idea, meant to be read together:

- **`docs/architectures/`** — the lessons. Start here. <!-- published as a site: `make docs-serve` -->
- **`examples/`** — one runnable, commented script per architecture. Run them; poke them.
- **`src/rag_lab/`** — the implementation. Small, framework-light, readable on purpose.

New concepts (embeddings, BM25, RRF, reranking…) each get a short primer in **`docs/concepts/`** so the chapters stay focused.

## Project status

Built in phases — each one is complete (code + tests + docs) before the next starts.

| Phase | What | Status |
|-------|------|--------|
| 0 | Foundation: config, core types/interfaces, pipeline runner, registry, LiteLLM + SentenceTransformers + Chroma providers, Typer CLI | ✅ Done |
| 0.5 | Repo-as-tutorial setup: README, mkdocs site, concept primers | ✅ Done |
| 1 | Ingestion & retrieval primitives: loaders, chunkers (fixed/recursive), Indexer (embed+Chroma+BM25), DenseRetriever, BM25Retriever, RRF fusion, HybridRetriever, CrossEncoderReranker | ✅ Done |
| 2 | Standard RAG — `StandardRAGPipeline`, generation module, teaching chapter, n8n spec | ✅ Done |
| 3 | Hybrid RAG | 🔜 Next |
| 4–10 | Remaining 8 architectures | ⏳ |

**What works today:**

```bash
# Index a document (real chunking + embedding + Chroma + BM25)
rag-lab index data/corpora/intro_to_rag.txt

# Query via Standard RAG pipeline (default architecture)
RAGLAB_LLM__PROFILE=ollama RAGLAB_LLM__MODEL="ollama/qwen2.5:3b" \
rag-lab query "What is the difference between BM25 and dense retrieval?"

# Or run the annotated example script
RAGLAB_LLM__PROFILE=ollama RAGLAB_LLM__MODEL="ollama/qwen2.5:3b" \
python examples/01_standard_rag.py
```

**What's next (Phase 3):** Hybrid RAG — adds BM25 keyword search alongside dense retrieval and merges the two ranked lists with Reciprocal Rank Fusion.

## Contributing

This is a learning resource — clarity is a feature. See [`CONTRIBUTING.md`](CONTRIBUTING.md); the bar for docs is as high as the bar for code.

## License

MIT — see [`LICENSE`](LICENSE).
