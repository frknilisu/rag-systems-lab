# CLAUDE.md — RAG-Lab

Guidance for Claude Code working in this repository. Read this fully before writing code. Keep it updated as the project evolves.

---

## 1. What this project is (dual mission)

**RAG-Lab** has two equally important purposes. Hold both in mind at all times:

1. **A learning + experimentation lab** — implement, test, debug, and compare the major RAG architectures with swappable LLM and Vector DB backends.
2. **A public GitHub teaching repository** — a tutorial series in code. Every implementation ships with documentation good enough that a stranger can clone the repo, read along, and *learn RAG from zero* — with enjoyment, not boredom.

> **Prime directive for this repo:** someone who has never built RAG should be able to start at the README, follow the learning path, and come out understanding all 10 architectures — because each one was *taught*, not just committed. If a doc is boring, dry, or assumes too much, it has failed even if the code is perfect.

This means **documentation and code are co-equal deliverables**. A pull request that adds an architecture but not its teaching material is incomplete (see §10 and §13). Code is *how it works*; the docs are *why it exists, what problem it solves, and how to think about it*.

**Audience to write for:** a competent Python developer who is new to RAG. Knows Python; does *not* know embeddings, rerankers, RRF, HyDE, etc. Define jargon the first time. Build intuition before formalism.

### Architectures in scope (10) — also the tutorial chapter order

| # | Name | Module | Family |
|---|------|--------|--------|
| 1 | Standard RAG | `standard` | Core |
| 2 | Hybrid RAG | `hybrid` | Core |
| 3 | Recursive / Multi-Step RAG | `recursive` | Core |
| 4 | Self-RAG | `self_rag` | Core |
| 5 | Graph-Augmented RAG | `graph` | Knowledge-structured |
| 6 | Knowledge-Enhanced RAG | `knowledge_enhanced` | Knowledge-structured |
| 7 | HyDE | `hyde` | Query/retrieval optimization |
| 8 | Query-Transforming RAG | `query_transform` | Query/retrieval optimization |
| 9 | Memory-Augmented RAG | `memory_rag` | Memory & temporal |
| 10 | Streaming / Real-Time RAG | `streaming` | Memory & temporal |

The numbering is also the **recommended reading order**: each chapter builds on the previous, and each one ends by exposing the limitation that motivates the next. Treat the architectures as a curriculum, not a grab-bag.

The same architectures will **later be re-implemented as n8n workflows** (separately, by the user). Every architecture must be designed as an explicit **graph of named, discrete steps over a shared state object** so it maps cleanly onto n8n nodes — see §9.

---

## 2. Core design principles

1. **Teach, don't just ship.** Every architecture answers, in its docs: what problem motivates it, the one-sentence idea, how it works step by step, when to use it, when not to, and how it compares to what came before. Implementation without explanation is half the job.
2. **Common contract.** Every architecture implements one interface (`RAGPipeline`: `index()` + `query()`), returns one result type (`RAGResult`), and is interchangeable behind config. This is what makes them comparable *and* what makes the tutorial coherent.
3. **Provider-agnostic.** LLM, embeddings, vector store, reranker, and graph store are behind interfaces and chosen via config. Never import a provider SDK inside an architecture.
4. **Steps over state.** Each pipeline is a sequence (or branching/looping graph) of named steps that read/write a shared `RAGState`. Steps are small, testable, traceable — and become both teaching units and n8n nodes.
5. **Config-driven, not hardcoded.** Models, top-k, chunk sizes, prompts, thresholds — all live in config.
6. **Observable by default.** Every step records a trace (inputs, outputs, scores, tokens, latency). The trace is also a *teaching tool*: docs show real traces so readers see what actually happened.
7. **Build the primitives ourselves.** Prefer thin, transparent implementations over heavy frameworks (no LangChain/LlamaIndex in the core). Readable code is teachable code, and it ports to n8n. Heavy libs appear only as optional, isolated adapters.
8. **Reproducible & runnable.** Every architecture has a `examples/` script a reader can run and see output. Seeded where possible; config snapshots saved with every eval run.
9. **Incremental + tested.** Land Standard RAG fully (code + tests + tutorial + example) before breadth. Every step and architecture gets tests.

---

## 3. Tech stack

- **Python** ≥ 3.11
- **Env / deps:** `uv` (lockfile committed)
- **Config:** `pydantic` + `pydantic-settings` + YAML (layered)
- **LLM + embeddings:** `litellm` behind our own `LLMProvider` / `EmbeddingProvider` interface
- **Vector stores:** Chroma (default, local), Qdrant (hybrid + prod-like), FAISS (in-memory) — behind a `VectorStore` interface
- **Sparse/keyword:** `rank-bm25` for the educational path; Qdrant/OpenSearch documented for prod
- **Reranking (optional):** cross-encoder via `sentence-transformers`, or Cohere rerank via config
- **Graph:** `networkx` (default) + optional `neo4j` adapter
- **CLI:** `typer`
- **API:** `fastapi` + `uvicorn` (also the n8n HTTP integration surface)
- **Tracing/logging:** `structlog`; optional Langfuse/Phoenix adapter
- **Eval:** our own metrics + optional `ragas`
- **Docs site:** `mkdocs` + `mkdocs-material` + `mkdocs-mermaid2-plugin` — the tutorial is published as a browsable site, not just loose markdown
- **Quality:** `ruff`, `mypy`, `pytest` + `pytest-cov`
- **Local infra:** `docker-compose.yml` for Qdrant, Neo4j, Ollama

Keep the core lean; optional/heavy deps go behind extras (`rag-lab[qdrant]`, `rag-lab[graph]`, `rag-lab[eval]`, `rag-lab[docs]`).

---

## 4. Folder structure

```
rag-lab/
├── CLAUDE.md
├── README.md                 # the front door: hook + learning path + quickstart (see §10)
├── CONTRIBUTING.md           # how to add an architecture (incl. the docs bar)
├── LICENSE                   # MIT (permissive — it's a learning resource)
├── CODE_OF_CONDUCT.md
├── pyproject.toml            # uv-managed, optional extras per backend
├── uv.lock
├── .env.example
├── .gitignore
├── Makefile
├── mkdocs.yml                # docs-site nav = the curriculum
├── docker-compose.yml        # qdrant, neo4j, ollama
├── config/
│   ├── default.yaml
│   ├── llm/ · vectordb/      # one profile file each
│   └── experiments/
├── src/rag_lab/
│   ├── config/               # pydantic schema + layered loader
│   ├── core/
│   │   ├── interfaces.py      # Protocols (LLM, Embedding, VectorStore, Retriever, Reranker, GraphStore, MemoryStore)
│   │   ├── types.py           # Document, Chunk, Query, RetrievalResult, RAGState, RAGResult, StepTrace
│   │   ├── registry.py        # name -> impl + factories
│   │   ├── pipeline.py        # Step protocol + Pipeline runner + tracing
│   │   └── errors.py
│   ├── providers/            # llm/ embeddings/ vectorstores/ rerankers/ graph/
│   ├── ingestion/            # loaders.py, chunkers.py, indexer.py
│   ├── retrieval/            # dense.py, sparse_bm25.py, hybrid.py, rerank.py, fusion.py
│   ├── generation/           # generator.py + prompts/ (versioned template files)
│   ├── memory/               # base.py, stores.py
│   ├── architectures/
│   │   ├── base.py            # RAGPipeline ABC
│   │   └── <key>/             # pipeline.py, steps.py, config.py, README.md  (one per architecture)
│   ├── evaluation/           # datasets, retrieval_metrics, generation_metrics, runner, compare
│   ├── observability/        # tracing.py, logging.py
│   ├── api/                  # app.py, routes.py
│   └── cli/                  # main.py
├── examples/                 # ONE runnable, heavily-commented script per architecture
│   ├── 01_standard_rag.py     # `python examples/01_standard_rag.py` -> prints answer + trace
│   ├── 02_hybrid_rag.py
│   └── ...
├── notebooks/                # optional annotated notebooks for interactive learners
├── tests/                    # unit/ integration/ fixtures/
├── data/                     # corpora/ (sample docs) · eval/ (gold Q/A)
├── docs/                     # the published tutorial (mkdocs)
│   ├── index.md               # landing page of the site
│   ├── learning-path.md       # the curriculum / suggested order
│   ├── concepts/              # prerequisite primers (embeddings, chunking, vector search, RRF, LLM-as-judge…)
│   ├── architectures/         # ONE teaching chapter per architecture (the heart of the tutorial)
│   ├── n8n/                   # node-by-node workflow spec per architecture
│   └── assets/                # diagrams, images
└── scripts/
```

Two doc surfaces per architecture, different jobs:
- `src/rag_lab/architectures/<key>/README.md` — short, for someone reading the *code*. Points to the chapter.
- `docs/architectures/<key>.md` — the **full teaching chapter**, the thing a learner reads. This is the one that must be engaging.

---

## 5. Key abstractions (the contract)

Define these early in `core/` and treat them as stable.

- **Interfaces (`core/interfaces.py`, `typing.Protocol`):** `LLMProvider`, `EmbeddingProvider`, `VectorStore`, `Reranker`, `GraphStore`, `MemoryStore`.
- **Types (`core/types.py`, pydantic):** `Document` → `Chunk`; `RetrievalResult`; `RAGState` (the shared, **JSON-serializable** object passed between steps — also the n8n item payload); `RAGResult` (`answer`, `contexts`, `trace`, `metrics`, `config_snapshot`); `StepTrace`.
- **Pipeline runner (`core/pipeline.py`):** `Step` = named callable `(RAGState, deps) -> RAGState`; `Pipeline` times each step, appends a `StepTrace`, supports explicit conditionals/loops.
- **Registry + factories (`core/registry.py`):** `@register_llm`, `@register_vectorstore`, `@register_architecture`; `build_*` reads config and injects dependencies (so architectures are unit-testable with fakes).
- **Architecture base (`architectures/base.py`):**

```python
class RAGPipeline(ABC):
    name: str
    def index(self, documents: list[Document]) -> IndexStats: ...
    def query(self, question: str, *, session_id: str | None = None) -> RAGResult: ...
```

If you change any contract, update this file and every implementer in the same PR.

---

## 6. Configuration system

Layered & validated. Precedence (low→high): `config/default.yaml` → profile files → experiment file → `RAGLAB_*` env vars → CLI flags. Secrets come only from env/`.env`. The resolved config is snapshotted into every `RAGResult` and eval run. Switching any backend is a **config change only**. Each architecture may add a `config.py` pydantic sub-model nested under the global config.

---

## 7. Evaluation & debugging

- **Retrieval:** recall@k, precision@k, MRR, nDCG vs gold judgments in `data/eval/`.
- **Generation:** faithfulness/groundedness, answer relevance, context precision (LLM-as-judge; optional `ragas`). Track hallucination rate for Self-RAG comparisons.
- **Operational:** end-to-end + per-step latency, tokens, estimated cost.
- **Runner/compare:** run an experiment across architectures on the same dataset; produce a side-by-side table that **feeds the docs** (the comparison chapter uses real numbers, not hand-waving).
- **Debugging:** read `RAGResult.trace` first. Add trace entries liberally — they're cheap, and they double as teaching artifacts.

Always evaluate a new architecture against at least Standard RAG as the baseline.

---

## 8. How to add a new architecture (use BOTH skills)

1. **Scaffold** with the `rag-architecture-scaffold` skill — generates the module, a runnable `examples/<nn>_<key>.py`, a test stub, the README, and the teaching-chapter + n8n-spec skeletons. Never hand-roll the layout.
2. **Write the teaching chapter design first** (`docs/architectures/<key>.md`), using the `rag-tutorial-doc` skill for structure, tone, and the anti-boredom bar. Designing the explanation forces you to think in clean steps before coding.
3. **Implement `steps.py`** reusing `ingestion/` `retrieval/` `generation/` building blocks. No provider SDK imports, no inline prompts (use `generation/prompts/`), no magic numbers (use `config.py`).
4. **Compose in `pipeline.py`** — explicit, readable control flow.
5. **Register** it and add its config defaults.
6. **Make the example runnable** and confirm it prints a sensible answer + trace.
7. **Finish the teaching chapter** with a real trace, a diagram, "when to use / not", pitfalls, and "what's new vs the previous chapter".
8. **Fill `docs/n8n/<key>.md`** — one node per step.
9. **Test**, then **eval** vs the baseline; fold numbers into the comparison chapter.

Reuse over duplication: shared steps (e.g. query rewrite used by HyDE and Query-Transform) get factored into a shared module.

---

## 9. n8n portability (design constraint)

- **Steps = nodes.** Single-purpose, clearly named.
- **State = item JSON.** `RAGState` stays JSON-serializable; provider calls happen inside steps, not via objects smuggled through state.
- **Document the mapping.** Every architecture gets `docs/n8n/<key>.md`: ordered nodes, node types, inputs/outputs, JSON shape between nodes.
- **HTTP bridge.** FastAPI exposes each architecture (and ideally each step) so an n8n workflow can replicate natively *or* call our service. Document both.
- **Prompts external.** Live in `generation/prompts/` so the exact text pastes into n8n LLM nodes.

Authoring an architecture implicitly authors its n8n blueprint; the `docs/n8n/<key>.md` is a required deliverable.

---

## 10. Documentation & teaching standard (first-class deliverable)

This repo *is* a tutorial. Use the `rag-tutorial-doc` skill whenever writing or editing a teaching chapter, concept primer, or the README — it holds the full house style. The essentials:

**README.md (the front door).** Must, in order: hook the reader with the problem RAG solves; one-paragraph "what you'll learn"; a visual map of the 10 architectures as a learning path; a 60-second quickstart that produces visible output; how to read the repo (code ↔ docs ↔ examples); link to the docs site. No corporate filler. A newcomer decides in 30 seconds whether to stay — earn it.

**Every architecture chapter (`docs/architectures/<key>.md`)** follows the same arc so the series feels coherent:
1. **The problem** — a concrete scenario the *previous* architecture handles badly. Start from pain, not definition.
2. **The one-sentence idea** — the whole trick in plain language before any detail.
3. **How it works** — step by step, each step tied to the code, with a **mermaid diagram** of the step graph.
4. **Watch it work** — the runnable example's real output, including a trimmed trace so the reader sees retrieval and reasoning actually happen.
5. **Watch it strain** — the limitation that motivates the next chapter. This is the connective tissue of the curriculum.
6. **When to use / when not to** — honest trade-offs (latency, cost, complexity, accuracy).
7. **Pitfalls** — the mistakes people actually make.
8. **Try it / break it** — 2–3 small exercises that invite tinkering.
9. **Recap + further reading** — original papers, good blog posts.

**The anti-boredom bar (hard rules):**
- Lead with a problem or example, never a definition dump.
- Define every term the first time it appears; assume no prior RAG knowledge.
- Short paragraphs. Break walls of text. Every nontrivial concept earns a diagram, an example, or an analogy.
- Show real output and real traces — concrete beats abstract.
- Conversational second person ("you'll notice…"), light personality welcome; never stiff or corporate.
- No undefined jargon, no restating the obvious, no "as we all know".
- Each chapter explicitly links backward ("remember the problem from Hybrid RAG?") and forward ("which breaks down when… → next chapter").

**Concept primers (`docs/concepts/`)** cover prerequisites once (embeddings, chunking, vector search, BM25, RRF, reranking, LLM-as-judge) so chapters can link instead of re-explaining.

A chapter that merely restates the code in prose has failed. Teach the *mental model*.

---

## 11. Coding conventions

- Type hints everywhere; `mypy` should pass. Public functions get docstrings stating intent + the n8n node they map to (if applicable).
- Code in `examples/` and `architectures/` is **read by learners** — comment the *why*, name things for clarity over cleverness, keep functions short. This code is exhibit material.
- No provider SDK imports outside `providers/`; no prompt literals outside `generation/prompts/`; no magic numbers inside architectures.
- Typed errors from `core/errors.py`; never swallow exceptions in a step — record in trace and re-raise/degrade explicitly.
- `structlog` with `step`, `architecture`, `session_id` bound.
- Modules small (~300 lines → split). Tests use deterministic fakes + a tiny fixture corpus; no real APIs in CI.

---

## 12. Commands (Makefile targets)

```
make setup        # uv sync + hooks
make infra-up     # docker compose up qdrant/neo4j/ollama
make lint         # ruff
make typecheck    # mypy
make test         # pytest (no network)
make example A=standard         # run examples/01_standard_rag.py
make index / query / eval / compare    # CLI workflows
make docs-serve   # mkdocs serve  -> live-preview the tutorial site
make docs-build   # mkdocs build  -> static site (CI publishes to GitHub Pages)
make serve        # uvicorn rag_lab.api.app:app --reload
```

CLI (Typer): `index`, `query`, `eval`, `compare`, `serve`, `trace`.

---

## 13. Build roadmap (phased — do in order)

- **Phase 0 — Foundation.** Repo scaffold, `pyproject` + extras, config, `core/` (interfaces, types, registry, pipeline runner), logging/tracing, one LiteLLM provider + embedder + Chroma store, Typer skeleton, unit tests. *Exit:* `rag-lab query` runs a trivial pipeline with a trace.
- **Phase 0.5 — Repo-as-tutorial setup.** README front door, LICENSE/CONTRIBUTING/CoC, mkdocs + mkdocs-material + mermaid, `docs/index.md` + `docs/learning-path.md`, and the first **concept primers** (embeddings, chunking, vector search). GitHub Pages publish workflow. Do this early so every later phase plugs its chapter into a living site.
- **Phase 1 — Ingestion & retrieval primitives** (+ primers for BM25, RRF, reranking).
- **Phase 2 — Standard RAG (the template).** Full slice: code + tests + runnable example + the first *complete* teaching chapter + n8n spec + baseline eval. Every later architecture copies this shape.
- **Phase 3 — Hybrid RAG.**
- **Phase 4 — Query optimization:** HyDE, then Query-Transforming.
- **Phase 5 — Recursive / Multi-Step RAG.**
- **Phase 6 — Self-RAG.**
- **Phase 7 — Knowledge-structured:** Graph-Augmented, then Knowledge-Enhanced.
- **Phase 8 — Memory-Augmented RAG.**
- **Phase 9 — Streaming / Real-Time RAG.**
- **Phase 10 — Compare & publish.** Comparison chapter with real eval numbers, finalize all n8n specs, polish the README and learning path, ship the docs site.

Land each phase complete before the next. "Complete" = code **and** tests **and** runnable example **and** an engaging teaching chapter **and** the n8n spec.

---

## 14. Guardrails for Claude Code

- Confirm the contract (§5) before implementing; propose contract changes explicitly.
- A phase/PR is **not done** without: code, tests, a runnable `examples/` script, an *engaging* `docs/architectures/<key>.md` chapter (use the `rag-tutorial-doc` skill), and `docs/n8n/<key>.md`.
- Write for a RAG newcomer. If you used a term you didn't define, fix it. If a chapter reads like a spec, rewrite it as a lesson.
- Prefer extending building blocks over duplicating them. Keep the core framework-light.
- Never commit secrets. Update this CLAUDE.md when structure or contracts change.
