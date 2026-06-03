# Standard RAG

The baseline: embed the question → dense search → grounded LLM generation.

## Code layout

| File | What it does |
|------|-------------|
| `pipeline.py` | `StandardRAGPipeline` — implements `RAGPipeline.index()` and `.query()` |
| `steps.py` | `retrieve_step`, `generate_step` — the two pipeline steps / n8n nodes |

## Quick start

```bash
# Index a corpus
rag-lab index data/corpora/intro_to_rag.txt

# Query with Standard RAG (default)
rag-lab query "What is retrieval-augmented generation?"

# Or run the annotated example
python examples/01_standard_rag.py
```

## Reading order

Start with the teaching chapter — it explains *why* this architecture exists and
walks through each step before touching the code:

→ **[docs/architectures/standard.md](../../../../docs/architectures/standard.md)**

The n8n blueprint (how to rebuild this as a no-code workflow):

→ **[docs/n8n/standard.md](../../../../docs/n8n/standard.md)**

## Evaluation baseline

Standard RAG is architecture #1 and the **required baseline** for every
evaluation run. When you add a new architecture, compare it against this one
using `make compare ARGS="standard,<new_arch>"`.
