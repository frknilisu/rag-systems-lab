.PHONY: setup infra-up infra-down lint format typecheck test example \
        index query eval compare docs-serve docs-build serve

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	uv sync --all-extras
	@echo "✓ Dependencies installed. Copy .env.example → .env and add your keys."

# ── Infrastructure (Qdrant, Neo4j, Ollama) ────────────────────────────────────
infra-up:
	docker compose up -d

infra-down:
	docker compose down

# ── Quality ───────────────────────────────────────────────────────────────────
lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

typecheck:
	uv run mypy src

test:
	uv run pytest

test-no-cov:
	uv run pytest --no-cov

# ── Run an example (usage: make example A=01_standard_rag) ───────────────────
example:
	uv run python examples/$(A).py

# ── CLI shortcuts ─────────────────────────────────────────────────────────────
index:
	uv run rag-lab index $(ARGS)

query:
	uv run rag-lab query $(ARGS)

eval:
	uv run rag-lab eval $(ARGS)

compare:
	uv run rag-lab compare $(ARGS)

# ── API server ────────────────────────────────────────────────────────────────
serve:
	uv run uvicorn rag_lab.api.app:app --reload

# ── Docs (MkDocs tutorial site) ───────────────────────────────────────────────
docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build
