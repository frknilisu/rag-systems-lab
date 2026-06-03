# Contributing to RAG-Lab

RAG-Lab is a learning resource first. That means **clarity is a feature**, and the bar for documentation is as high as the bar for code. A perfectly correct architecture with a boring or confusing chapter is not done.

## The golden rule

Write for a competent Python developer who is **new to RAG**. Define jargon the first time. Build intuition before mechanics. If a newcomer would get lost or bored, revise.

## Adding a new architecture

The full contract lives in [`CLAUDE.md`](CLAUDE.md) (§5, §8, §10). In short:

1. **Scaffold it** with the `rag-architecture-scaffold` skill — never hand-roll the layout. This generates the module, a runnable example, a test stub, and the teaching-chapter + n8n-spec skeletons.
2. **Implement** `steps.py` and `pipeline.py` using the shared building blocks. No provider SDK imports in an architecture, no inline prompts, no magic numbers.
3. **Make the example runnable** and capture its real answer + trace.
4. **Write the teaching chapter** with the `rag-tutorial-doc` skill. Open with a problem, paste the real trace, and end by motivating the next chapter.
5. **Fill the n8n spec** — one node per step.
6. **Test and eval** against the Standard RAG baseline.

### Definition of done

A change that adds or modifies an architecture must include **all** of:

- [ ] Code (`pipeline.py`, `steps.py`, `config.py`) following the contract
- [ ] A runnable, commented `examples/<NN>_<key>.py`
- [ ] Unit tests (steps with fakes) + one integration test
- [ ] An *engaging* `docs/architectures/<key>.md` chapter
- [ ] The `docs/n8n/<key>.md` spec
- [ ] `make lint && make typecheck && make test` all green

## Docs checklist (read it back as a newcomer)

- [ ] Opens with a concrete problem, not a definition
- [ ] States the one-sentence idea before the mechanics
- [ ] Has a mermaid step diagram whose node names match `steps.py`
- [ ] Shows real example output + a trimmed trace (never fabricated)
- [ ] Links forward to the architecture that fixes its limitation
- [ ] Defines every term on first use; no walls of text

## Style & tooling

- Python ≥ 3.11, `uv` for deps. Run `make setup` once.
- `ruff` for lint/format, `mypy` for types, `pytest` for tests — all enforced in CI.
- Keep the core framework-light and readable; this code is exhibit material.

Thanks for helping make RAG easier to learn.
