"""RAG-Lab CLI — built with Typer.

Commands
--------
  rag-lab query   Ask a question (Phase 0: trivial pipeline demo)
  rag-lab index   Index documents into the vector store
  rag-lab providers  List registered providers
  rag-lab config  Show the resolved configuration

Phase 0 ships the skeleton; full retrieval-augmented commands arrive in Phase 2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="rag-lab",
    help="RAG-Lab — learn and run RAG architectures.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


def _get_config(config_root: Optional[Path] = None):  # type: ignore[return]
    from rag_lab.config import load_config
    from rag_lab.observability import configure_logging

    cfg = load_config(config_root=config_root)
    configure_logging(cfg.log_level)
    return cfg


# ── query ─────────────────────────────────────────────────────────────────────

@app.command()
def query(
    question: str = typer.Argument(..., help="Question to answer"),
    architecture: str = typer.Option("standard", "--arch", "-a", help="Architecture name"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
    config_root: Optional[Path] = typer.Option(None, "--config-root", hidden=True),
) -> None:
    """Ask a question and get an answer with a pipeline trace.

    Phase 0: runs a trivial two-step pipeline (retrieve=no-op, generate=LLM call)
    to demonstrate the framework. Phase 2 wires in real retrieval.
    """
    cfg = _get_config(config_root)

    from rag_lab.core.pipeline import Pipeline, PipelineDeps
    from rag_lab.core.registry import build_llm
    from rag_lab.core.types import RAGResult, RAGState
    from rag_lab.observability.tracing import format_trace

    llm = build_llm(cfg.llm)
    deps = PipelineDeps(llm=llm)

    def _retrieve_step(state: RAGState, d: PipelineDeps) -> RAGState:
        # Phase 0: no-op retrieval — returns empty context list.
        # Replaced in Phase 2 when Standard RAG is implemented.
        state.retrieval_results = []
        return state

    def _generate_step(state: RAGState, d: PipelineDeps) -> RAGState:
        ctx_text = (
            "\n\n".join(r.chunk.text for r in state.retrieval_results)
            or "(no retrieved context — direct LLM answer)"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer the question using the provided context. "
                    "If no context is provided, answer from your own knowledge."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{ctx_text}\n\nQuestion: {state.question}",
            },
        ]
        state.answer = d.llm.complete(messages)
        return state

    pipeline = Pipeline(
        steps=[("retrieve", _retrieve_step), ("generate", _generate_step)],
        deps=deps,
    )

    state = RAGState(question=question)
    try:
        state, traces = pipeline.run(state)
    except Exception as exc:
        err_console.print(f"[red]Pipeline error:[/red] {exc}")
        raise typer.Exit(1) from exc

    result = RAGResult(
        answer=state.answer,
        contexts=state.retrieval_results,
        trace=traces,
        config_snapshot=cfg.model_dump(),
    )

    if json_output:
        console.print_json(result.model_dump_json())
        return

    console.print(Panel(result.answer, title="Answer", border_style="green"))
    console.print(format_trace(result))


# ── index ─────────────────────────────────────────────────────────────────────

@app.command()
def index(
    path: str = typer.Argument(..., help="Path to a file or directory of documents"),
    architecture: str = typer.Option("standard", "--arch", "-a"),
    config_root: Optional[Path] = typer.Option(None, "--config-root", hidden=True),
) -> None:
    """Index documents into the vector store.

    Phase 0: prints a placeholder. Full indexing arrives in Phase 1
    (ingestion primitives) and Phase 2 (Standard RAG pipeline).
    """
    _get_config(config_root)
    console.print(
        f"[yellow]index[/yellow] command is a Phase 1 feature. "
        f"Path '{path}', arch '{architecture}' noted."
    )


# ── providers ─────────────────────────────────────────────────────────────────

@app.command()
def providers() -> None:
    """List all registered providers."""
    from rag_lab.core.registry import list_providers

    data = list_providers()
    table = Table(title="Registered providers", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Name(s)", style="white")
    for category, names in data.items():
        table.add_row(category, ", ".join(names) if names else "—")
    console.print(table)


# ── config ────────────────────────────────────────────────────────────────────

@app.command(name="config")
def show_config(
    config_root: Optional[Path] = typer.Option(None, "--config-root", hidden=True),
) -> None:
    """Print the resolved configuration as JSON."""
    cfg = _get_config(config_root)
    console.print_json(cfg.model_dump_json(indent=2))


# ── eval / compare (stubs) ────────────────────────────────────────────────────

@app.command()
def eval(
    dataset: str = typer.Argument(..., help="Eval dataset name or path"),
    architecture: str = typer.Option("standard", "--arch", "-a"),
) -> None:
    """Evaluate an architecture on a dataset. (Phase 7 feature)"""
    console.print("[yellow]eval[/yellow] command arrives in Phase 7.")


@app.command()
def compare(
    architectures_arg: str = typer.Argument(..., help="Comma-separated architecture names"),
    dataset: str = typer.Option("default", "--dataset", "-d"),
) -> None:
    """Compare multiple architectures side-by-side. (Phase 10 feature)"""
    console.print("[yellow]compare[/yellow] command arrives in Phase 10.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
