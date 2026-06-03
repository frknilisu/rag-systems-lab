"""RAG-Lab CLI — built with Typer.

Commands
--------
  rag-lab index   Chunk, embed, and index documents into the vector store
  rag-lab query   Retrieve context and generate an answer
  rag-lab providers  List registered providers
  rag-lab config  Show the resolved configuration
"""

from __future__ import annotations

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


# ── index ─────────────────────────────────────────────────────────────────────

@app.command()
def index(
    path: str = typer.Argument(..., help="Path to a file or directory of documents"),
    config_root: Optional[Path] = typer.Option(None, "--config-root", hidden=True),
) -> None:
    """Chunk, embed, and index documents into the vector store.

    Accepts a single .txt/.md file or a directory (recursively finds all
    .txt and .md files). Re-indexing the same file is safe — chunks are
    upserted by ID, so duplicates are overwritten, not doubled.
    """
    cfg = _get_config(config_root)

    from rag_lab.core.registry import build_embedding, build_vectorstore
    from rag_lab.ingestion.chunkers import build_chunker
    from rag_lab.ingestion.indexer import Indexer

    console.print(f"[cyan]Loading providers…[/cyan]")
    try:
        embedder = build_embedding(cfg.embeddings)
        store = build_vectorstore(cfg.vectordb)
    except Exception as exc:
        err_console.print(f"[red]Provider error:[/red] {exc}")
        raise typer.Exit(1) from exc

    chunker = build_chunker(cfg.ingestion)
    indexer = Indexer(
        embedder=embedder,
        store=store,
        chunker=chunker,
        bm25_index_path=cfg.ingestion.bm25_index_path,
    )

    console.print(f"[cyan]Indexing[/cyan] {path} …")
    try:
        stats = indexer.index_file(path)
    except Exception as exc:
        err_console.print(f"[red]Indexing error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(
        f"[green]✓ Indexed[/green] "
        f"{stats.num_documents} doc(s) → "
        f"[bold]{stats.num_chunks}[/bold] chunks "
        f"({stats.duration_ms:.0f} ms)"
    )
    console.print(
        f"  Vector store: [dim]{cfg.vectordb.path}[/dim] "
        f"(total: {store.count()} chunks)\n"
        f"  BM25 index:   [dim]{cfg.ingestion.bm25_index_path}[/dim]"
    )


# ── query ─────────────────────────────────────────────────────────────────────

@app.command()
def query(
    question: str = typer.Argument(..., help="Question to answer"),
    architecture: str = typer.Option("standard", "--arch", "-a", help="Architecture to use"),
    top_k: int = typer.Option(0, "--top-k", "-k", help="Override retrieval top-k (0 = use config)"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
    config_root: Optional[Path] = typer.Option(None, "--config-root", hidden=True),
) -> None:
    """Retrieve relevant context and generate an answer with a pipeline trace.

    Delegates to the registered architecture (default: standard). The full
    step trace — including per-step latency and I/O summary — is printed
    after the answer so you can see exactly what the pipeline did.
    """
    cfg = _get_config(config_root)

    if top_k > 0:
        from rag_lab.config.schema import RetrievalConfig
        cfg = cfg.model_copy(
            update={"retrieval": RetrievalConfig(top_k=top_k, hybrid=cfg.retrieval.hybrid)}
        )

    from rag_lab.core.registry import build_architecture
    from rag_lab.observability.tracing import format_trace

    try:
        pipeline = build_architecture(architecture, config=cfg)
    except Exception as exc:
        err_console.print(f"[red]Architecture error:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        result = pipeline.query(question)
    except Exception as exc:
        err_console.print(f"[red]Pipeline error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if json_output:
        console.print_json(result.model_dump_json())
        return

    console.print(Panel(result.answer, title=f"Answer  [{architecture}]", border_style="green"))
    console.print(format_trace(result))


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
