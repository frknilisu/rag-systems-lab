"""Trace rendering helpers.

Used by the CLI and examples to pretty-print a RAGResult trace so readers
can see what happened inside the pipeline at each step.
"""

from __future__ import annotations

from rag_lab.core.types import RAGResult, StepTrace


def format_trace(result: RAGResult) -> str:
    """Return a human-readable multi-line trace summary."""
    lines: list[str] = ["", "── Pipeline trace ────────────────────────────────────"]
    total = sum(t.latency_ms for t in result.trace)
    for t in result.trace:
        lines.append(f"  {t.step:<25} {t.latency_ms:>8.1f} ms")
    lines.append(f"  {'TOTAL':<25} {total:>8.1f} ms")
    lines.append("─────────────────────────────────────────────────────")
    if result.contexts:
        lines.append(f"\n── Top {len(result.contexts)} context(s) ──────────────────────────")
        for r in result.contexts:
            snippet = r.chunk.text[:120].replace("\n", " ")
            lines.append(f"  [{r.rank}] score={r.score:.3f}  {snippet}…")
    return "\n".join(lines)


def step_trace_dict(trace: StepTrace) -> dict[str, object]:
    """Compact dict representation suitable for JSON output."""
    return {
        "step": trace.step,
        "latency_ms": round(trace.latency_ms, 2),
        "started_at": trace.started_at.isoformat(),
        "ended_at": trace.ended_at.isoformat(),
    }
