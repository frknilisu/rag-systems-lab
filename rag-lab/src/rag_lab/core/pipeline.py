"""Pipeline runner with per-step tracing.

A Pipeline is a list of (name, callable) pairs. Each callable has the signature:
    fn(state: RAGState, deps: PipelineDeps) -> RAGState

PipelineDeps is a plain dataclass holding provider instances. Steps are
intentionally kept small and single-purpose — they map 1-to-1 onto n8n nodes.

Usage:
    pipeline = Pipeline(
        steps=[("retrieve", retrieve_step), ("generate", generate_step)],
        deps=PipelineDeps(llm=..., embedder=..., store=...),
    )
    state, traces = pipeline.run(RAGState(question="..."))
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from rag_lab.core.types import RAGState, StepTrace


# ── Dependency container ──────────────────────────────────────────────────────

@dataclass
class PipelineDeps:
    """Holds provider instances injected into every step.

    Attributes are typed as Any so steps can be written and tested without
    importing concrete provider classes. The real types are protocol-checked
    at runtime via isinstance() in providers/__init__.py.
    """

    llm: Any = None
    embedder: Any = None
    store: Any = None
    reranker: Any = None
    graph: Any = None
    memory: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


# ── Step decorator ────────────────────────────────────────────────────────────

StepFn = Callable[[RAGState, PipelineDeps], RAGState]


def step(name: str) -> Callable[[StepFn], StepFn]:
    """Decorator that tags a function with a step name (cosmetic, optional)."""
    def decorator(fn: StepFn) -> StepFn:
        fn.__step_name__ = name  # type: ignore[attr-defined]
        return fn
    return decorator


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    """Runs a sequence of named steps over a shared RAGState, recording traces.

    Each step is timed individually. Exceptions bubble up with the step name
    attached so the caller can identify which node failed.
    """

    def __init__(
        self,
        steps: list[tuple[str, StepFn]],
        deps: PipelineDeps,
    ) -> None:
        self.steps = steps
        self.deps = deps

    def run(self, state: RAGState) -> tuple[RAGState, list[StepTrace]]:
        """Execute all steps in order; return final state and trace list."""
        traces: list[StepTrace] = []
        for name, fn in self.steps:
            started_at = datetime.now(timezone.utc)
            t0 = time.perf_counter()
            try:
                state = fn(state, self.deps)
            except Exception as exc:
                raise RuntimeError(f"Step '{name}' failed: {exc}") from exc
            latency_ms = (time.perf_counter() - t0) * 1000
            traces.append(
                StepTrace(
                    step=name,
                    started_at=started_at,
                    ended_at=datetime.now(timezone.utc),
                    latency_ms=latency_ms,
                )
            )
        return state, traces
