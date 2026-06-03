"""Tests for core/pipeline.py — step execution and tracing."""

from __future__ import annotations

import pytest

from rag_lab.core.pipeline import Pipeline, PipelineDeps
from rag_lab.core.types import RAGState


def _make_append_step(word: str):
    def fn(state: RAGState, deps: PipelineDeps) -> RAGState:
        state.answer += word
        return state
    return fn


def test_pipeline_runs_steps_in_order():
    pipeline = Pipeline(
        steps=[
            ("step_a", _make_append_step("hello ")),
            ("step_b", _make_append_step("world")),
        ],
        deps=PipelineDeps(),
    )
    state = RAGState(question="test")
    final_state, traces = pipeline.run(state)

    assert final_state.answer == "hello world"
    assert len(traces) == 2
    assert traces[0].step == "step_a"
    assert traces[1].step == "step_b"


def test_pipeline_records_latency():
    def slow_step(state: RAGState, deps: PipelineDeps) -> RAGState:
        import time
        time.sleep(0.01)
        return state

    pipeline = Pipeline(steps=[("slow", slow_step)], deps=PipelineDeps())
    _, traces = pipeline.run(RAGState(question="x"))
    assert traces[0].latency_ms >= 5.0  # at least 5 ms


def test_pipeline_wraps_step_errors():
    def bad_step(state: RAGState, deps: PipelineDeps) -> RAGState:
        raise ValueError("oops")

    pipeline = Pipeline(steps=[("bad", bad_step)], deps=PipelineDeps())
    with pytest.raises(RuntimeError, match="Step 'bad' failed"):
        pipeline.run(RAGState(question="x"))


def test_pipeline_passes_deps():
    captured: list[object] = []

    def capture_step(state: RAGState, deps: PipelineDeps) -> RAGState:
        captured.append(deps.llm)
        return state

    sentinel = object()
    pipeline = Pipeline(
        steps=[("capture", capture_step)],
        deps=PipelineDeps(llm=sentinel),
    )
    pipeline.run(RAGState(question="x"))
    assert captured[0] is sentinel


def test_empty_pipeline_returns_unchanged_state():
    pipeline = Pipeline(steps=[], deps=PipelineDeps())
    state = RAGState(question="unchanged")
    final, traces = pipeline.run(state)
    assert final.question == "unchanged"
    assert traces == []
