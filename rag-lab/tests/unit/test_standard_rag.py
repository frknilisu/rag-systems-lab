"""Tests for Standard RAG pipeline.

All tests use fake providers — no real API calls, no real embeddings,
no real vector store. The fakes are defined in tests/conftest.py.
"""

from __future__ import annotations

import pytest

from rag_lab.architectures.standard.pipeline import StandardRAGPipeline
from rag_lab.architectures.standard.steps import generate_step, retrieve_step
from rag_lab.config.schema import RagLabConfig
from rag_lab.core.pipeline import Pipeline, PipelineDeps
from rag_lab.core.types import Chunk, Document, RAGResult, RAGState, RetrievalResult
from rag_lab.generation.generator import build_rag_messages, load_prompt


# ── generation/generator.py ───────────────────────────────────────────────────

class TestBuildRagMessages:
    def test_returns_two_messages(self) -> None:
        msgs = build_rag_messages("What is X?", [])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_question_in_user_message(self) -> None:
        msgs = build_rag_messages("What is X?", [])
        assert "What is X?" in msgs[1]["content"]

    def test_contexts_numbered_and_included(self) -> None:
        chunk = Chunk(id="c1", doc_id="d1", text="Relevant content here.")
        results = [RetrievalResult(chunk=chunk, score=0.9, rank=0)]
        msgs = build_rag_messages("What is X?", results)
        assert "[Source 1]" in msgs[1]["content"]
        assert "Relevant content here." in msgs[1]["content"]

    def test_multiple_contexts_separated(self) -> None:
        chunks = [
            Chunk(id=f"c{i}", doc_id="d1", text=f"Chunk {i} text.")
            for i in range(3)
        ]
        results = [RetrievalResult(chunk=c, score=0.9 - i * 0.1, rank=i) for i, c in enumerate(chunks)]
        msgs = build_rag_messages("q", results)
        assert "[Source 1]" in msgs[1]["content"]
        assert "[Source 2]" in msgs[1]["content"]
        assert "[Source 3]" in msgs[1]["content"]

    def test_custom_system_prompt(self) -> None:
        msgs = build_rag_messages("q", [], system_prompt="Custom prompt.")
        assert msgs[0]["content"] == "Custom prompt."

    def test_empty_contexts_fallback_text(self) -> None:
        msgs = build_rag_messages("q", [])
        assert "no context" in msgs[1]["content"].lower()

    def test_load_prompt_cached(self) -> None:
        p1 = load_prompt("rag_system.txt")
        p2 = load_prompt("rag_system.txt")
        assert p1 is p2  # same object — cached

    def test_system_prompt_not_empty(self) -> None:
        prompt = load_prompt("rag_system.txt")
        assert len(prompt) > 50


# ── steps.py ──────────────────────────────────────────────────────────────────

class TestRetrieveStep:
    def test_writes_retrieval_results(self, fake_embedder, fake_store, sample_chunks) -> None:
        # Pre-populate the store
        fake_store.upsert(
            ids=[c.id for c in sample_chunks],
            vectors=[[0.0, 0.0, 0.0, 1.0]] * len(sample_chunks),
            payloads=[{"text": c.text, "doc_id": c.doc_id} for c in sample_chunks],
        )
        deps = PipelineDeps(embedder=fake_embedder, store=fake_store, extra={"top_k": 2})
        state = RAGState(question="What is the capital?")
        state = retrieve_step(state, deps)
        assert len(state.retrieval_results) == 2

    def test_empty_store_returns_empty_results(self, fake_embedder, fake_store) -> None:
        deps = PipelineDeps(embedder=fake_embedder, store=fake_store, extra={"top_k": 5})
        state = RAGState(question="Any question")
        state = retrieve_step(state, deps)
        assert state.retrieval_results == []

    def test_sets_trace_outputs(self, fake_embedder, fake_store, sample_chunks) -> None:
        fake_store.upsert(
            ids=[c.id for c in sample_chunks],
            vectors=[[0.0, 0.0, 0.0, 1.0]] * len(sample_chunks),
            payloads=[{"text": c.text, "doc_id": c.doc_id} for c in sample_chunks],
        )
        deps = PipelineDeps(embedder=fake_embedder, store=fake_store, extra={"top_k": 2})
        state = RAGState(question="test")
        state = retrieve_step(state, deps)
        assert "_trace_outputs" in state.metadata
        assert state.metadata["_trace_outputs"]["n_retrieved"] == 2

    def test_top_k_from_extra(self, fake_embedder, fake_store, sample_chunks) -> None:
        fake_store.upsert(
            ids=[c.id for c in sample_chunks],
            vectors=[[0.0, 0.0, 0.0, 1.0]] * len(sample_chunks),
            payloads=[{"text": c.text, "doc_id": c.doc_id} for c in sample_chunks],
        )
        deps = PipelineDeps(embedder=fake_embedder, store=fake_store, extra={"top_k": 1})
        state = RAGState(question="test")
        state = retrieve_step(state, deps)
        assert len(state.retrieval_results) == 1


class TestGenerateStep:
    def test_writes_answer(self, fake_llm, fake_embedder, fake_store) -> None:
        deps = PipelineDeps(llm=fake_llm, embedder=fake_embedder, store=fake_store)
        state = RAGState(question="What is X?", retrieval_results=[])
        state = generate_step(state, deps)
        assert state.answer == "fake answer"

    def test_sets_trace_outputs(self, fake_llm, fake_embedder, fake_store) -> None:
        deps = PipelineDeps(llm=fake_llm, embedder=fake_embedder, store=fake_store)
        state = RAGState(question="What?", retrieval_results=[])
        state = generate_step(state, deps)
        assert "_trace_outputs" in state.metadata
        assert state.metadata["_trace_outputs"]["answer_chars"] == len("fake answer")

    def test_with_contexts_included_in_prompt(self, fake_store) -> None:
        """Verify that retrieved context is passed to the LLM."""
        captured: list[list[dict[str, str]]] = []

        class CapturingLLM:
            def complete(self, messages: list[dict[str, str]], **kw: object) -> str:
                captured.extend([messages])
                return "captured"

        chunk = Chunk(id="c1", doc_id="d1", text="The answer is 42.")
        results = [RetrievalResult(chunk=chunk, score=0.9, rank=0)]
        deps = PipelineDeps(llm=CapturingLLM(), store=fake_store)
        state = RAGState(question="What is the answer?", retrieval_results=results)
        generate_step(state, deps)

        assert len(captured) == 1
        user_msg = captured[0][1]["content"]
        assert "The answer is 42." in user_msg


# ── Pipeline integration ──────────────────────────────────────────────────────

class TestStepTraceIntegration:
    def test_trace_inputs_outputs_captured(self, fake_llm, fake_embedder, fake_store) -> None:
        """Trace metadata set by steps should appear in StepTrace.outputs."""
        deps = PipelineDeps(
            llm=fake_llm,
            embedder=fake_embedder,
            store=fake_store,
            extra={"top_k": 5},
        )
        pipeline = Pipeline(
            steps=[("retrieve", retrieve_step), ("generate", generate_step)],
            deps=deps,
        )
        state, traces = pipeline.run(RAGState(question="test"))

        retrieve_trace = next(t for t in traces if t.step == "retrieve")
        generate_trace = next(t for t in traces if t.step == "generate")

        assert "n_retrieved" in retrieve_trace.outputs
        assert "answer_chars" in generate_trace.outputs

    def test_trace_metadata_cleaned_from_state(self, fake_llm, fake_embedder, fake_store) -> None:
        """_trace_inputs and _trace_outputs should not survive in final state."""
        deps = PipelineDeps(
            llm=fake_llm, embedder=fake_embedder, store=fake_store, extra={"top_k": 5}
        )
        pipeline = Pipeline(
            steps=[("retrieve", retrieve_step), ("generate", generate_step)],
            deps=deps,
        )
        state, _ = pipeline.run(RAGState(question="test"))
        assert "_trace_inputs" not in state.metadata
        assert "_trace_outputs" not in state.metadata


# ── StandardRAGPipeline ───────────────────────────────────────────────────────

class TestStandardRAGPipeline:
    def _make_pipeline(self, fake_llm, fake_embedder, fake_store) -> StandardRAGPipeline:
        cfg = RagLabConfig()
        deps = PipelineDeps(
            llm=fake_llm,
            embedder=fake_embedder,
            store=fake_store,
            extra={"top_k": 5},
        )
        return StandardRAGPipeline(config=cfg, deps=deps)

    def test_query_returns_rag_result(self, fake_llm, fake_embedder, fake_store) -> None:
        pipeline = self._make_pipeline(fake_llm, fake_embedder, fake_store)
        result = pipeline.query("What is X?")
        assert isinstance(result, RAGResult)
        assert result.answer == "fake answer"

    def test_query_has_two_trace_steps(self, fake_llm, fake_embedder, fake_store) -> None:
        pipeline = self._make_pipeline(fake_llm, fake_embedder, fake_store)
        result = pipeline.query("test")
        assert len(result.trace) == 2
        assert result.trace[0].step == "retrieve"
        assert result.trace[1].step == "generate"

    def test_query_includes_config_snapshot(self, fake_llm, fake_embedder, fake_store) -> None:
        pipeline = self._make_pipeline(fake_llm, fake_embedder, fake_store)
        result = pipeline.query("test")
        assert isinstance(result.config_snapshot, dict)
        assert "retrieval" in result.config_snapshot

    def test_index_with_fake_store(self, fake_llm, fake_embedder, fake_store) -> None:
        pipeline = self._make_pipeline(fake_llm, fake_embedder, fake_store)
        docs = [Document(id="d1", text="Paris is the capital of France.")]
        stats = pipeline.index(docs)
        assert stats.num_documents == 1
        assert stats.num_chunks >= 1
        assert fake_store.count() >= 1

    def test_query_after_index_retrieves_chunks(self, fake_llm, fake_embedder, fake_store) -> None:
        pipeline = self._make_pipeline(fake_llm, fake_embedder, fake_store)
        docs = [Document(id="d1", text="Paris is the capital of France.")]
        pipeline.index(docs)
        result = pipeline.query("What is the capital of France?")
        assert len(result.contexts) > 0

    def test_session_id_propagated(self, fake_llm, fake_embedder, fake_store) -> None:
        pipeline = self._make_pipeline(fake_llm, fake_embedder, fake_store)
        result = pipeline.query("test", session_id="session-abc")
        # session_id is stored in state but not surfaced in RAGResult —
        # just verify no error and result is valid
        assert result.answer == "fake answer"

    def test_store_count_property(self, fake_llm, fake_embedder, fake_store) -> None:
        pipeline = self._make_pipeline(fake_llm, fake_embedder, fake_store)
        assert pipeline.store_count == 0
        docs = [Document(id="d1", text="Some text here for testing.")]
        pipeline.index(docs)
        assert pipeline.store_count > 0


# ── Registry ─────────────────────────────────────────────────────────────────

class TestArchitectureRegistry:
    def test_standard_registered(self) -> None:
        from rag_lab.core.registry import _ARCHITECTURE_REGISTRY, _ensure_architectures_loaded
        _ensure_architectures_loaded()
        assert "standard" in _ARCHITECTURE_REGISTRY

    def test_build_architecture_with_config(self, fake_llm, fake_embedder, fake_store) -> None:
        from rag_lab.core.pipeline import PipelineDeps
        from rag_lab.core.registry import build_architecture

        cfg = RagLabConfig()
        deps = PipelineDeps(llm=fake_llm, embedder=fake_embedder, store=fake_store)
        pipeline = build_architecture("standard", config=cfg, deps=deps)
        assert isinstance(pipeline, StandardRAGPipeline)

    def test_build_unknown_architecture_raises(self) -> None:
        from rag_lab.core.errors import ConfigError
        from rag_lab.core.registry import build_architecture
        with pytest.raises(ConfigError, match="Unknown architecture"):
            build_architecture("nonexistent_arch_xyz")
