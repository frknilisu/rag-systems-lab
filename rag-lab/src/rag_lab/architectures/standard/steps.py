"""Standard RAG pipeline steps.

Two steps, executed in order:

  retrieve  — embed the question, search the vector store, return top-k chunks
  generate  — build a grounded prompt and call the LLM

Each step is a pure function (RAGState, PipelineDeps) → RAGState. They are
intentionally single-purpose so they map directly onto n8n nodes. See
docs/n8n/standard.md for the node-by-node spec.
"""

from __future__ import annotations

import structlog

from rag_lab.core.pipeline import PipelineDeps
from rag_lab.core.types import RAGState
from rag_lab.generation.generator import build_rag_messages
from rag_lab.retrieval.dense import DenseRetriever

logger = structlog.get_logger(__name__)


def retrieve_step(state: RAGState, deps: PipelineDeps) -> RAGState:
    """Embed the question and fetch the most similar chunks from the vector store.

    Reads  : state.question
    Writes : state.retrieval_results

    n8n node: "Dense Retrieve"
    """
    top_k: int = deps.extra.get("top_k", 5)
    retriever = DenseRetriever(embedder=deps.embedder, store=deps.store)
    results = retriever.retrieve(state.question, top_k=top_k)
    state.retrieval_results = results

    state.metadata["_trace_inputs"] = {"question": state.question[:120], "top_k": top_k}
    state.metadata["_trace_outputs"] = {
        "n_retrieved": len(results),
        "top_score": round(results[0].score, 4) if results else None,
        "bottom_score": round(results[-1].score, 4) if results else None,
    }

    logger.debug(
        "retrieved",
        question=state.question[:80],
        n_results=len(results),
        top_score=round(results[0].score, 4) if results else None,
    )
    return state


def generate_step(state: RAGState, deps: PipelineDeps) -> RAGState:
    """Build a grounded prompt from retrieved contexts and call the LLM.

    Reads  : state.question, state.retrieval_results
    Writes : state.answer

    n8n node: "Generate Answer"
    """
    messages = build_rag_messages(state.question, state.retrieval_results)

    state.metadata["_trace_inputs"] = {
        "n_contexts": len(state.retrieval_results),
        "question": state.question[:120],
    }

    answer = deps.llm.complete(messages)
    state.answer = answer

    state.metadata["_trace_outputs"] = {
        "answer_chars": len(answer),
        "answer_preview": answer[:100].replace("\n", " "),
    }

    logger.debug("generated", answer_chars=len(answer))
    return state
