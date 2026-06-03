"""RAG answer generation — prompt building and LLM call helpers.

Prompts are loaded from generation/prompts/ as plain text files.
This keeps prompt text out of Python code and makes it copy-pasteable
into n8n LLM nodes without editing source.

n8n node: "Build Prompt" (build_rag_messages)
"""

from __future__ import annotations

from pathlib import Path

from rag_lab.core.types import RetrievalResult

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_PROMPT_CACHE: dict[str, str] = {}


def load_prompt(name: str) -> str:
    """Load a prompt template from generation/prompts/<name>. Cached after first read."""
    if name not in _PROMPT_CACHE:
        _PROMPT_CACHE[name] = (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()
    return _PROMPT_CACHE[name]


def build_rag_messages(
    question: str,
    contexts: list[RetrievalResult],
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Assemble the LLM message list from retrieved contexts and a question.

    Produces the standard [system, user] message pair used by every RAG
    architecture. Custom system prompts (e.g. persona, domain constraints)
    can be injected via ``system_prompt``; the default is loaded from
    ``generation/prompts/rag_system.txt``.

    n8n node: "Build Prompt"
    """
    if system_prompt is None:
        system_prompt = load_prompt("rag_system.txt")

    if contexts:
        ctx_parts = [
            f"[Source {i + 1}]\n{r.chunk.text}"
            for i, r in enumerate(contexts)
        ]
        ctx_text = "\n\n---\n\n".join(ctx_parts)
    else:
        ctx_text = "(no context retrieved — answer from knowledge if possible)"

    user_content = f"Context:\n{ctx_text}\n\nQuestion: {question}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
