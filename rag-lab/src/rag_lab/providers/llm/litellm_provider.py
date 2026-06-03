"""LiteLLM provider — thin wrapper over litellm.completion().

LiteLLM speaks to 100+ models (OpenAI, Anthropic, OpenRouter, Groq, Ollama, …)
through a single interface. We only import it here, never in architecture code.

n8n node mapping: "LLM Call" node (HTTP Request to provider or built-in LLM node)
"""

from __future__ import annotations

import os
from typing import Iterator

import litellm
import structlog

from rag_lab.config.schema import LLMConfig
from rag_lab.core.errors import ProviderError
from rag_lab.core.registry import register_llm

log = structlog.get_logger(__name__)

# Suppress litellm's verbose default logging unless DEBUG is set.
litellm.suppress_debug_info = True


@register_llm("litellm")
class LiteLLMProvider:
    """Wraps litellm.completion() for chat-style completions."""

    def __init__(self, config: LLMConfig) -> None:
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

        self.api_key: str | None = None
        if config.api_key_env:
            self.api_key = os.environ.get(config.api_key_env)

        self.base_url: str | None = None
        if config.base_url_env:
            self.base_url = os.environ.get(config.base_url_env) or config.base_url
        elif config.base_url:
            self.base_url = config.base_url

        log.debug(
            "litellm_provider_ready",
            model=self.model,
            base_url=self.base_url,
        )

    def _call_kwargs(self) -> dict[str, object]:
        """Build extra kwargs for litellm — only include keys that are set."""
        kw: dict[str, object] = {}
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.base_url:
            kw["api_base"] = self.base_url
        return kw

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self._call_kwargs(),
                **kwargs,
            )
            return str(response.choices[0].message.content)
        except Exception as exc:
            raise ProviderError("litellm", str(exc)) from exc

    def stream(self, messages: list[dict[str, str]], **kwargs: object) -> Iterator[str]:
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **self._call_kwargs(),
                **kwargs,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            raise ProviderError("litellm", str(exc)) from exc
