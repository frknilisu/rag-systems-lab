try:
    from rag_lab.providers.llm.litellm_provider import LiteLLMProvider  # noqa: F401
except ImportError:
    pass  # litellm not installed; install rag-lab[llm]
