"""Provider package — importing this triggers all @register_* decorators.

Each sub-package's __init__.py tries to import its implementation. Heavy
optional dependencies (litellm, chromadb, sentence-transformers) are guarded
with try/except so the core framework loads even if an extra isn't installed.
"""

from rag_lab.providers import llm, embeddings, vectorstores

__all__ = ["llm", "embeddings", "vectorstores"]
