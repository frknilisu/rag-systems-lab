"""Architectures package — one sub-package per RAG architecture.

Importing this package triggers all @register_architecture() decorators.
Add each new architecture here as it's implemented (mirrors providers/__init__.py).
"""

# Each import below executes the module's top-level @register_architecture decorator.
from rag_lab.architectures import standard  # noqa: F401

__all__ = ["standard"]
