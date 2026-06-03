try:
    from rag_lab.providers.embeddings.sentence_transformers_provider import (  # noqa: F401
        SentenceTransformersProvider,
    )
except ImportError:
    pass  # sentence-transformers not installed; install rag-lab[embeddings]
