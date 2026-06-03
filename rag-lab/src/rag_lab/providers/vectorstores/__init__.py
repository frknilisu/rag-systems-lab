try:
    from rag_lab.providers.vectorstores.chroma_store import ChromaVectorStore  # noqa: F401
except ImportError:
    pass  # chromadb not installed; install rag-lab[chroma]
