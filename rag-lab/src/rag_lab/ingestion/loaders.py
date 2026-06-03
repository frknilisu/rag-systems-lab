"""Document loaders — turn files and strings into Document objects.

Each loader has a single method: load() -> list[Document].
The factory function load_documents() auto-detects source type by path.

n8n node mapping: "Read File" / "HTTP Request" → this produces the Document list
that feeds into the chunker node.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import structlog

from rag_lab.core.errors import RagLabError
from rag_lab.core.types import Document

log = structlog.get_logger(__name__)

# File extensions we handle by default
_TEXT_EXTENSIONS = {".txt", ".text"}
_MARKDOWN_EXTENSIONS = {".md", ".markdown"}
_ALL_EXTENSIONS = _TEXT_EXTENSIONS | _MARKDOWN_EXTENSIONS


def _file_id(path: Path) -> str:
    """Stable document ID derived from the file path."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", path.stem)[:64]


def _content_id(text: str) -> str:
    """ID for inline/string documents: short content hash."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class DocumentLoader(Protocol):
    def load(self) -> list[Document]: ...


# ── Concrete loaders ──────────────────────────────────────────────────────────

class TextFileLoader:
    """Load a single plain-text file as one Document."""

    def __init__(self, path: str | Path, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.encoding = encoding

    def load(self) -> list[Document]:
        if not self.path.exists():
            raise RagLabError(f"File not found: {self.path}")
        text = self.path.read_text(encoding=self.encoding).strip()
        doc = Document(
            id=_file_id(self.path),
            text=text,
            metadata={"source": str(self.path), "type": "text"},
        )
        log.debug("loaded_text_file", path=str(self.path), chars=len(text))
        return [doc]


class MarkdownLoader:
    """Load a Markdown file, optionally stripping front-matter and HTML."""

    def __init__(
        self,
        path: str | Path,
        strip_frontmatter: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        self.path = Path(path)
        self.strip_frontmatter = strip_frontmatter
        self.encoding = encoding

    def load(self) -> list[Document]:
        if not self.path.exists():
            raise RagLabError(f"File not found: {self.path}")
        text = self.path.read_text(encoding=self.encoding)
        if self.strip_frontmatter:
            text = _strip_frontmatter(text)
        text = text.strip()
        doc = Document(
            id=_file_id(self.path),
            text=text,
            metadata={"source": str(self.path), "type": "markdown"},
        )
        log.debug("loaded_markdown", path=str(self.path), chars=len(text))
        return [doc]


class DirectoryLoader:
    """Recursively load all text/markdown files from a directory.

    Args:
        path: Directory to scan.
        glob: Glob pattern relative to path (default: all .txt and .md files).
        extensions: Set of file extensions to include (default: .txt + .md).
    """

    def __init__(
        self,
        path: str | Path,
        glob: str = "**/*",
        extensions: set[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.glob = glob
        self.extensions = extensions or _ALL_EXTENSIONS

    def load(self) -> list[Document]:
        if not self.path.is_dir():
            raise RagLabError(f"Directory not found: {self.path}")
        documents: list[Document] = []
        for file_path in sorted(self.path.glob(self.glob)):
            if file_path.is_file() and file_path.suffix.lower() in self.extensions:
                loader = _pick_loader(file_path)
                documents.extend(loader.load())
        log.info("directory_loaded", path=str(self.path), n_docs=len(documents))
        return documents


class StringLoader:
    """Create a Document directly from a string — useful for tests and notebooks."""

    def __init__(self, text: str, doc_id: str | None = None, metadata: dict | None = None) -> None:
        self.text = text
        self.doc_id = doc_id or _content_id(text)
        self.metadata = metadata or {}

    def load(self) -> list[Document]:
        return [Document(id=self.doc_id, text=self.text, metadata=self.metadata)]


# ── Convenience factory ────────────────────────────────────────────────────────

def load_documents(path: str | Path) -> list[Document]:
    """Auto-detect source type and load documents.

    - File path   → TextFileLoader or MarkdownLoader based on extension
    - Directory   → DirectoryLoader (recurses into all .txt/.md files)
    """
    p = Path(path)
    if p.is_dir():
        return DirectoryLoader(p).load()
    return _pick_loader(p).load()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pick_loader(path: Path) -> DocumentLoader:
    ext = path.suffix.lower()
    if ext in _MARKDOWN_EXTENSIONS:
        return MarkdownLoader(path)
    return TextFileLoader(path)


def _strip_frontmatter(text: str) -> str:
    """Remove YAML front-matter (--- ... ---) from the top of a Markdown file."""
    if not text.startswith("---"):
        return text
    end = text.find("---", 3)
    if end == -1:
        return text
    return text[end + 3:].lstrip("\n")
