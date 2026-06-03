"""Text chunkers — split Documents into Chunks ready for embedding.

Two implementations ship with Phase 1:

  FixedSizeChunker  — splits every `chunk_size` characters with `chunk_overlap`
                       overlap. Predictable, fast, ignores sentence/word boundaries.

  RecursiveChunker  — tries paragraph → line → sentence → word boundaries
                       in order, merging splits back into chunks up to `chunk_size`.
                       Respects natural text structure; almost always the right choice.

Use build_chunker(config) to get the right one from config.

n8n node mapping: "Split Text" Function node between "Read File" and "Embeddings".
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog

from rag_lab.config.schema import IngestionConfig
from rag_lab.core.types import Chunk, Document

log = structlog.get_logger(__name__)


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


# ── Fixed-size chunker ────────────────────────────────────────────────────────

class FixedSizeChunker:
    """Split text into fixed-width windows with overlap.

    Simple and predictable — good baseline for benchmarking.
    Downside: cuts across word and sentence boundaries.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        for i, start in enumerate(range(0, max(len(text), 1), step)):
            window = text[start : start + self.chunk_size]
            if window.strip():
                chunks.append(
                    Chunk(
                        id=f"{document.id}__chunk_{i:04d}",
                        doc_id=document.id,
                        text=window,
                        metadata={**document.metadata, "chunk_index": i},
                    )
                )
        log.debug("fixed_chunked", doc_id=document.id, n_chunks=len(chunks))
        return chunks


# ── Recursive chunker ─────────────────────────────────────────────────────────

class RecursiveChunker:
    """Split text recursively on natural boundaries, then merge back to chunk_size.

    Algorithm:
      1. Try separators in order (paragraph → line → sentence → word).
      2. For any segment still larger than chunk_size, recurse with the next
         separator in the list.
      3. Merge small segments greedily into chunks up to chunk_size.
      4. When starting a new chunk, prepend the last `chunk_overlap` characters
         of the previous chunk so context is not lost at boundaries.

    This is the default chunker and works well for most prose.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    # ── Public ────────────────────────────────────────────────────────────────

    def chunk(self, document: Document) -> list[Chunk]:
        raw_splits = self._split(document.text, self.separators)
        texts = self._merge_with_overlap(raw_splits)
        chunks = [
            Chunk(
                id=f"{document.id}__chunk_{i:04d}",
                doc_id=document.id,
                text=t,
                metadata={**document.metadata, "chunk_index": i},
            )
            for i, t in enumerate(texts)
        ]
        log.debug("recursive_chunked", doc_id=document.id, n_chunks=len(chunks))
        return chunks

    # ── Internal ──────────────────────────────────────────────────────────────

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text on the first useful separator."""
        if not text.strip():
            return []

        # If text already fits, no further splitting needed.
        if len(text) <= self.chunk_size:
            return [text]

        # No more separators: fall back to hard character split.
        if not separators:
            step = self.chunk_size - self.chunk_overlap
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), step)
                if text[i : i + self.chunk_size].strip()
            ]

        sep, remaining_seps = separators[0], separators[1:]
        parts = text.split(sep) if sep else list(text)

        result: list[str] = []
        for part in parts:
            if not part.strip():
                continue
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._split(part, remaining_seps))
        return result

    def _merge_with_overlap(self, splits: list[str]) -> list[str]:
        """Greedily merge splits into chunks ≤ chunk_size, with overlap at joins."""
        if not splits:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split)
            separator = " " if current else ""

            if current_len + len(separator) + split_len > self.chunk_size and current:
                chunks.append(" ".join(current))
                # Seed the next chunk with trailing splits that fit in the overlap window.
                overlap: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) + 1 > self.chunk_overlap:
                        break
                    overlap.insert(0, s)
                    overlap_len += len(s) + 1
                current = overlap
                current_len = overlap_len

            current.append(split)
            current_len += len(separator) + split_len

        if current:
            chunks.append(" ".join(current))

        return chunks


# ── Factory ───────────────────────────────────────────────────────────────────

def build_chunker(config: IngestionConfig) -> Chunker:
    """Return the chunker named in config.chunker."""
    name = config.chunker.lower()
    if name in ("fixed", "fixed_size"):
        return FixedSizeChunker(config.chunk_size, config.chunk_overlap)
    if name in ("recursive",):
        return RecursiveChunker(config.chunk_size, config.chunk_overlap)
    raise ValueError(
        f"Unknown chunker '{config.chunker}'. Available: fixed, recursive"
    )
