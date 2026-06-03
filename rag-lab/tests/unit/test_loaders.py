"""Tests for ingestion/loaders.py."""

from __future__ import annotations

import pytest

from rag_lab.ingestion.loaders import (
    DirectoryLoader,
    MarkdownLoader,
    StringLoader,
    TextFileLoader,
    load_documents,
)
from rag_lab.core.types import Document


def test_string_loader_returns_document():
    loader = StringLoader("Hello world", doc_id="test")
    docs = loader.load()
    assert len(docs) == 1
    assert docs[0].id == "test"
    assert docs[0].text == "Hello world"


def test_string_loader_auto_id():
    loader = StringLoader("some text")
    docs = loader.load()
    assert docs[0].id  # non-empty


def test_string_loader_metadata():
    loader = StringLoader("text", metadata={"source": "unit-test"})
    assert loader.load()[0].metadata["source"] == "unit-test"


def test_text_file_loader(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Line one.\nLine two.")
    docs = TextFileLoader(f).load()
    assert len(docs) == 1
    assert "Line one" in docs[0].text
    assert docs[0].metadata["type"] == "text"


def test_markdown_loader_strips_frontmatter(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("---\ntitle: Test\n---\n\n# Heading\n\nBody text.")
    docs = MarkdownLoader(f).load()
    assert "---" not in docs[0].text
    assert "Body text" in docs[0].text


def test_markdown_loader_no_frontmatter(tmp_path):
    f = tmp_path / "plain.md"
    f.write_text("# Just a heading\n\nSome content.")
    docs = MarkdownLoader(f).load()
    assert "Just a heading" in docs[0].text


def test_text_file_loader_missing_file():
    from rag_lab.core.errors import RagLabError
    with pytest.raises(RagLabError, match="not found"):
        TextFileLoader("/nonexistent/path/file.txt").load()


def test_directory_loader(tmp_path):
    (tmp_path / "a.txt").write_text("Document A")
    (tmp_path / "b.md").write_text("Document B")
    (tmp_path / "skip.py").write_text("not a doc")
    docs = DirectoryLoader(tmp_path).load()
    assert len(docs) == 2
    texts = {d.text for d in docs}
    assert "Document A" in texts
    assert "Document B" in texts


def test_directory_loader_missing():
    from rag_lab.core.errors import RagLabError
    with pytest.raises(RagLabError, match="not found"):
        DirectoryLoader("/nonexistent/dir").load()


def test_load_documents_file(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Sample content.")
    docs = load_documents(f)
    assert len(docs) == 1


def test_load_documents_directory(tmp_path):
    (tmp_path / "one.txt").write_text("One")
    (tmp_path / "two.txt").write_text("Two")
    docs = load_documents(tmp_path)
    assert len(docs) == 2
