"""Tests for config/loader.py — layered YAML + env merging."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_lab.config.loader import load_config
from rag_lab.config.schema import RagLabConfig

# Use the real config root so we test against actual profile files.
_REAL_CONFIG_ROOT = Path(__file__).parents[2] / "config"


def test_load_config_returns_raglab_config():
    cfg = load_config(config_root=_REAL_CONFIG_ROOT)
    assert isinstance(cfg, RagLabConfig)


def test_default_llm_profile_is_openrouter_free():
    cfg = load_config(config_root=_REAL_CONFIG_ROOT)
    assert cfg.llm.profile == "openrouter_free"


def test_llm_profile_merges_model():
    cfg = load_config(config_root=_REAL_CONFIG_ROOT)
    # openrouter_free.yaml sets provider=litellm and a model
    assert cfg.llm.provider == "litellm"
    assert "llama" in cfg.llm.model.lower() or cfg.llm.model != ""


def test_embeddings_profile_merges_dimension():
    cfg = load_config(config_root=_REAL_CONFIG_ROOT)
    assert cfg.embeddings.dimension == 384
    assert cfg.embeddings.model == "all-MiniLM-L6-v2"


def test_env_override_changes_log_level(monkeypatch):
    monkeypatch.setenv("RAGLAB_LOG_LEVEL", "DEBUG")
    cfg = load_config(config_root=_REAL_CONFIG_ROOT)
    assert cfg.log_level == "DEBUG"


def test_env_override_nested_key(monkeypatch):
    monkeypatch.setenv("RAGLAB_RETRIEVAL__TOP_K", "10")
    cfg = load_config(config_root=_REAL_CONFIG_ROOT)
    assert cfg.retrieval.top_k == 10


def test_extra_dict_overrides():
    cfg = load_config(
        config_root=_REAL_CONFIG_ROOT,
        extra={"log_level": "WARNING", "retrieval": {"top_k": 3}},
    )
    assert cfg.log_level == "WARNING"
    assert cfg.retrieval.top_k == 3


def test_missing_profile_file_falls_back_gracefully(tmp_path):
    """A missing profile file should not crash — just use defaults."""
    # Create a minimal config dir with an unknown profile
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "default.yaml").write_text("llm:\n  profile: nonexistent\n")
    (cfg_dir / "llm").mkdir()
    (cfg_dir / "embeddings").mkdir()
    (cfg_dir / "vectordb").mkdir()

    cfg = load_config(config_root=cfg_dir)
    assert cfg.llm.profile == "nonexistent"  # kept, just no extra keys
