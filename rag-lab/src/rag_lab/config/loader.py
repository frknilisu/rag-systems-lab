"""Layered config loader: default.yaml → profile YAMLs → env vars → validated model."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from rag_lab.config.schema import RagLabConfig

# Load .env on first import so env vars are available during config construction.
load_dotenv()

_CONFIG_ROOT = Path(__file__).parents[3] / "config"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply RAGLAB_* environment variables.

    Maps RAGLAB_LLM__PROFILE → raw["llm"]["profile"], etc.
    Double underscore separates nested keys.
    """
    prefix = "RAGLAB_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        target = raw
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return raw


def load_config(
    config_root: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> RagLabConfig:
    """Build a fully-merged, validated RagLabConfig.

    Resolution order (last wins):
      1. config/default.yaml
      2. config/llm/<profile>.yaml      (merged into raw["llm"])
      3. config/embeddings/<profile>.yaml (merged into raw["embeddings"])
      4. config/vectordb/<profile>.yaml  (merged into raw["vectordb"])
      5. RAGLAB_* environment variables
      6. `extra` dict (programmatic overrides, useful in tests)
    """
    root = config_root or _CONFIG_ROOT
    raw: dict[str, Any] = _load_yaml(root / "default.yaml")

    # Apply env overrides early so RAGLAB_LLM__PROFILE etc. influence which
    # profile YAML is loaded in the next step.
    raw = _apply_env_overrides(raw)

    # Pull and resolve profile files (profile may have been changed by env).
    for section, sub_dir in [
        ("llm", "llm"),
        ("embeddings", "embeddings"),
        ("vectordb", "vectordb"),
    ]:
        profile = raw.get(section, {}).get("profile", "")
        if profile:
            profile_path = root / sub_dir / f"{profile}.yaml"
            profile_data = _load_yaml(profile_path)
            # Profile data provides defaults; env-set values already in raw win.
            raw[section] = _deep_merge(profile_data, raw.get(section, {}))

    if extra:
        raw = _deep_merge(raw, extra)

    return RagLabConfig.model_validate(raw)
