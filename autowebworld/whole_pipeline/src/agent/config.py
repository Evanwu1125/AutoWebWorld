"""Shared model configuration loader for the agent framework."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_THIS_DIR = Path(__file__).resolve().parent  # src/agent/
DEFAULT_CONFIG_PATH = _THIS_DIR / "config" / "fsm.config.yaml"


def load_model_config(
    config_path: Path | str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Load model config from yaml.

    Args:
        config_path: Path to config file (default: config/fsm.config.yaml).
        model_name: Model to look up. If None, uses default_model from config.

    Returns dict with keys: model, api_base, api_key, max_output_tokens.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    target = model_name or raw.get("default_model", "")
    for entry in raw.get("model_list", []):
        if entry.get("model_name") == target:
            params = entry.get("litellm_params", {})
            info = entry.get("model_info", {})
            api_key = params.get("api_key")
            api_key_env = params.get("api_key_env")
            if not api_key and api_key_env:
                api_key = os.getenv(str(api_key_env))

            return {
                "model": params.get("model", target),
                "api_base": params.get("api_base"),
                "api_key": api_key,
                "max_output_tokens": info.get("max_output_tokens"),
            }
    return {}
