'''
 # @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
 # @ Author: Pallab Maji
 # @ Create Time: 2025-10-30 17:10:00
 # @ Modified time: 2025-10-30 17:26:00
 # @ Description: Utility helpers for configuration loading and filesystem management used across the project.
'''

from __future__ import annotations

"""Shared utility helpers for configuration and filesystem management."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

LOGGER = logging.getLogger("gai_yolov12.utils")


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file from disk."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at root of config, got {type(data)!r}")
    return data


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Persist a JSON payload with deterministic formatting."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Dict[str, Any]:
    """Load JSON content as a dictionary."""
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)





