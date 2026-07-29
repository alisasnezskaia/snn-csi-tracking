from __future__ import annotations

from pathlib import Path

import yaml


def load_config(*paths: str | Path) -> dict:
    """Load and shallow-merge one or more YAML config files, later files taking precedence."""
    merged: dict = {}
    for path in paths:
        with open(path) as f:
            merged.update(yaml.safe_load(f) or {})
    return merged
