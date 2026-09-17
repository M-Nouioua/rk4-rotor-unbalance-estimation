"""Config loading + small shared helpers."""
from __future__ import annotations

import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "rig.yaml"


def load_config(path: str | pathlib.Path = DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
