"""Config loader. Every threshold in the project comes from config.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    md = Path(cfg["data"]["matches_dir"])
    if not md.is_absolute():
        md = (path.parent / md).resolve()
    cfg["data"]["matches_dir"] = md
    return cfg
