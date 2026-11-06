"""Config loading.

All modules take a plain ``dict`` produced by :func:`load_config`, which makes
the code easy to unit-test (a test can construct a dict pointing at a temp dir)
while keeping a single ``config.yaml`` as the source of truth for real runs.
"""

from __future__ import annotations

import os

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load ``config.yaml`` and resolve relative paths against its directory."""
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    base = os.path.dirname(path)

    def resolve(p: str | None) -> str | None:
        if p is None:
            return None
        return p if os.path.isabs(p) else os.path.join(base, p)

    cfg.setdefault("data", {})
    cfg["data"]["source"] = resolve(cfg["data"].get("source", "data/dataset.csv"))

    cfg.setdefault("registry", {})
    cfg["registry"]["dir"] = resolve(cfg["registry"].get("dir", "models"))

    cfg.setdefault("runs", {})
    cfg["runs"]["dir"] = resolve(cfg["runs"].get("dir", "runs"))

    return cfg
