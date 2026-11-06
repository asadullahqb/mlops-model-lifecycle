"""Versioned model registry + atomic promotion.

Layout::

    models/
      current.json          # pointer to the live model (atomically swapped)
      <version>/
        model.pkl
        metadata.json
"""

from __future__ import annotations

import json
import os

import joblib


def current_file(cfg: dict) -> str:
    return os.path.join(cfg["registry"]["dir"], "current.json")


def promote(cfg: dict, version: str) -> dict:
    """Atomically point the registry at ``version`` (temp file + os.replace)."""
    registry_dir = cfg["registry"]["dir"]
    os.makedirs(registry_dir, exist_ok=True)

    payload = {
        "version": version,
        "model_path": os.path.join(registry_dir, version, "model.pkl"),
        "metadata_path": os.path.join(registry_dir, version, "metadata.json"),
    }

    cf = current_file(cfg)
    tmp = cf + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, cf)
    return payload


def load_current(cfg: dict) -> tuple:
    """Return ``(model, metadata, payload)`` or ``(None, None, None)``."""
    cf = current_file(cfg)
    if not os.path.exists(cf):
        return None, None, None
    with open(cf, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    model = joblib.load(payload["model_path"])
    with open(payload["metadata_path"], "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    return model, metadata, payload


def list_versions(cfg: dict) -> list[str]:
    registry_dir = cfg["registry"]["dir"]
    if not os.path.isdir(registry_dir):
        return []
    return sorted(
        d for d in os.listdir(registry_dir)
        if os.path.isdir(os.path.join(registry_dir, d))
    )
