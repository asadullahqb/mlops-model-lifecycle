"""Training + persistence.

The model factory is the single seam for swapping in a real model: register a
new ``model.type`` in :data:`MODEL_FACTORIES` (or subclass it) and reference it
from ``config.yaml``. See the README for XGBoost (UHI) and churn examples.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from . import data as data_mod
from . import drift as drift_mod
from . import evaluate as evaluate_mod
from . import registry as registry_mod


def _xgboost(params: dict):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - exercised only if xgboost installed
        raise RuntimeError("model.type=xgboost.XGBRegressor requires `pip install xgboost`") from exc
    return XGBRegressor(**params)


MODEL_FACTORIES = {
    "sklearn.ensemble.GradientBoostingRegressor": lambda p: GradientBoostingRegressor(**p),
    "sklearn.ensemble.RandomForestRegressor": lambda p: RandomForestRegressor(**p),
    "sklearn.linear_model.LinearRegression": lambda p: LinearRegression(),
    "xgboost.XGBRegressor": _xgboost,
}


def build_model(cfg: dict):
    mtype = cfg.get("model", {}).get("type", "sklearn.ensemble.GradientBoostingRegressor")
    factory = MODEL_FACTORIES.get(mtype)
    if factory is None:
        raise ValueError(
            f"Unknown model.type '{mtype}'. "
            f"Available: {sorted(MODEL_FACTORIES)}. "
            "Register a new adapter in src/train.py:MODEL_FACTORIES."
        )
    return factory(dict(cfg.get("model", {}).get("params", {})))


def _new_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def _split(cfg: dict, X, y):
    params = cfg.get("model", {}).get("params", {})
    test_size = cfg.get("model", {}).get("test_size", 0.2)
    random_state = params.get("random_state", 42)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _persist(cfg: dict, model, X_train, metrics: dict, version: str | None = None) -> dict:
    version = version or _new_version()
    model_dir = os.path.join(cfg["registry"]["dir"], version)
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "model.pkl"))

    metadata = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric": metrics,
        "n_train": int(len(X_train)),
        "features": list(X_train.columns),
        "baseline_psi": {c: drift_mod.compute_psi_bins(X_train[c]) for c in X_train.columns},
    }
    with open(os.path.join(model_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata


def train(cfg: dict, promote: bool = True) -> dict:
    """Train on the current dataset and (optionally) promote."""
    df = data_mod.ensure_dataset(cfg)
    X, y = data_mod.split_xy(df, cfg)
    Xtr, Xte, ytr, yte = _split(cfg, X, y)

    model = build_model(cfg)
    model.fit(Xtr, ytr)
    metrics = evaluate_mod.evaluate(model, Xte, yte)

    metadata = _persist(cfg, model, Xtr, metrics)
    if promote:
        registry_mod.promote(cfg, metadata["version"])
    return metadata


def retrain_and_compare(cfg: dict, current_model) -> dict:
    """Retrain on the full dataset and compare against ``current_model`` on the
    SAME held-out split. Returns a report that includes a ``promoted`` flag."""
    df = data_mod.load_data(cfg)
    X, y = data_mod.split_xy(df, cfg)
    Xtr, Xte, ytr, yte = _split(cfg, X, y)

    new_model = build_model(cfg)
    new_model.fit(Xtr, ytr)

    new_metrics = evaluate_mod.evaluate(new_model, Xte, yte)
    current_metrics = evaluate_mod.evaluate(current_model, Xte, yte)

    metadata = _persist(cfg, new_model, Xtr, new_metrics)
    return {
        "version": metadata["version"],
        "new_metrics": new_metrics,
        "current_metrics": current_metrics,
        "promoted": new_metrics["r2"] > current_metrics["r2"],
    }


def _cli() -> None:
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metadata = train(cfg, promote=not args.no_promote)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    _cli()
