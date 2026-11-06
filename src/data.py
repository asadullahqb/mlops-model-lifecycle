"""Data loading + refresh.

The reference implementation uses a deterministic, bundled synthetic tabular
regression dataset so the whole lifecycle runs offline with zero external
dependencies.

To point this at a REAL data source, implement :func:`refresh` against your
own store (S3, Postgres, Snowflake, an event stream, ...). The contract is
simple: ``refresh(cfg)`` must return a ``pandas.DataFrame`` containing the new
batch and must append it to ``cfg["data"]["source"]``. Everything downstream
(orchestrate) only consumes that returned DataFrame.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import load_config


def feature_names(n: int = 10) -> list[str]:
    return [f"f{i}" for i in range(n)]


def target_fn(X: pd.DataFrame) -> np.ndarray:
    """Deterministic, non-linear target so GradientBoosting has signal to learn."""
    X = pd.DataFrame(X)
    cols = list(X.columns)

    def col(i: int) -> np.ndarray:
        return X[cols[i]].to_numpy()

    f0, f1, f2, f3, f4, f5, f6, f7, f8, f9 = (col(i) for i in range(10))
    return (
        5.0 * f0
        + 3.0 * f1**2
        + 2.5 * f2 * f3
        + 2.0 * np.sin(f4 * 6.0)
        + 1.5 * f5
        + 2.0 * f6
        + 1.5 * f7 * f8
        + 0.5 * f9
        - 2.0
    )


def generate_synthetic(
    n_samples: int = 1000, n_features: int = 10, seed: int = 42, noise: float = 0.5
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = feature_names(n_features)
    X = pd.DataFrame(rng.uniform(0.0, 1.0, size=(n_samples, n_features)), columns=cols)
    y = target_fn(X) + rng.normal(0.0, noise, size=n_samples)
    X["target"] = y
    return X


def reset_dataset(cfg: dict) -> pd.DataFrame:
    """Regenerate the baseline dataset from scratch (deterministic)."""
    src = cfg["data"]["source"]
    os.makedirs(os.path.dirname(src), exist_ok=True)
    df = generate_synthetic(
        n_samples=cfg["data"].get("n_samples", 1000),
        seed=cfg["data"].get("seed", 42),
    )
    df.to_csv(src, index=False)
    return df


def ensure_dataset(cfg: dict) -> pd.DataFrame:
    """Create the baseline dataset if it does not exist yet."""
    src = cfg["data"]["source"]
    if os.path.exists(src):
        return load_data(cfg)
    return reset_dataset(cfg)


def load_data(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(cfg["data"]["source"])


def split_xy(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    target = cfg["data"]["target"]
    feats = cfg["data"].get("features") or [c for c in df.columns if c != target]
    return df[feats], df[target]


def _apply_drift(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Inject a covariate shift into a batch (used only for the demo)."""
    d = cfg["data"].get("drift") or {}
    mag = float(d.get("shift_magnitude", 3.0))
    specified = d.get("shifted_features", "all")
    target = cfg["data"]["target"]
    if specified is None or specified == "all":
        feats = [c for c in df.columns if c != target]
    else:
        feats = [c for c in specified if c in df.columns]

    out = df.copy()
    for c in feats:
        out[c] = out[c] + mag
    return out


def refresh(cfg: dict, inject_drift: bool | None = None) -> pd.DataFrame:
    """Simulate a new incoming batch and append it to the source CSV.

    ``inject_drift`` overrides ``data.drift.enabled`` in config so a demo can
    force a distribution shift without editing the file.
    """
    d = cfg["data"].get("refresh") or {}
    batch_size = int(d.get("batch_size", 200))
    seed = int(d.get("seed", 7))

    df = generate_synthetic(n_samples=batch_size, seed=seed)
    if inject_drift is None:
        inject_drift = bool((cfg["data"].get("drift") or {}).get("enabled", False))
    if inject_drift:
        df = _apply_drift(df, cfg)

    src = cfg["data"]["source"]
    os.makedirs(os.path.dirname(src), exist_ok=True)
    df.to_csv(src, mode="a", header=not os.path.exists(src), index=False)
    return df


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Dataset utilities")
    parser.add_argument("command", choices=["reset", "refresh"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--inject-drift", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.command == "reset":
        df = reset_dataset(cfg)
        print(f"reset {cfg['data']['source']} -> {len(df)} rows")
    else:
        batch = refresh(cfg, inject_drift=args.inject_drift)
        print(f"appended {len(batch)} rows to {cfg['data']['source']}")


if __name__ == "__main__":
    _cli()
