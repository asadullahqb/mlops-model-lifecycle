"""Drift detection.

Two signals are computed:

* **model-quality drift** -- the key metric (R2) on a fresh batch drops below
  the configured threshold.
* **data drift** -- Population Stability Index (PSI) of each feature against the
  distribution the current model was trained on (stored as ``baseline_psi`` in
  its metadata).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-4


def compute_psi_bins(series, bins: int = 10) -> dict:
    """Compute a JSON-serialisable baseline histogram for one feature."""
    arr = np.asarray(series, dtype=float)
    counts, edges = np.histogram(arr, bins=bins)
    expected = counts / max(counts.sum(), 1)
    return {
        "bin_edges": [float(e) for e in edges],
        "expected": [float(p) for p in expected],
    }


def psi_from_baseline(baseline: dict, new_series) -> float:
    """PSI of ``new_series`` vs a stored baseline histogram."""
    edges = np.asarray(baseline["bin_edges"], dtype=float)
    expected = np.asarray(baseline["expected"], dtype=float)

    arr = np.asarray(new_series, dtype=float)
    counts, _ = np.histogram(arr, bins=edges)
    actual = counts / max(counts.sum(), 1)

    expected = np.clip(expected, EPS, None)
    actual = np.clip(actual, EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def data_drift(baseline_psi: dict, X_new, threshold: float) -> dict:
    """Compare each new-batch feature against the training-time baseline."""
    per_feature: dict[str, float] = {}
    max_psi = 0.0
    for col in X_new.columns:
        if col not in baseline_psi:
            continue
        psi = psi_from_baseline(baseline_psi[col], X_new[col])
        per_feature[col] = round(psi, 4)
        max_psi = max(max_psi, psi)
    return {
        "drifted": max_psi > threshold,
        "max_psi": round(max_psi, 4),
        "threshold": threshold,
        "per_feature": per_feature,
    }


def model_quality_drift(metrics: dict, threshold: float) -> dict:
    value = float(metrics.get("r2", 0.0))
    return {
        "drifted": value < threshold,
        "metric": "r2",
        "value": round(value, 4),
        "threshold": threshold,
    }
