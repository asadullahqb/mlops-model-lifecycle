import numpy as np

from src import drift


def _series(seed, shift=0.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=2000) + shift


def test_psi_low_for_same_distribution():
    baseline = drift.compute_psi_bins(_series(1))
    psi = drift.psi_from_baseline(baseline, _series(2))
    assert psi < 0.25


def test_psi_high_for_shifted_distribution():
    baseline = drift.compute_psi_bins(_series(1))
    psi = drift.psi_from_baseline(baseline, _series(2, shift=3.0))
    assert psi > 0.25


def test_model_quality_drift_flags_low_r2():
    assert drift.model_quality_drift({"r2": 0.1}, threshold=0.5)["drifted"] is True
    assert drift.model_quality_drift({"r2": 0.9}, threshold=0.5)["drifted"] is False
