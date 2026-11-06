import pytest


def make_cfg(tmp_path):
    """Build a config dict pointing at a temp dir (mirrors config.yaml)."""
    return {
        "data": {
            "source": str(tmp_path / "data" / "dataset.csv"),
            "target": "target",
            "features": [f"f{i}" for i in range(10)],
            "n_samples": 400,
            "seed": 42,
            "refresh": {"batch_size": 100, "seed": 7},
            "drift": {"enabled": False, "shifted_features": "all", "shift_magnitude": 3.0},
        },
        "model": {
            "type": "sklearn.ensemble.GradientBoostingRegressor",
            "params": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
            "test_size": 0.25,
        },
        "metric": {"name": "r2", "threshold": 0.5},
        "drift": {
            "model_quality": {"enabled": True, "threshold": 0.5},
            "data": {"enabled": True, "method": "psi", "psi_threshold": 0.25},
        },
        "registry": {"dir": str(tmp_path / "models")},
        "runs": {"dir": str(tmp_path / "runs")},
        "scheduler": {"interval_minutes": 5},
    }


@pytest.fixture
def cfg(tmp_path):
    return make_cfg(tmp_path)
