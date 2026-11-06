from src import data


def test_generate_synthetic_is_deterministic():
    a = data.generate_synthetic(n_samples=100, seed=42)
    b = data.generate_synthetic(n_samples=100, seed=42)
    assert a.equals(b)
    assert "target" in a.columns
    assert len(a) == 100


def test_refresh_appends_rows(cfg):
    data.reset_dataset(cfg)
    before = len(data.load_data(cfg))
    data.refresh(cfg)
    after = len(data.load_data(cfg))
    assert after == before + cfg["data"]["refresh"]["batch_size"]


def test_refresh_injects_drift(cfg):
    data.reset_dataset(cfg)
    baseline = data.load_data(cfg)
    shifted = data.refresh(cfg, inject_drift=True)
    # every feature should have moved by ~shift_magnitude
    for c in [f"f{i}" for i in range(10)]:
        assert shifted[c].mean() > baseline[c].mean() + 2.0
