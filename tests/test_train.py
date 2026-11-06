import json
import os

from src import registry, train


def test_train_persists_and_promotes(cfg):
    meta = train.train(cfg)
    model_dir = os.path.join(cfg["registry"]["dir"], meta["version"])
    assert os.path.exists(os.path.join(model_dir, "model.pkl"))
    assert os.path.exists(os.path.join(model_dir, "metadata.json"))

    with open(os.path.join(model_dir, "metadata.json")) as fh:
        stored = json.load(fh)
    assert stored["metric"]["r2"] > cfg["metric"]["threshold"]

    model, metadata, payload = registry.load_current(cfg)
    assert payload["version"] == meta["version"]
    assert metadata["version"] == meta["version"]


def test_unknown_model_type_raises(cfg):
    cfg["model"]["type"] = "not.a.real.Model"
    try:
        train.build_model(cfg)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "not.a.real.Model" in str(exc)
