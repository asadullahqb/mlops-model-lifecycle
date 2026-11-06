import json
import os

from src import orchestrate, registry


def test_closed_loop_initial_then_drift_promotes(cfg):
    # first run: trains + promotes a baseline
    first = orchestrate.run_once(cfg)
    assert first["status"] == "initial_train"
    v1 = first["version"]

    # second run with injected drift: detect -> retrain -> promote
    second = orchestrate.run_once(cfg, inject_drift=True)
    assert second["status"] == "promoted"
    assert second["version"] != v1

    _, _, payload = registry.load_current(cfg)
    assert payload["version"] == second["version"]

    # an event file was written for each run
    runs = os.listdir(cfg["runs"]["dir"])
    assert len(runs) == 2
    events = [
        json.load(open(os.path.join(cfg["runs"]["dir"], f)))
        for f in runs
    ]
    types = sorted(e["type"] for e in events)
    assert types == ["initial_train", "retrain"]


def test_no_drift_keeps_current(cfg):
    orchestrate.run_once(cfg)  # initial train
    result = orchestrate.run_once(cfg)  # normal refresh, no injected drift
    assert result["status"] == "no_drift"
