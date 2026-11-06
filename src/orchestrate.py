"""The closed loop: refresh -> evaluate -> detect drift -> retrain -> promote.

Run once (``python -m src.orchestrate``) or on a schedule
(``python -m src.orchestrate --loop``). Every run appends an event to ``runs/``.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from .config import load_config
from . import data as data_mod
from . import drift as drift_mod
from . import evaluate as evaluate_mod
from . import registry as registry_mod
from . import train as train_mod


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(cfg: dict, event: dict) -> str:
    runs_dir = cfg["runs"]["dir"]
    os.makedirs(runs_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    event["timestamp"] = _now()
    path = os.path.join(runs_dir, f"{stamp}_{event['type']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(event, fh, indent=2)
    return path


def run_once(cfg: dict, inject_drift: bool = False) -> dict:
    data_mod.ensure_dataset(cfg)

    model, meta, _payload = registry_mod.load_current(cfg)
    if model is None:
        meta = train_mod.train(cfg, promote=True)
        event = {
            "type": "initial_train",
            "version": meta["version"],
            "metric": meta["metric"],
            "note": "no model promoted yet; trained and promoted a baseline",
        }
        path = log_event(cfg, event)
        print(f"[orchestrate] initial_train -> {meta['version']} (r2={meta['metric']['r2']:.4f})")
        return {"status": "initial_train", "version": meta["version"], "log": path}

    batch = data_mod.refresh(cfg, inject_drift=inject_drift)
    Xb, yb = data_mod.split_xy(batch, cfg)

    metrics = evaluate_mod.evaluate(model, Xb, yb)

    mq_cfg = cfg["drift"]["model_quality"]
    dd_cfg = cfg["drift"]["data"]
    mq = drift_mod.model_quality_drift(metrics, mq_cfg["threshold"]) if mq_cfg.get("enabled", True) else {"drifted": False}
    dd = (
        drift_mod.data_drift(meta.get("baseline_psi", {}), Xb, dd_cfg["psi_threshold"])
        if dd_cfg.get("enabled", True)
        else {"drifted": False, "max_psi": None}
    )
    drifted = bool(mq["drifted"] or dd["drifted"])

    print(f"[orchestrate] batch r2={metrics['r2']:.4f} rmse={metrics['rmse']:.4f} | "
          f"quality_drift={mq['drifted']} (r2<{mq['threshold']}) | "
          f"data_drift={dd['drifted']} (psi={dd.get('max_psi')})")

    if not drifted:
        event = {
            "type": "no_drift",
            "version": meta["version"],
            "metric": metrics,
            "model_quality_drift": mq,
            "data_drift": dd,
        }
        path = log_event(cfg, event)
        return {"status": "no_drift", "version": meta["version"], "log": path, "metric": metrics}

    comparison = train_mod.retrain_and_compare(cfg, model)
    if comparison["promoted"]:
        registry_mod.promote(cfg, comparison["version"])

    event = {
        "type": "retrain",
        "drifted": True,
        "model_quality_drift": mq,
        "data_drift": dd,
        "current_version": meta["version"],
        "new_version": comparison["version"],
        "current_metrics": comparison["current_metrics"],
        "new_metrics": comparison["new_metrics"],
        "promoted": comparison["promoted"],
    }
    path = log_event(cfg, event)

    print(
        f"[orchestrate] drifted -> retrain {comparison['version']} "
        f"(current r2={comparison['current_metrics']['r2']:.4f} vs new r2={comparison['new_metrics']['r2']:.4f}) -> "
        f"{'PROMOTED' if comparison['promoted'] else 'KEPT CURRENT'}"
    )
    return {
        "status": "promoted" if comparison["promoted"] else "retrained_not_promoted",
        "version": comparison["version"] if comparison["promoted"] else meta["version"],
        "log": path,
        "comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one orchestration cycle (or loop)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--inject-drift", action="store_true", help="force a distribution shift in the refresh")
    parser.add_argument("--loop", action="store_true", help="run on a recurring schedule")
    parser.add_argument("--interval", type=int, default=None, help="loop interval in minutes")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.loop:
        from apscheduler.schedulers.blocking import BlockingScheduler

        interval = args.interval or cfg.get("scheduler", {}).get("interval_minutes", 5)
        scheduler = BlockingScheduler()
        scheduler.add_job(run_once, "interval", minutes=interval, args=[cfg, args.inject_drift])
        print(f"[orchestrate] scheduler started, every {interval} minute(s)")
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            pass
    else:
        print(json.dumps(run_once(cfg, inject_drift=args.inject_drift), indent=2))


if __name__ == "__main__":
    main()
