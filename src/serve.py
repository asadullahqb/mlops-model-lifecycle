"""FastAPI serving layer.

Serves whatever model ``models/current.json`` currently points at. The pointer
file is re-read (and the model hot-reloaded) whenever it changes on disk, so a
promotion from ``orchestrate`` is picked up without restarting the process.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_config
from .registry import current_file


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    prediction: float
    version: str


class ModelRegistry:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model = None
        self._version: str | None = None
        self._mtime: float | None = None

    def _ensure_loaded(self):
        cf = current_file(self.cfg)
        if not os.path.exists(cf):
            return None
        mtime = os.path.getmtime(cf)
        if self._model is None or mtime != self._mtime:
            with open(cf, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self._model = joblib.load(payload["model_path"])
            self._version = payload["version"]
            self._mtime = mtime
        return self._model

    def predict(self, X: np.ndarray) -> tuple[float, str]:
        model = self._ensure_loaded()
        if model is None:
            raise HTTPException(status_code=503, detail="no model promoted yet; run `python -m src.train`")
        if hasattr(model, "feature_names_in_"):
            X = pd.DataFrame(X, columns=list(model.feature_names_in_))
        return float(model.predict(X)[0]), self._version


def create_app(cfg: dict) -> FastAPI:
    registry = ModelRegistry(cfg)
    app = FastAPI(title="mlops-model-lifecycle")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        X = np.asarray([req.features], dtype=float)
        prediction, version = registry.predict(X)
        return PredictResponse(prediction=prediction, version=version)

    return app


app = create_app(load_config(os.environ.get("MLOPS_CONFIG", "config.yaml")))
