"""Evaluation: the key metric (regression R2 + RMSE)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate(model, X, y) -> dict:
    preds = model.predict(X)
    return {
        "r2": float(r2_score(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
    }
