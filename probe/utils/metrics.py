"""Metrics: variance-weighted R², MSE, mean cosine similarity for direction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def r2_variance_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sklearn r2_score with multioutput='variance_weighted' (scalar). Falls back
    to scalar R² for 1D targets (multioutput is a no-op there)."""
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        return float(r2_score(y_true, y_pred))
    return float(r2_score(y_true, y_pred, multioutput="variance_weighted"))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(((y_true - y_pred) ** 2).mean())


def mean_cosine(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean cosine similarity between rows; assumes y_true is unit (caller's responsibility)."""
    p = y_pred / np.clip(np.linalg.norm(y_pred, axis=1, keepdims=True), 1e-12, None)
    return float((p * y_true).sum(axis=1).mean())
