"""Calibrate the Retrieval Utility Score onto a probability-like [0, 1] scale so
it is directly comparable with the confidence ``C_i`` in the routing rule
``ΔC = calibrated(RUS) - C_i``.

Calibration is fit on the validation split, mapping RUS -> P(retrieval helps),
where the binary target is the oracle decision ``1[Loss_r < Loss_p]``.
"""
from __future__ import annotations

import numpy as np


class Calibrator:
    def __init__(self, method: str = "platt"):
        self.method = method
        self._model = None
        self._lo = 0.0
        self._hi = 1.0

    def fit(self, rus: np.ndarray, target: np.ndarray) -> "Calibrator":
        rus = np.asarray(rus, dtype=np.float64).reshape(-1, 1)
        target = np.asarray(target, dtype=np.int64).reshape(-1)
        # degenerate target (all 0 or all 1) -> fall back to a monotone min-max
        if len(np.unique(target)) < 2 or self.method == "minmax":
            self.method = "minmax"
            self._lo = float(rus.min())
            self._hi = float(rus.max()) + 1e-9
            return self
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self._model = IsotonicRegression(out_of_bounds="clip",
                                             y_min=0.0, y_max=1.0)
            self._model.fit(rus.reshape(-1), target.astype(float))
        else:  # platt (logistic)
            from sklearn.linear_model import LogisticRegression
            self.method = "platt"
            self._model = LogisticRegression(C=1.0, max_iter=1000)
            self._model.fit(rus, target)
        return self

    def transform(self, rus: np.ndarray) -> np.ndarray:
        rus = np.asarray(rus, dtype=np.float64)
        if self.method == "minmax":
            out = (rus - self._lo) / (self._hi - self._lo)
            return np.clip(out, 0.0, 1.0).astype(np.float32)
        if self.method == "isotonic":
            return self._model.predict(rus.reshape(-1)).astype(np.float32)
        proba = self._model.predict_proba(rus.reshape(-1, 1))[:, 1]
        return proba.astype(np.float32)
