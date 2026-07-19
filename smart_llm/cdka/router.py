"""SMART-LLM router (Confidence-Driven Knowledge Arbitration).

Routing rule (no double inference — uses only h_L, mu_K, B_pred, C_i):

    RUS(x, K)      = alpha * sim(x, K) + beta * B_pred
    ΔC(x)          = calibrated(RUS) - C_i
    decision(x)    = retrieve  if ΔC > threshold  else  trust internal model

alpha/beta are tuned on validation (grid, beta = 1 - alpha) to maximise agreement
with the oracle ``1[Loss_r < Loss_p]`` (or, if losses are supplied, to minimise
regret). ``sim`` and ``B_pred`` are z-standardised (val stats) before mixing so
the weights are comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import Config
from .calibration import Calibrator


def uncertainty(norm_entropy: np.ndarray, ci: np.ndarray, lam: float) -> np.ndarray:
    """U(x) = lam * normalised_entropy + (1 - lam) * (1 - C_i)."""
    norm_entropy = np.asarray(norm_entropy, dtype=np.float64)
    ci = np.asarray(ci, dtype=np.float64)
    return (lam * norm_entropy + (1.0 - lam) * (1.0 - ci)).astype(np.float32)


@dataclass
class Router:
    cfg: Config
    alpha: float = 0.5
    beta: float = 0.5
    threshold: float = 0.0
    calibrator: Optional[Calibrator] = None
    # z-standardisation stats for sim / B_pred (fit on val)
    _sim_mu: float = 0.0
    _sim_sd: float = 1.0
    _b_mu: float = 0.0
    _b_sd: float = 1.0
    fit_report: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def _rus(self, sim: np.ndarray, bpred: np.ndarray) -> np.ndarray:
        sim_z = (np.asarray(sim) - self._sim_mu) / self._sim_sd
        b_z = (np.asarray(bpred) - self._b_mu) / self._b_sd
        return (self.alpha * sim_z + self.beta * b_z).astype(np.float32)

    def fit(self, sim: np.ndarray, bpred: np.ndarray, ci: np.ndarray,
            oracle: np.ndarray,
            loss_p: Optional[np.ndarray] = None,
            loss_r: Optional[np.ndarray] = None,
            fixed_alpha: Optional[float] = None) -> "Router":
        sim = np.asarray(sim, dtype=np.float64)
        bpred = np.asarray(bpred, dtype=np.float64)
        ci = np.asarray(ci, dtype=np.float64)
        oracle = np.asarray(oracle, dtype=np.int64)

        self._sim_mu, self._sim_sd = float(sim.mean()), float(sim.std() + 1e-6)
        self._b_mu, self._b_sd = float(bpred.mean()), float(bpred.std() + 1e-6)
        self.threshold = self.cfg.router.decision_threshold

        if fixed_alpha is not None:
            alphas = [fixed_alpha]
        elif not self.cfg.router.tune_alpha_beta:
            alphas = [self.cfg.router.alpha]
        else:
            alphas = np.linspace(0.0, 1.0, self.cfg.router.tune_grid)
        best = None
        for alpha in alphas:
            self.alpha, self.beta = float(alpha), float(1.0 - alpha)
            rus = self._rus(sim, bpred)
            cal = Calibrator(self.cfg.router.calibration).fit(rus, oracle)
            delta = cal.transform(rus) - ci
            decision = (delta > self.threshold).astype(np.int64)
            agreement = float(np.mean(decision == oracle))
            score = agreement
            if loss_p is not None and loss_r is not None:
                chosen_loss = np.where(decision == 1, loss_r, loss_p)
                oracle_loss = np.minimum(loss_p, loss_r)
                regret = float(np.mean(chosen_loss - oracle_loss))
                score = agreement - regret  # prefer high agreement, low regret
            if best is None or score > best["score"]:
                best = dict(alpha=self.alpha, beta=self.beta, calibrator=cal,
                            agreement=agreement, score=score)
        self.alpha, self.beta = best["alpha"], best["beta"]
        self.calibrator = best["calibrator"]
        self.fit_report = {"alpha": self.alpha, "beta": self.beta,
                           "val_agreement": best["agreement"]}
        return self

    # ------------------------------------------------------------------ #
    def predict(self, sim: np.ndarray, bpred: np.ndarray, ci: np.ndarray) -> dict:
        assert self.calibrator is not None, "call .fit() first"
        rus = self._rus(sim, bpred)
        calibrated = self.calibrator.transform(rus)
        delta = calibrated.astype(np.float64) - np.asarray(ci, dtype=np.float64)
        decision = (delta > self.threshold).astype(np.int64)
        return {
            "rus": rus.astype(np.float32),
            "calibrated_rus": calibrated.astype(np.float32),
            "delta_c": delta.astype(np.float32),
            "decision": decision,
        }
