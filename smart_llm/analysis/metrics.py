"""Evaluation metrics (numpy only, so they are unit-testable without sklearn).

Covers everything the paper reports: RBE regression quality (R2), router quality
against the oracle (agreement, regret), classification (accuracy, macro-F1),
confidence calibration (ECE), and retrieval frequency.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def r2_score(pred: np.ndarray, true: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(true) == 0:
        return float("nan")
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def accuracy(pred: np.ndarray, label: np.ndarray) -> float:
    pred = np.asarray(pred); label = np.asarray(label)
    if len(label) == 0:
        return float("nan")
    return float(np.mean(pred == label))


def macro_f1(pred: np.ndarray, label: np.ndarray, n_classes: int = None) -> float:
    pred = np.asarray(pred, dtype=np.int64)
    label = np.asarray(label, dtype=np.int64)
    if len(label) == 0:
        return float("nan")
    classes = range(n_classes) if n_classes else np.unique(
        np.concatenate([pred, label]))
    f1s = []
    for c in classes:
        tp = int(np.sum((pred == c) & (label == c)))
        fp = int(np.sum((pred == c) & (label != c)))
        fn = int(np.sum((pred != c) & (label == c)))
        if tp + fp == 0 and tp + fn == 0:
            continue  # class absent from both -> skip
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else float("nan")


def routing_agreement(decision: np.ndarray, oracle: np.ndarray) -> float:
    decision = np.asarray(decision, dtype=np.int64)
    oracle = np.asarray(oracle, dtype=np.int64)
    if len(oracle) == 0:
        return float("nan")
    return float(np.mean(decision == oracle))


def regret_per_sample(decision: np.ndarray, loss_p: np.ndarray,
                      loss_r: np.ndarray) -> np.ndarray:
    """chosen_loss - oracle_loss  (>= 0). decision=1 => use retrieval."""
    decision = np.asarray(decision, dtype=np.int64)
    loss_p = np.asarray(loss_p, dtype=np.float64)
    loss_r = np.asarray(loss_r, dtype=np.float64)
    chosen = np.where(decision == 1, loss_r, loss_p)
    oracle = np.minimum(loss_p, loss_r)
    return chosen - oracle


def mean_regret(decision, loss_p, loss_r) -> float:
    reg = regret_per_sample(decision, loss_p, loss_r)
    return float(np.mean(reg)) if len(reg) else float("nan")


def retrieval_frequency(decision: np.ndarray) -> float:
    decision = np.asarray(decision, dtype=np.int64)
    if len(decision) == 0:
        return float("nan")
    return float(np.mean(decision))


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE for a confidence signal vs binary correctness."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if not np.any(m):
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def classification_summary(pred: np.ndarray, label: np.ndarray,
                           n_classes: int = None) -> Dict[str, float]:
    return {"accuracy": accuracy(pred, label),
            "macro_f1": macro_f1(pred, label, n_classes)}
