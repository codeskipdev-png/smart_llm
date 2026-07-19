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


def precision_recall_f1_macro(pred: np.ndarray, label: np.ndarray,
                              n_classes: int = None) -> dict:
    """Macro-averaged precision / recall / F1 (classes absent from both skipped)."""
    pred = np.asarray(pred, dtype=np.int64)
    label = np.asarray(label, dtype=np.int64)
    if len(label) == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    classes = range(n_classes) if n_classes else np.unique(
        np.concatenate([pred, label]))
    ps, rs, fs = [], [], []
    for c in classes:
        tp = int(np.sum((pred == c) & (label == c)))
        fp = int(np.sum((pred == c) & (label != c)))
        fn = int(np.sum((pred != c) & (label == c)))
        if tp + fp == 0 and tp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ps.append(prec); rs.append(rec); fs.append(f1)
    m = lambda a: float(np.mean(a)) if a else float("nan")
    return {"precision": m(ps), "recall": m(rs), "f1": m(fs)}


def binary_prf(decision: np.ndarray, oracle: np.ndarray) -> dict:
    """Precision/recall/F1 treating 'retrieve' (=1) as the positive class."""
    decision = np.asarray(decision, dtype=np.int64)
    oracle = np.asarray(oracle, dtype=np.int64)
    tp = int(np.sum((decision == 1) & (oracle == 1)))
    fp = int(np.sum((decision == 1) & (oracle == 0)))
    fn = int(np.sum((decision == 0) & (oracle == 1)))
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and (prec + rec) else float("nan"))
    return {"precision": prec, "recall": rec, "f1": f1}


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if len(true) == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - true)))


def brier_score(conf: np.ndarray, correct: np.ndarray) -> float:
    """Confidence Brier score: mean((conf - correct)^2), correct in {0,1}."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return float("nan")
    return float(np.mean((conf - correct) ** 2))


def reliability_curve(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """Return (bin_conf, bin_acc, bin_count) for a reliability diagram."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bc, ba, cnt = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if not np.any(m):
            continue
        bc.append(float(conf[m].mean()))
        ba.append(float(correct[m].mean()))
        cnt.append(int(m.sum()))
    return np.array(bc), np.array(ba), np.array(cnt)


def difficulty_tertiles(score: np.ndarray) -> np.ndarray:
    """Split by tertiles of a difficulty score (higher=harder) into 0/1/2 =
    Easy/Medium/Hard. Returns an int label array."""
    score = np.asarray(score, dtype=np.float64)
    if len(score) == 0:
        return np.array([], dtype=np.int64)
    q1, q2 = np.quantile(score, [1 / 3, 2 / 3])
    lab = np.zeros(len(score), dtype=np.int64)
    lab[score > q1] = 1
    lab[score > q2] = 2
    return lab


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
