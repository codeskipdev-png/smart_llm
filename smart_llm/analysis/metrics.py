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


# --------------------------------------------------------------------------- #
# Uncertainty and significance (numpy-only, deterministic).
#
# Reviewers require intervals and paired tests, not bare point estimates. These
# helpers operate on per-sample arrays so they compose with any of the metrics
# above and stay reproducible under a fixed seed.
# --------------------------------------------------------------------------- #
def bootstrap_ci(sample: np.ndarray, stat_fn=None, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for a statistic of a 1-D per-sample array.

    Returns (point, lo, hi). ``stat_fn`` defaults to the mean. The resample
    indices are drawn once and reused across the array, so paired variants can
    share the scheme; here we only need the marginal CI.
    """
    sample = np.asarray(sample, dtype=np.float64)
    n = len(sample)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    stat_fn = stat_fn or (lambda a: float(np.mean(a)))
    point = float(stat_fn(sample))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([stat_fn(sample[i]) for i in idx], dtype=np.float64)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return point, lo, hi


def ci_halfwidth(sample: np.ndarray, **kw) -> float:
    """Symmetric-ish half-width (hi-lo)/2 of the bootstrap CI, for '± e' cells."""
    _, lo, hi = bootstrap_ci(sample, **kw)
    if lo != lo or hi != hi:
        return float("nan")
    return (hi - lo) / 2.0


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, stat_fn=None,
                          n_boot: int = 10000, alpha: float = 0.05,
                          seed: int = 0):
    """Paired bootstrap for stat(a) - stat(b) on aligned per-sample arrays.

    Resamples sample *indices* (not a and b independently), preserving pairing.
    Returns dict(diff, lo, hi, p_value) where p is a two-sided bootstrap p-value
    for H0: diff = 0 (fraction of resamples on the opposite side of 0, doubled).
    Use for mean-regret or oracle-agreement comparisons between two policies.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b) or len(a) == 0:
        return {"diff": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p_value": float("nan")}
    stat_fn = stat_fn or (lambda x: float(np.mean(x)))
    n = len(a)
    diff = float(stat_fn(a) - stat_fn(b))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = np.array([stat_fn(a[i]) - stat_fn(b[i]) for i in idx], dtype=np.float64)
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    # two-sided bootstrap p-value about 0
    frac_le = float(np.mean(diffs <= 0.0))
    frac_ge = float(np.mean(diffs >= 0.0))
    p = min(1.0, 2.0 * min(frac_le, frac_ge))
    return {"diff": diff, "lo": lo, "hi": hi, "p_value": p}


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired McNemar test on two per-sample correctness masks (0/1).

    Compares two classifiers on the *same* test samples. Uses the exact binomial
    two-sided p on discordant pairs when they are few, and a continuity-corrected
    chi-square approximation when many. Returns dict(n01, n10, statistic, p_value).
    """
    a = np.asarray(correct_a, dtype=np.int64)
    b = np.asarray(correct_b, dtype=np.int64)
    if len(a) != len(b) or len(a) == 0:
        return {"n01": 0, "n10": 0, "statistic": float("nan"),
                "p_value": float("nan")}
    n01 = int(np.sum((a == 0) & (b == 1)))   # a wrong, b right
    n10 = int(np.sum((a == 1) & (b == 0)))   # a right, b wrong
    nd = n01 + n10
    if nd == 0:
        return {"n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0}
    if nd <= 100:                            # exact two-sided binomial (p=0.5)
        from math import comb
        k = min(n01, n10)
        tail = sum(comb(nd, i) for i in range(0, k + 1)) * (0.5 ** nd)
        p = min(1.0, 2.0 * tail)
        stat = float(min(n01, n10))
    else:                                    # chi-square with continuity correction
        stat = (abs(n01 - n10) - 1.0) ** 2 / nd
        # survival of chi-square with 1 dof = erfc(sqrt(stat/2))
        from math import erfc, sqrt
        p = float(erfc(sqrt(stat / 2.0)))
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": float(p)}


def sig_marker(p_value: float, alpha: float = 0.05) -> str:
    """'*' if significant, 'n.s.' otherwise, '' if undefined — for table cells."""
    if p_value is None or p_value != p_value:
        return ""
    return "*" if p_value < alpha else "n.s."
