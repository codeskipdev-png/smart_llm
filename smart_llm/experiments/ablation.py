"""Ablation (Analysis 7): isolate the contribution of each decision-time module.

All variants share the same frozen features, confidence probe, and RBE; only the
arbitration rule changes, so differences are attributable to the ablated module:

  * SMART (full)        RUS = alpha*sim + beta*B_pred, Platt-calibrated
  * - RBE               drop the predicted-benefit term (similarity-only routing)
  * - Calibration       replace Platt with an uncalibrated (min-max) map
  * Confidence-only     retrieve iff C_i < tau (no retrieval-utility signal)
  * Always / Never RAG  trivial references
  * Oracle              retrieve iff Loss_r < Loss_p (upper bound)

(UAAS is ablated separately in train_uaas: adaptive rank vs. static LoRA.)
Reports oracle agreement + precision/recall/F1 of the retrieve decision, mean
regret, end-task accuracy, and retrieval frequency on the test split.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..analysis import metrics as M
from ..cdka.router import Router
from ..config import add_config_args, config_from_args
from ..utils.io import atomic_write_csv
from ..utils.logging import get_logger
from ..utils.seed import seed_everything
from .cache import load_features
from .train_cdka import (_select_device, _split, _train_pooling,
                        apply_stable_benefit)

_log = get_logger("smart_llm.ablation")


def _eval(decision, oracle, loss_p, loss_r, pred_p, pred_r, label):
    smart_pred = np.where(decision == 1, pred_r, pred_p)
    b = M.binary_prf(decision, oracle)
    return {
        "oracle_agreement": M.routing_agreement(decision, oracle),
        "precision": b["precision"], "recall": b["recall"], "f1": b["f1"],
        "mean_regret": M.mean_regret(decision, loss_p, loss_r),
        "accuracy": M.accuracy(smart_pred, label),
        "retrieval_freq": M.retrieval_frequency(decision),
    }


def run(cfg, device="auto") -> pd.DataFrame:
    device = _select_device(device)
    pool = cfg.pooling.default
    feats = load_features(cfg.paths.cache_dir, cfg.data.dataset,
                          need_tokens=(pool == "attention"))
    apply_stable_benefit(feats, cfg)          # correct B_true from stored losses
    n = feats.n
    labels = feats.labels
    tr, va, te = _split(n, cfg.seed, cfg.data.val_fraction, cfg.data.val_fraction)

    art = _train_pooling(feats, pool, cfg, tr, va, device)
    clean = feats.conds["clean"]
    loss_p = feats.meta["loss_p"].to_numpy(dtype=np.float64)
    loss_r = clean["loss_r"].astype(np.float64)
    pred_p = feats.meta["pred_p"].to_numpy(dtype=np.int64)
    pred_r = clean["pred_r"].astype(np.int64)
    oracle = clean["oracle"]
    ci = art.ci
    bpred = art.rbe.predict(np.concatenate([art.pooled, clean["centroid"]], axis=1))

    def router_decision(fixed_alpha=None, calibration=None):
        saved = cfg.router.calibration
        if calibration is not None:
            cfg.router.calibration = calibration
        r = Router(cfg).fit(clean["sim"][va], bpred[va], ci[va], oracle[va],
                            loss_p[va], loss_r[va], fixed_alpha=fixed_alpha)
        cfg.router.calibration = saved
        return r.predict(clean["sim"], bpred, ci)["decision"]

    rows = []

    def add(name, decision):
        rows.append({"Variant": name, **_eval(
            decision[te], oracle[te], loss_p[te], loss_r[te],
            pred_p[te], pred_r[te], labels[te])})

    add("SMART (full)", art.router.predict(clean["sim"], bpred, ci)["decision"])
    add("- RBE (similarity only)", router_decision(fixed_alpha=1.0))
    add("- Calibration (raw RUS)", router_decision(calibration="minmax"))

    # confidence-only: retrieve iff C_i < tau (tau tuned on val for agreement)
    taus = np.quantile(ci[va], np.linspace(0.05, 0.95, 19))
    best_tau = max(taus, key=lambda t: M.routing_agreement(
        (ci[va] < t).astype(int), oracle[va]))
    add(f"Confidence-only (C_i<{best_tau:.2f})", (ci < best_tau).astype(int))

    add("Always RAG", np.ones(n, dtype=int))
    add("Never RAG", np.zeros(n, dtype=int))
    add("Oracle (upper bound)", oracle.astype(int))

    df = pd.DataFrame(rows)
    atomic_write_csv(df, f"{cfg.paths.tables_dir}/ablation_{cfg.data.dataset}.csv")
    _log.info("\n%s", df.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser(description="SMART-LLM module ablation")
    add_config_args(ap)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg.seed)
    run(cfg, device=args.device)


if __name__ == "__main__":
    main()
