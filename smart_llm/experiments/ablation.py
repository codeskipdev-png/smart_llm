"""Router-signal ablation (supplementary Table 6).

Isolates the contribution of each routing signal, using the cached features and
the default pooling. All variants share the same probe + RBE; only the routing
rule changes:

  * SMART (full)        : RUS = alpha*sim + beta*B_pred  (tuned)
  * B_pred only         : alpha = 0
  * sim only            : beta  = 0
  * Confidence-only     : retrieve iff C_i < tau  (tau tuned on val)
  * Always / Never RAG  : trivial references
  * Oracle              : retrieve iff Loss_r < Loss_p  (upper bound)

Reports oracle agreement, mean regret, and end-task accuracy on the test split.
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
from .train_cdka import _select_device, _split, _train_pooling

_log = get_logger("smart_llm.ablation")


def _eval_decision(decision, oracle, loss_p, loss_r, pred_p, pred_r, label):
    smart_pred = np.where(decision == 1, pred_r, pred_p)
    return {
        "oracle_agreement": M.routing_agreement(decision, oracle),
        "mean_regret": M.mean_regret(decision, loss_p, loss_r),
        "accuracy": M.accuracy(smart_pred, label),
        "retrieval_freq": M.retrieval_frequency(decision),
    }


def run(cfg, device="auto") -> pd.DataFrame:
    device = _select_device(device)
    pool = cfg.pooling.default
    feats = load_features(cfg.paths.cache_dir, cfg.data.dataset,
                          need_tokens=(pool == "attention"))
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

    def router_variant(fixed_alpha):
        r = Router(cfg).fit(clean["sim"][va], bpred[va], ci[va], oracle[va],
                            loss_p[va], loss_r[va], fixed_alpha=fixed_alpha)
        return r.predict(clean["sim"], bpred, ci)["decision"]

    rows = []

    def add(name, decision):
        m = _eval_decision(decision[te], oracle[te], loss_p[te], loss_r[te],
                           pred_p[te], pred_r[te], labels[te])
        rows.append({"Variant": name, **m})

    add("SMART (full)", art.router.predict(clean["sim"], bpred, ci)["decision"])
    add("B_pred only (alpha=0)", router_variant(0.0))
    add("sim only (beta=0)", router_variant(1.0))

    # confidence-only: retrieve iff C_i < tau, tau tuned on val for agreement
    taus = np.quantile(ci[va], np.linspace(0.05, 0.95, 19))
    best_tau, best_agree = taus[0], -1
    for tau in taus:
        agree = M.routing_agreement((ci[va] < tau).astype(int), oracle[va])
        if agree > best_agree:
            best_agree, best_tau = agree, tau
    add(f"Confidence-only (C_i<{best_tau:.2f})", (ci < best_tau).astype(int))

    add("Always RAG", np.ones(n, dtype=int))
    add("Never RAG", np.zeros(n, dtype=int))
    add("Oracle (upper bound)", oracle.astype(int))

    df = pd.DataFrame(rows)
    out = f"{cfg.paths.tables_dir}/table6_rus_ablation_{cfg.data.dataset}.csv"
    atomic_write_csv(df, out)
    _log.info("Wrote router ablation -> %s", out)
    return df


def main():
    ap = argparse.ArgumentParser(description="SMART-LLM router-signal ablation")
    add_config_args(ap)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg.seed)
    print(run(cfg, device=args.device).to_string(index=False))


if __name__ == "__main__":
    main()
