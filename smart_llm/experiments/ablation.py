"""Ablation (Analysis 7): isolate each decision-time module AND compare against
external decision-policy baselines at a matched retrieval budget.

All variants share the same frozen features, confidence probe, and RBE; only the
decision rule changes, so differences are attributable to it:

  Internal ablations
  * SMART (full)        RUS = alpha*sim + beta*B_pred, Platt-calibrated
  * - RBE               drop the predicted-benefit term (similarity-only routing)
  * - Calibration       replace Platt with an uncalibrated (min-max) map

  External decision-policy baselines (reviewer-requested)
  * Confidence-only     retrieve iff C_i < tau (no retrieval-utility signal)
  * Entropy-gated       retrieve iff predictive entropy > tau (Adaptive-RAG/FLARE-
                        style uncertainty gating), tau tuned on val for agreement
  * Random (budget-matched)  retrieve on a random subset sized to SMART's test
                        retrieval frequency — isolates 'decide well' from
                        'retrieve less'

  References
  * Always / Never RAG  trivial static policies
  * Oracle              retrieve iff Loss_r < Loss_p (upper bound)

Every row reports oracle agreement, precision/recall/F1 of the retrieve decision,
mean regret, end-task accuracy (with a 95% bootstrap CI half-width), and
retrieval frequency on the test split, plus paired significance vs. SMART (full):
a McNemar test for accuracy and a paired bootstrap for mean regret.
(UAAS is ablated separately in train_uaas.)
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

N_BOOT = 2000   # bootstrap resamples for CI half-widths (paper uses 10000)


def _per_sample(decision, loss_p, loss_r, pred_p, pred_r, label):
    """Per-sample correctness (0/1) and regret arrays for a decision."""
    smart_pred = np.where(decision == 1, pred_r, pred_p)
    correct = (smart_pred == np.asarray(label)).astype(np.float64)
    regret = M.regret_per_sample(decision, loss_p, loss_r)
    return correct, regret


def _eval(decision, oracle, loss_p, loss_r, pred_p, pred_r, label, seed=0):
    correct, regret = _per_sample(decision, loss_p, loss_r, pred_p, pred_r, label)
    b = M.binary_prf(decision, oracle)
    return {
        "oracle_agreement": M.routing_agreement(decision, oracle),
        "precision": b["precision"], "recall": b["recall"], "f1": b["f1"],
        "mean_regret": float(np.mean(regret)) if len(regret) else float("nan"),
        "accuracy": float(np.mean(correct)) if len(correct) else float("nan"),
        "accuracy_ci95": M.ci_halfwidth(correct, n_boot=N_BOOT, seed=seed),
        "regret_ci95": M.ci_halfwidth(regret, n_boot=N_BOOT, seed=seed),
        "retrieval_freq": M.retrieval_frequency(decision),
    }, correct, regret


def run(cfg, device="auto", seed=None, write=True) -> pd.DataFrame:
    """Single-seed ablation + external baselines. `seed` overrides cfg.seed so the
    multi-seed driver can re-draw the split and re-fit probe/RBE/calibrator on the
    same cached features (no LLM re-run). `write=False` suppresses the CSV so the
    aggregator can pool results itself."""
    device = _select_device(device)
    seed = cfg.seed if seed is None else int(seed)
    seed_everything(seed)                     # vary probe/RBE init per seed
    pool = cfg.pooling.default
    feats = load_features(cfg.paths.cache_dir, cfg.data.dataset,
                          need_tokens=(pool == "attention"))
    apply_stable_benefit(feats, cfg)          # correct B_true from stored losses
    n = feats.n
    labels = feats.labels
    tr, va, te = _split(n, seed, cfg.data.val_fraction, cfg.data.val_fraction)

    art = _train_pooling(feats, pool, cfg, tr, va, device)
    clean = feats.conds["clean"]
    loss_p = feats.meta["loss_p"].to_numpy(dtype=np.float64)
    loss_r = clean["loss_r"].astype(np.float64)
    pred_p = feats.meta["pred_p"].to_numpy(dtype=np.int64)
    pred_r = clean["pred_r"].astype(np.int64)
    oracle = clean["oracle"]
    ci = art.ci
    entropy = art.entropy
    bpred = art.rbe.predict(np.concatenate([art.pooled, clean["centroid"]], axis=1))

    def router_decision(fixed_alpha=None, calibration=None):
        saved = cfg.router.calibration
        if calibration is not None:
            cfg.router.calibration = calibration
        r = Router(cfg).fit(clean["sim"][va], bpred[va], ci[va], oracle[va],
                            loss_p[va], loss_r[va], fixed_alpha=fixed_alpha)
        cfg.router.calibration = saved
        return r.predict(clean["sim"], bpred, ci)["decision"]

    rows, per_correct, per_regret = [], {}, {}

    def add(name, decision):
        m, correct, regret = _eval(
            decision[te], oracle[te], loss_p[te], loss_r[te],
            pred_p[te], pred_r[te], labels[te], seed=seed)
        rows.append({"Variant": name, **m})
        per_correct[name] = correct
        per_regret[name] = regret

    # --- internal ablations ---
    add("SMART (full)", art.router.predict(clean["sim"], bpred, ci)["decision"])
    add("- RBE (similarity only)", router_decision(fixed_alpha=1.0))
    add("- Calibration (raw RUS)", router_decision(calibration="minmax"))

    # --- external decision-policy baselines ---
    # NB: variant names are STABLE across seeds (tuned thresholds are logged, not
    # embedded in the name) so the multi-seed aggregator can align rows by name.
    # confidence-only: retrieve iff C_i < tau (tau tuned on val for agreement)
    taus = np.quantile(ci[va], np.linspace(0.05, 0.95, 19))
    best_tau = max(taus, key=lambda t: M.routing_agreement(
        (ci[va] < t).astype(int), oracle[va]))
    _log.info("confidence-gate tau=%.3f  entropy-gate tuned below", best_tau)
    add("Confidence-only", (ci < best_tau).astype(int))

    # entropy-gated (Adaptive-RAG/FLARE-style): retrieve iff entropy > tau
    etaus = np.quantile(entropy[va], np.linspace(0.05, 0.95, 19))
    best_etau = max(etaus, key=lambda t: M.routing_agreement(
        (entropy[va] > t).astype(int), oracle[va]))
    _log.info("entropy-gate tau=%.3f", best_etau)
    add("Entropy-gated", (entropy > best_etau).astype(int))

    # random budget-matched: retrieve on a random subset sized to SMART's test freq
    smart_dec_te = art.router.predict(clean["sim"], bpred, ci)["decision"][te]
    target_k = int(round(float(np.mean(smart_dec_te)) * n))
    rng = np.random.default_rng(seed + 7)
    rand_dec = np.zeros(n, dtype=int)
    rand_dec[rng.permutation(n)[:target_k]] = 1
    add("Random (budget-matched)", rand_dec)

    # --- references ---
    add("Always RAG", np.ones(n, dtype=int))
    add("Never RAG", np.zeros(n, dtype=int))
    add("Oracle (upper bound)", oracle.astype(int))

    # --- paired significance vs SMART (full) ---
    ref_c, ref_r = per_correct["SMART (full)"], per_regret["SMART (full)"]
    for row in rows:
        name = row["Variant"]
        if name == "SMART (full)":
            row["acc_p_vs_full"], row["regret_p_vs_full"] = float("nan"), float("nan")
            row["acc_sig"], row["regret_sig"] = "(ref)", "(ref)"
            continue
        mc = M.mcnemar_test(ref_c.astype(int), per_correct[name].astype(int))
        pb = M.paired_bootstrap_diff(per_regret[name], ref_r, n_boot=N_BOOT,
                                     seed=seed)
        row["acc_p_vs_full"] = mc["p_value"]
        row["regret_p_vs_full"] = pb["p_value"]
        row["acc_sig"] = M.sig_marker(mc["p_value"])
        row["regret_sig"] = M.sig_marker(pb["p_value"])

    df = pd.DataFrame(rows)
    if write:
        atomic_write_csv(df, f"{cfg.paths.tables_dir}/ablation_{cfg.data.dataset}.csv")
        _log.info("\n%s", df.to_string(index=False))
    return df


_AGG_METRICS = ["oracle_agreement", "precision", "recall", "f1",
                "mean_regret", "accuracy", "retrieval_freq"]


def run_multiseed(cfg, seeds=(0, 1, 2, 3, 4), device="auto") -> pd.DataFrame:
    """Aggregate the ablation over several seeds. Only the light Stage-2 work is
    repeated (split + probe/RBE/calibrator re-fit on the SAME cached features), so
    no LLM re-run is needed. Reports the seed-mean of each metric and the across-
    seed std (±), which is the honest point/interval the manuscript's protocol
    calls for. Writes the aggregate to the ablation table so downstream rendering
    picks it up unchanged."""
    seeds = list(seeds)
    per_seed = [run(cfg, device=device, seed=s, write=False) for s in seeds]
    variants = list(per_seed[0]["Variant"])
    out = []
    for v in variants:
        rowset = [d[d["Variant"] == v].iloc[0] for d in per_seed if v in
                  set(d["Variant"])]
        agg = {"Variant": v, "n_seeds": len(rowset)}
        for m in _AGG_METRICS:
            vals = np.array([float(r[m]) for r in rowset
                             if r[m] == r[m]], dtype=np.float64)
            agg[m] = float(np.mean(vals)) if len(vals) else float("nan")
            agg[m + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(agg)
    df = pd.DataFrame(out)
    atomic_write_csv(df, f"{cfg.paths.tables_dir}/ablation_{cfg.data.dataset}.csv")
    atomic_write_csv(df,
                     f"{cfg.paths.tables_dir}/ablation_multiseed_{cfg.data.dataset}.csv")
    _log.info("multi-seed (%d seeds)\n%s", len(seeds), df.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser(description="SMART-LLM module ablation")
    add_config_args(ap)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", default="", help="comma-separated seeds for a "
                    "multi-seed aggregate (e.g. 0,1,2,3,4); empty = single seed")
    args = ap.parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg.seed)
    if args.seeds.strip():
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
        run_multiseed(cfg, seeds=seeds, device=args.device)
    else:
        run(cfg, device=args.device)


if __name__ == "__main__":
    main()
