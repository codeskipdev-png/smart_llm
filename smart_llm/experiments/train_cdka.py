"""Stage 2 (light): train CDKA and produce the master results CSV.

For each pooling type it trains the confidence probe, the RBE, and the router
(fit on the *clean* condition, validation split), then evaluates the frozen
router across every retrieval condition. One CSV row per
(pooling × condition × sample) is written with the full logging schema, so all
paper figures/tables are reproducible from a single file.

This stage is cheap: it consumes cached features, so CDKA can be re-run/ablated
without touching the 7B backbone.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..analysis import metrics as M
from ..cdka.probe import ProbeData, fit_probe
from ..cdka.rbe import RBEData, fit_rbe
from ..cdka.router import Router
from ..config import add_config_args, config_from_args
from ..utils.io import atomic_write_csv, save_json
from ..utils.logging import get_logger
from ..utils.seed import seed_everything
from .cache import load_features

_log = get_logger("smart_llm.stage2")

MASTER_COLUMNS = [
    "id", "dataset", "label", "pooling", "condition", "split",
    "C_i", "entropy", "conf_llm", "entropy_llm",
    "sim", "B_pred", "B_true", "RUS", "calibrated_RUS", "delta_C",
    "smart_decision", "oracle_decision",
    "loss_without_retrieval", "loss_with_retrieval",
    "pred_p", "pred_r", "smart_pred", "regret",
    "t_p", "t_r",
]


def _select_device(name: str):
    if name != "auto":
        return name
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _split(n: int, seed: int, val_frac: float, test_frac: float
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    test = perm[:n_test]
    val = perm[n_test:n_test + n_val]
    train = perm[n_test + n_val:]
    return np.sort(train), np.sort(val), np.sort(test)


@dataclass
class PoolingArtifacts:
    probe: object
    rbe: object
    router: Router
    ci: np.ndarray          # [n] probe confidence C_i
    entropy: np.ndarray     # [n] normalised entropy (probe)
    pooled: np.ndarray      # [n, D] pooled hidden used as RBE h_L input


def _probe_inputs(feats, pooling: str):
    """Return (x, mask) arrays for the given pooling type."""
    if pooling == "last":
        return feats.h_last, None
    if pooling == "mean":
        return feats.h_mean, None
    if pooling == "attention":
        if feats.h_tokens is None:
            raise RuntimeError("attention pooling needs cached token window "
                               "(reload features with need_tokens=True).")
        return feats.h_tokens, feats.token_mask
    raise ValueError(f"Unknown pooling '{pooling}'")


def _train_pooling(feats, pooling, cfg, tr, va, device) -> PoolingArtifacts:
    x, mask = _probe_inputs(feats, pooling)
    labels = feats.labels
    dim = x.shape[-1]
    n_classes = len(feats.label_names)

    train_pd = ProbeData(x=x[tr], y=labels[tr],
                         mask=None if mask is None else mask[tr])
    val_pd = ProbeData(x=x[va], y=labels[va],
                       mask=None if mask is None else mask[va])
    probe, _ = fit_probe(train_pd, val_pd, dim, n_classes, pooling, cfg, device)

    pred = probe.predict(x, mask)
    ci, entropy = pred["confidence"], pred["entropy"]
    pooled = probe.pooled_features(x, mask)         # [n, D]

    clean = feats.conds["clean"]
    rbe_x = np.concatenate([pooled, clean["centroid"]], axis=1)
    rbe, _ = fit_rbe(RBEData(x=rbe_x[tr], b_true=clean["btrue"][tr]),
                     RBEData(x=rbe_x[va], b_true=clean["btrue"][va]), cfg, device)
    bpred_clean = rbe.predict(rbe_x)

    router = Router(cfg).fit(
        sim=clean["sim"][va], bpred=bpred_clean[va], ci=ci[va],
        oracle=clean["oracle"][va],
        loss_p=feats.meta["loss_p"].to_numpy()[va], loss_r=clean["loss_r"][va])
    return PoolingArtifacts(probe, rbe, router, ci, entropy, pooled)


def run(cfg, device: str = "auto") -> dict:
    device = _select_device(device)
    need_tokens = "attention" in cfg.pooling.types
    feats = load_features(cfg.paths.cache_dir, cfg.data.dataset,
                          need_tokens=need_tokens)
    n = feats.n
    labels = feats.labels
    loss_p = feats.meta["loss_p"].to_numpy(dtype=np.float64)
    conf_llm = feats.meta["conf_llm"].to_numpy()
    ent_llm = feats.meta["entropy_llm"].to_numpy()
    pred_p = feats.meta["pred_p"].to_numpy(dtype=np.int64)
    t_p = feats.meta["t_p"].to_numpy(dtype=np.float64)
    ids = feats.meta["id"].tolist()

    tr, va, te = _split(n, cfg.seed, cfg.data.val_fraction, cfg.data.val_fraction)
    split_of = np.empty(n, dtype=object)
    split_of[tr] = "train"; split_of[va] = "val"; split_of[te] = "test"
    n_classes = len(feats.label_names)

    _log.info("Stage-2 on %s: n=%d (train=%d val=%d test=%d) device=%s",
              cfg.data.dataset, n, len(tr), len(va), len(te), device)

    rows = []
    metrics_out = {"dataset": cfg.data.dataset, "n": n, "n_classes": n_classes,
                   "timing": feats.manifest.get("timing", {}), "pooling": {}}

    for pooling in cfg.pooling.types:
        art = _train_pooling(feats, pooling, cfg, tr, va, device)
        _log.info("[%s] router fit: %s", pooling, art.router.fit_report)

        for cond in feats.conditions:
            cd = feats.conds[cond]
            rbe_x = np.concatenate([art.pooled, cd["centroid"]], axis=1)
            bpred = art.rbe.predict(rbe_x)
            rout = art.router.predict(cd["sim"], bpred, art.ci)
            decision = rout["decision"]
            loss_r = cd["loss_r"].astype(np.float64)
            btrue = cd["btrue"]
            oracle = cd["oracle"]
            pred_r = cd["pred_r"].astype(np.int64)
            t_r = cd["t_r"].astype(np.float64)
            smart_pred = np.where(decision == 1, pred_r, pred_p)
            regret = M.regret_per_sample(decision, loss_p, loss_r)

            for i in range(n):
                rows.append({
                    "id": ids[i], "dataset": cfg.data.dataset,
                    "label": int(labels[i]), "pooling": pooling,
                    "condition": cond, "split": split_of[i],
                    "C_i": float(art.ci[i]), "entropy": float(art.entropy[i]),
                    "conf_llm": float(conf_llm[i]), "entropy_llm": float(ent_llm[i]),
                    "sim": float(cd["sim"][i]), "B_pred": float(bpred[i]),
                    "B_true": float(btrue[i]), "RUS": float(rout["rus"][i]),
                    "calibrated_RUS": float(rout["calibrated_rus"][i]),
                    "delta_C": float(rout["delta_c"][i]),
                    "smart_decision": int(decision[i]),
                    "oracle_decision": int(oracle[i]),
                    "loss_without_retrieval": float(loss_p[i]),
                    "loss_with_retrieval": float(loss_r[i]),
                    "pred_p": int(pred_p[i]), "pred_r": int(pred_r[i]),
                    "smart_pred": int(smart_pred[i]), "regret": float(regret[i]),
                    "t_p": float(t_p[i]), "t_r": float(t_r[i]),
                })

        # ---- per-pooling headline metrics on TEST / clean ----
        clean = feats.conds["clean"]
        rbe_x = np.concatenate([art.pooled, clean["centroid"]], axis=1)
        bpred_c = art.rbe.predict(rbe_x)
        rout_c = art.router.predict(clean["sim"], bpred_c, art.ci)
        metrics_out["pooling"][pooling] = {
            "rbe_r2_test": M.r2_score(bpred_c[te], clean["btrue"][te]),
            "rbe_pearson_test": M.pearson_r(bpred_c[te], clean["btrue"][te]),
            "routing_agreement_test": M.routing_agreement(
                rout_c["decision"][te], clean["oracle"][te]),
            "mean_regret_test": M.mean_regret(
                rout_c["decision"][te], loss_p[te], clean["loss_r"][te]),
            "router_fit": art.router.fit_report,
            "probe_ece_test": M.expected_calibration_error(
                art.ci[te], (pred_p[te] == labels[te]).astype(float)),
        }

    df = pd.DataFrame(rows, columns=MASTER_COLUMNS)
    out_csv = f"{cfg.paths.results_dir}/master_{cfg.data.dataset}.csv"
    atomic_write_csv(df, out_csv)
    out_json = f"{cfg.paths.results_dir}/cdka_metrics_{cfg.data.dataset}.json"
    save_json(metrics_out, out_json)
    _log.info("Wrote %s (%d rows) and %s", out_csv, len(df), out_json)
    return metrics_out


def main():
    ap = argparse.ArgumentParser(description="SMART-LLM Stage-2 CDKA training")
    add_config_args(ap)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg.seed)
    run(cfg, device=args.device)


if __name__ == "__main__":
    main()
