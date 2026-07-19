"""Build the manuscript tables from the master results CSV + metrics JSON.

This is a single-dataset *behavioural study* of decision-time retrieval
arbitration, so the tables emphasise depth over breadth. Every value is computed
from ``results/master_<dataset>.csv`` and ``cdka_metrics_<dataset>.json``.

Core tables (paper):
  T1 main performance            acc / macro-P/R/F1 / latency / retrieval freq
  T2 router vs oracle            agreement / precision / recall / F1 / regret
  T3 RBE prediction              R^2 / MAE / Pearson r
  T4 noise robustness            per-condition accuracy + retrieval freq + regret
  T5 calibration                 ECE / Brier for probe C_i vs LLM confidence
  T6 ablation                    loaded from the ablation driver
  T7 computation                 latency / retrieval reduction / rel. compute / tokens
Supporting tables:
  behavior (Analysis 4)          retrieval frequency / demos / prompt length / counts
  difficulty (Analysis 8)        easy/medium/hard tiers: conf / entropy / freq / acc
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..config import Config
from ..utils.io import atomic_write_csv, load_json
from . import metrics as M


# --------------------------------------------------------------------------- #
def load_master(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.paths.results_dir) / f"master_{cfg.data.dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run Stage 1 + Stage 2 on the GPU box first.")
    return pd.read_csv(path)


def load_metrics(cfg: Config) -> dict:
    path = Path(cfg.paths.results_dir) / f"cdka_metrics_{cfg.data.dataset}.json"
    return load_json(str(path)) if path.exists() else {}


def _slice(df, pooling, condition, split="test"):
    return df[(df.pooling == pooling) & (df.condition == condition)
              & (df.split == split)]


def _save(df: pd.DataFrame, cfg: Config, name: str) -> pd.DataFrame:
    atomic_write_csv(df, str(Path(cfg.paths.tables_dir) /
                            f"{name}_{cfg.data.dataset}.csv"))
    return df


def _overhead_s(mets: dict) -> float:
    """Per-sample retrieval-side overhead (query embedding + FAISS search), sec."""
    t = mets.get("timing", {})
    n = max(1, t.get("n_eval_for_amortization", 1))
    embed_q = t.get("embed_query_time", 0.0) / n
    search = t.get("search_time", {}).get("clean", 0.0) / n
    return embed_q + search


# --------------------------------------------------------------------------- #
def table1_main(cfg, df, mets) -> pd.DataFrame:
    d = _slice(df, cfg.pooling.default, "clean")
    y = d.label.to_numpy()
    oh = _overhead_s(mets)
    t_p, t_r, dec = d.t_p.to_numpy(), d.t_r.to_numpy(), d.smart_decision.to_numpy()

    def prf(pred):
        r = M.precision_recall_f1_macro(pred, y)
        return r["precision"], r["recall"], r["f1"]

    rows = []
    for name, pred, lat, freq in [
        ("No retrieval", d.pred_p.to_numpy(), np.mean(t_p), 0.0),
        ("Always RAG", d.pred_r.to_numpy(), np.mean(t_r) + oh, 1.0),
        ("SMART-LLM (ours)", d.smart_pred.to_numpy(),
         np.mean(t_p + dec * t_r) + oh, float(np.mean(dec))),
    ]:
        p, r, f = prf(pred)
        rows.append({"System": name, "Accuracy": M.accuracy(pred, y),
                     "Macro-F1": f, "Macro-P": p, "Macro-R": r,
                     "Latency (ms)": lat * 1e3, "Retrieval freq.": freq})
    return _save(pd.DataFrame(rows), cfg, "table1_main")


def table2_router_oracle(cfg, df) -> pd.DataFrame:
    rows = []
    for pool in cfg.pooling.types:
        d = _slice(df, pool, "clean")
        dec = d.smart_decision.to_numpy(); orc = d.oracle_decision.to_numpy()
        b = M.binary_prf(dec, orc)
        rows.append({"Pooling": pool,
                     "Agreement": M.routing_agreement(dec, orc),
                     "Precision": b["precision"], "Recall": b["recall"],
                     "F1": b["f1"],
                     "Mean regret": M.mean_regret(
                         dec, d.loss_without_retrieval.to_numpy(),
                         d.loss_with_retrieval.to_numpy())})
    return _save(pd.DataFrame(rows), cfg, "table2_router_oracle")


def table3_rbe(cfg, df) -> pd.DataFrame:
    rows = []
    for pool in cfg.pooling.types:
        d = _slice(df, pool, "clean")
        bp, bt = d.B_pred.to_numpy(), d.B_true.to_numpy()
        rows.append({"Pooling": pool, "R2": M.r2_score(bp, bt),
                     "MAE": M.mae(bp, bt), "Pearson r": M.pearson_r(bp, bt)})
    return _save(pd.DataFrame(rows), cfg, "table3_rbe")


def table4_noise(cfg, df) -> pd.DataFrame:
    pool = cfg.pooling.default
    ref = _slice(df, pool, "clean")
    no_acc = M.accuracy(ref.pred_p.to_numpy(), ref.label.to_numpy())
    rows = []
    for cond in sorted(df.condition.unique()):
        d = _slice(df, pool, cond); y = d.label.to_numpy()
        rows.append({"Condition": cond, "No-retrieval acc": no_acc,
                     "Always-RAG acc": M.accuracy(d.pred_r.to_numpy(), y),
                     "SMART acc": M.accuracy(d.smart_pred.to_numpy(), y),
                     "SMART retrieval freq.": M.retrieval_frequency(
                         d.smart_decision.to_numpy()),
                     "SMART mean regret": M.mean_regret(
                         d.smart_decision.to_numpy(),
                         d.loss_without_retrieval.to_numpy(),
                         d.loss_with_retrieval.to_numpy())})
    return _save(pd.DataFrame(rows), cfg, "table4_noise")


def table5_calibration(cfg, df) -> pd.DataFrame:
    d = _slice(df, cfg.pooling.default, "clean")
    y = d.label.to_numpy()
    rows = [
        {"Confidence signal": "Probe C_i (calibrated)",
         "ECE": M.expected_calibration_error(
             d.C_i.to_numpy(), (d.probe_pred.to_numpy() == y).astype(float)),
         "Brier": M.brier_score(
             d.C_i.to_numpy(), (d.probe_pred.to_numpy() == y).astype(float))},
        {"Confidence signal": "LLM verbalizer confidence",
         "ECE": M.expected_calibration_error(
             d.conf_llm.to_numpy(), (d.pred_p.to_numpy() == y).astype(float)),
         "Brier": M.brier_score(
             d.conf_llm.to_numpy(), (d.pred_p.to_numpy() == y).astype(float))},
    ]
    return _save(pd.DataFrame(rows), cfg, "table5_calibration")


def table6_ablation(cfg) -> pd.DataFrame:
    p = Path(cfg.paths.tables_dir) / f"ablation_{cfg.data.dataset}.csv"
    if p.exists():
        return _save(pd.read_csv(p), cfg, "table6_ablation")
    return _save(pd.DataFrame([{"Variant": "(run smart_llm.experiments.ablation)",
                                "note": "table6 produced by the ablation driver"}]),
                 cfg, "table6_ablation")


def table7_computation(cfg, df, mets) -> pd.DataFrame:
    d = _slice(df, cfg.pooling.default, "clean")
    oh = _overhead_s(mets)
    t_p, t_r = d.t_p.to_numpy(), d.t_r.to_numpy()
    ntok_p, ntok_r = d.n_tokens_p.to_numpy(), d.n_tokens_r.to_numpy()
    dec = d.smart_decision.to_numpy()
    lat_no = np.mean(t_p)
    lat_rag = np.mean(t_r) + oh
    lat_smart = np.mean(t_p + dec * t_r) + oh
    rows = [
        {"System": "No retrieval", "Latency (ms)": lat_no * 1e3,
         "Retrieval freq.": 0.0, "Retrieval reduction vs RAG": 1.0,
         "Rel. compute (vs RAG)": lat_no / lat_rag,
         "Avg prompt tokens": float(np.mean(ntok_p))},
        {"System": "Always RAG", "Latency (ms)": lat_rag * 1e3,
         "Retrieval freq.": 1.0, "Retrieval reduction vs RAG": 0.0,
         "Rel. compute (vs RAG)": 1.0, "Avg prompt tokens": float(np.mean(ntok_r))},
        {"System": "SMART-LLM (ours)", "Latency (ms)": lat_smart * 1e3,
         "Retrieval freq.": float(np.mean(dec)),
         "Retrieval reduction vs RAG": float(1.0 - np.mean(dec)),
         "Rel. compute (vs RAG)": lat_smart / lat_rag,
         "Avg prompt tokens": float(np.mean(np.where(dec == 1, ntok_r, ntok_p)))},
    ]
    return _save(pd.DataFrame(rows), cfg, "table7_computation")


def table_behavior(cfg, df) -> pd.DataFrame:
    d = _slice(df, cfg.pooling.default, "clean")
    dec = d.smart_decision.to_numpy()
    freq = float(np.mean(dec))
    rows = [{
        "Retrieval frequency": freq,
        "Avg retrieved examples / query": cfg.retrieval.k * freq,
        "Avg prompt tokens (no retrieval)": float(np.mean(d.n_tokens_p.to_numpy())),
        "Avg prompt tokens (retrieval)": float(np.mean(d.n_tokens_r.to_numpy())),
        "# retrieve": int(np.sum(dec == 1)),
        "# trust internal": int(np.sum(dec == 0)),
    }]
    return _save(pd.DataFrame(rows), cfg, "table_behavior")


def table_difficulty(cfg, df) -> pd.DataFrame:
    d = _slice(df, cfg.pooling.default, "clean")
    # difficulty = predictive entropy (higher = harder)
    tiers = M.difficulty_tertiles(d.entropy.to_numpy())
    names = {0: "Easy", 1: "Medium", 2: "Hard"}
    y = d.label.to_numpy()
    rows = []
    for t in [0, 1, 2]:
        m = tiers == t
        if not np.any(m):
            continue
        rows.append({
            "Tier": names[t], "n": int(m.sum()),
            "Mean C_i": float(d.C_i.to_numpy()[m].mean()),
            "Mean entropy": float(d.entropy.to_numpy()[m].mean()),
            "SMART retrieval freq.": float(d.smart_decision.to_numpy()[m].mean()),
            "Oracle retrieval freq.": float(d.oracle_decision.to_numpy()[m].mean()),
            "SMART accuracy": M.accuracy(d.smart_pred.to_numpy()[m], y[m]),
            "No-retrieval accuracy": M.accuracy(d.pred_p.to_numpy()[m], y[m]),
        })
    return _save(pd.DataFrame(rows), cfg, "table_difficulty")


def build_all(cfg: Config) -> Dict[str, pd.DataFrame]:
    df = load_master(cfg)
    mets = load_metrics(cfg)
    return {
        "table1_main": table1_main(cfg, df, mets),
        "table2_router_oracle": table2_router_oracle(cfg, df),
        "table3_rbe": table3_rbe(cfg, df),
        "table4_noise": table4_noise(cfg, df),
        "table5_calibration": table5_calibration(cfg, df),
        "table6_ablation": table6_ablation(cfg),
        "table7_computation": table7_computation(cfg, df, mets),
        "table_behavior": table_behavior(cfg, df),
        "table_difficulty": table_difficulty(cfg, df),
    }
