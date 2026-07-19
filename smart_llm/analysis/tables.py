"""Build the five paper tables from the master results CSV + metrics JSON.

Every number comes from ``results/master_<dataset>.csv`` (produced by Stage 2 on
the GPU box). Nothing is hard-coded. Tables are written as CSV under
``tables_dir`` and returned as DataFrames for the DOCX generator.

Table 1  Main classification performance   (No-retrieval / Always-RAG / SMART)
Table 2  Router accuracy against the oracle (agreement, regret, alpha/beta, RBE R2)
Table 3  Retrieval-noise robustness        (per condition: acc + retrieval freq)
Table 4  Computation efficiency            (latency ms/sample + retrieval freq)
Table 5  Ablation study                    (pooling: RBE R2 / agreement / regret)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

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


# --------------------------------------------------------------------------- #
def table1_main(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    pool = cfg.pooling.default
    d = _slice(df, pool, "clean")
    y = d.label.to_numpy()
    rows = [
        {"System": "No retrieval",
         "Accuracy": M.accuracy(d.pred_p.to_numpy(), y),
         "Macro-F1": M.macro_f1(d.pred_p.to_numpy(), y),
         "Retrieval freq.": 0.0},
        {"System": "Always RAG",
         "Accuracy": M.accuracy(d.pred_r.to_numpy(), y),
         "Macro-F1": M.macro_f1(d.pred_r.to_numpy(), y),
         "Retrieval freq.": 1.0},
        {"System": "SMART-LLM (ours)",
         "Accuracy": M.accuracy(d.smart_pred.to_numpy(), y),
         "Macro-F1": M.macro_f1(d.smart_pred.to_numpy(), y),
         "Retrieval freq.": M.retrieval_frequency(d.smart_decision.to_numpy())},
    ]
    return _save(pd.DataFrame(rows), cfg, "table1_main")


def table2_router(cfg: Config, df: pd.DataFrame, mets: dict) -> pd.DataFrame:
    rows = []
    for pool in cfg.pooling.types:
        d = _slice(df, pool, "clean")
        pm = mets.get("pooling", {}).get(pool, {})
        rows.append({
            "Pooling": pool,
            "Oracle agreement": M.routing_agreement(
                d.smart_decision.to_numpy(), d.oracle_decision.to_numpy()),
            "Mean regret": M.mean_regret(
                d.smart_decision.to_numpy(),
                d.loss_without_retrieval.to_numpy(),
                d.loss_with_retrieval.to_numpy()),
            "RBE R2": pm.get("rbe_r2_test", float("nan")),
            "alpha": pm.get("router_fit", {}).get("alpha", float("nan")),
            "beta": pm.get("router_fit", {}).get("beta", float("nan")),
        })
    return _save(pd.DataFrame(rows), cfg, "table2_router")


def table3_robustness(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    pool = cfg.pooling.default
    rows = []
    # No-retrieval reference (condition-independent)
    ref = _slice(df, pool, "clean")
    no_acc = M.accuracy(ref.pred_p.to_numpy(), ref.label.to_numpy())
    for cond in sorted(df.condition.unique()):
        d = _slice(df, pool, cond)
        y = d.label.to_numpy()
        rows.append({
            "Condition": cond,
            "No-retrieval acc": no_acc,
            "Always-RAG acc": M.accuracy(d.pred_r.to_numpy(), y),
            "SMART acc": M.accuracy(d.smart_pred.to_numpy(), y),
            "SMART retrieval freq.": M.retrieval_frequency(d.smart_decision.to_numpy()),
            "SMART mean regret": M.mean_regret(
                d.smart_decision.to_numpy(),
                d.loss_without_retrieval.to_numpy(),
                d.loss_with_retrieval.to_numpy()),
        })
    return _save(pd.DataFrame(rows), cfg, "table3_robustness")


def table4_efficiency(cfg: Config, df: pd.DataFrame, mets: dict) -> pd.DataFrame:
    pool = cfg.pooling.default
    d = _slice(df, pool, "clean")
    timing = mets.get("timing", {})
    n_amort = max(1, timing.get("n_eval_for_amortization", 1))
    embed_q = timing.get("embed_query_time", 0.0) / n_amort
    search = timing.get("search_time", {}).get("clean", 0.0) / n_amort
    overhead = embed_q + search                      # retrieval-side cost / sample

    t_p = d.t_p.to_numpy()
    t_r = d.t_r.to_numpy()
    dec = d.smart_decision.to_numpy()

    def ms(x):
        return float(np.mean(x)) * 1e3

    rows = [
        {"System": "No retrieval",
         "Latency (ms/sample)": ms(t_p),
         "Retrieval freq.": 0.0},
        {"System": "Always RAG",
         "Latency (ms/sample)": ms(t_r) + overhead * 1e3,
         "Retrieval freq.": 1.0},
        {"System": "SMART-LLM (ours)",
         # always pays the parametric pass + retrieval-side overhead;
         # pays the second LLM pass only when it decides to retrieve.
         "Latency (ms/sample)": ms(t_p + dec * t_r) + overhead * 1e3,
         "Retrieval freq.": float(np.mean(dec))},
    ]
    return _save(pd.DataFrame(rows), cfg, "table4_efficiency")


def table5_ablation(cfg: Config, df: pd.DataFrame, mets: dict) -> pd.DataFrame:
    rows = []
    for pool in cfg.pooling.types:
        pm = mets.get("pooling", {}).get(pool, {})
        d = _slice(df, pool, "clean")
        rows.append({
            "Pooling": pool,
            "RBE R2": pm.get("rbe_r2_test", float("nan")),
            "RBE Pearson r": pm.get("rbe_pearson_test", float("nan")),
            "Oracle agreement": M.routing_agreement(
                d.smart_decision.to_numpy(), d.oracle_decision.to_numpy()),
            "Mean regret": pm.get("mean_regret_test", float("nan")),
            "Probe ECE": pm.get("probe_ece_test", float("nan")),
        })
    return _save(pd.DataFrame(rows), cfg, "table5_ablation")


def build_all(cfg: Config) -> Dict[str, pd.DataFrame]:
    df = load_master(cfg)
    mets = load_metrics(cfg)
    return {
        "table1_main": table1_main(cfg, df),
        "table2_router": table2_router(cfg, df, mets),
        "table3_robustness": table3_robustness(cfg, df),
        "table4_efficiency": table4_efficiency(cfg, df, mets),
        "table5_ablation": table5_ablation(cfg, df, mets),
    }
