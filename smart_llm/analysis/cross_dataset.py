"""Cross-dataset comparison (generalization evidence for the core contribution).

The central hypothesis is that a *learned* Retrieval Benefit Estimator earns its
place over similarity-only routing where semantic similarity is a weaker proxy for
retrieval usefulness. Topic classification (20 Newsgroups) is strongly
similarity-shaped; sentiment (Financial PhraseBank) is much less so. This module
puts the two datasets side by side on the metrics that decide contribution #1:

  * RBE R^2 / Pearson                        (is benefit predictable?)
  * Agreement(full) - Agreement(-RBE)        (does B_pred help routing vs similarity?)
  * Regret(-RBE)   - Regret(full)            (does B_pred lower regret?)
  * adversarial robustness (Always-RAG vs SMART)

It reads each dataset's already-generated table CSVs (run make_all per dataset
first) and writes one comparison table + one figure. Nothing is recomputed from
the LLM; nothing is invented.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import add_config_args, config_from_args
from ..utils.io import atomic_write_csv
from ..utils.logging import get_logger

_log = get_logger("smart_llm.cross")


def _read(cfg, dataset, table, row_col, row_val, col) -> Optional[float]:
    p = Path(cfg.paths.tables_dir) / f"{table}_{dataset}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[df[row_col] == row_val]
    if sub.empty or col not in df.columns:
        return None
    try:
        return float(sub.iloc[0][col])
    except (TypeError, ValueError):
        return None


def _row(cfg, ds, pool) -> dict:
    ag_full = _read(cfg, ds, "ablation", "Variant", "SMART (full)", "oracle_agreement")
    ag_sim = _read(cfg, ds, "ablation", "Variant", "- RBE (similarity only)", "oracle_agreement")
    rg_full = _read(cfg, ds, "ablation", "Variant", "SMART (full)", "mean_regret")
    rg_sim = _read(cfg, ds, "ablation", "Variant", "- RBE (similarity only)", "mean_regret")
    d_ag = (ag_full - ag_sim) if None not in (ag_full, ag_sim) else None
    d_rg = (rg_sim - rg_full) if None not in (rg_full, rg_sim) else None
    return {
        "Dataset": ds,
        "RBE R2": _read(cfg, ds, "table3_rbe", "Pooling", pool, "R2"),
        "RBE Pearson r": _read(cfg, ds, "table3_rbe", "Pooling", pool, "Pearson r"),
        "Agreement (full)": ag_full,
        "Agreement (-RBE)": ag_sim,
        "Δ Agreement (RBE gain)": d_ag,
        "Regret (full)": rg_full,
        "Regret (-RBE)": rg_sim,
        "Δ Regret (RBE gain)": d_rg,
        "SMART acc": _read(cfg, ds, "table1_main", "System", "SMART-LLM (ours)", "Accuracy"),
        "Always-RAG acc": _read(cfg, ds, "table1_main", "System", "Always RAG", "Accuracy"),
        "No-retr acc": _read(cfg, ds, "table1_main", "System", "No retrieval", "Accuracy"),
        "Retrieval freq": _read(cfg, ds, "table1_main", "System", "SMART-LLM (ours)", "Retrieval freq."),
        "Adv Always-RAG acc": _read(cfg, ds, "table4_noise", "Condition", "adversarial", "Always-RAG acc"),
        "Adv SMART acc": _read(cfg, ds, "table4_noise", "Condition", "adversarial", "SMART acc"),
    }


def _figure(cfg, comp: pd.DataFrame) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3})
    ds = comp["Dataset"].tolist()
    x = np.arange(len(ds)); w = 0.35
    r2 = comp["RBE R2"].fillna(0).to_numpy()
    dag = comp["Δ Agreement (RBE gain)"].fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(1.9 * len(ds) + 3, 4.4))
    ax.bar(x - w / 2, r2, w, label="RBE $R^2$", color="#48c")
    ax.bar(x + w / 2, dag, w, label="Δ agreement (full − similarity-only)", color="#e07")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(ds, fontsize=9)
    ax.set_ylabel("value")
    ax.set_title("Does the learned RBE earn its place? (higher = more RBE value)")
    ax.legend(frameon=False, fontsize=9)
    out = Path(cfg.paths.figures_dir) / "figure9_cross_dataset"
    plt.savefig(str(out) + ".png", bbox_inches="tight")
    plt.savefig(str(out) + ".pdf", bbox_inches="tight")
    plt.close()
    return str(out) + ".png"


def run(cfg, datasets: List[str]) -> pd.DataFrame:
    pool = cfg.pooling.default
    comp = pd.DataFrame([_row(cfg, ds, pool) for ds in datasets])
    atomic_write_csv(comp, str(Path(cfg.paths.tables_dir) / "cross_dataset_comparison.csv"))
    fig = _figure(cfg, comp)
    _log.info("Cross-dataset comparison:\n%s", comp.to_string(index=False))
    _log.info("Figure -> %s", fig)
    return comp


def main():
    ap = argparse.ArgumentParser(description="SMART-LLM cross-dataset comparison")
    add_config_args(ap)
    ap.add_argument("--datasets", type=str, required=True,
                    help="comma-separated dataset names (must be already analysed)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    run(cfg, [d.strip() for d in args.datasets.split(",")])


if __name__ == "__main__":
    main()
