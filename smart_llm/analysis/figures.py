"""Generate the paper figures from the master results CSV.

Figure 1  SMART-LLM architecture (schematic; drawn, not data)
Figure 2  RBE prediction quality        (B_true vs B_pred scatter + R2)
Figure 3  Retrieval robustness curve     (accuracy & retrieval-freq vs condition)
Figure 4  Accuracy-computation Pareto    (threshold sweep on ΔC + baselines)
Figure 5  Uncertainty vs retrieval freq. (binned U(x) -> mean decision)

All figures 2-5 are computed from ``results/master_<dataset>.csv`` — no invented
data. Saved as both PNG and PDF into ``figures_dir``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..config import Config
from ..cdka.router import uncertainty
from . import metrics as M


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "figure.autolayout": True,
    })
    return plt


def _save(plt, cfg: Config, name: str) -> str:
    base = Path(cfg.paths.figures_dir) / f"{name}_{cfg.data.dataset}"
    plt.savefig(str(base) + ".png", bbox_inches="tight")
    plt.savefig(str(base) + ".pdf", bbox_inches="tight")
    plt.close()
    return str(base) + ".png"


def _slice(df, pooling, condition, split="test"):
    return df[(df.pooling == pooling) & (df.condition == condition)
             & (df.split == split)]


# --------------------------------------------------------------------------- #
def figure1_architecture(cfg: Config) -> str:
    """Self-contained schematic of the SMART-LLM inference path."""
    plt = _mpl()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                     linewidth=1.2, edgecolor="#333", facecolor=color))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=12, color="#555", lw=1.2))

    box(0.2, 2.4, 1.6, 1.2, "Input x", "#eef3fb")
    box(2.2, 3.4, 2.2, 1.2, "Frozen LLM\n(no-retrieval pass)\n-> h_L, Loss_p", "#dfe9f7")
    box(2.2, 1.2, 2.2, 1.0, "FAISS retrieval\n-> K, mu_K, sim", "#f7efdf")
    box(4.8, 3.6, 1.9, 1.0, "Confidence probe\nC_i = max softmax", "#e7f7df")
    box(4.8, 1.4, 1.9, 1.0, "RBE\nB_pred", "#e7f7df")
    box(7.0, 2.4, 1.4, 1.2, "Router\nΔC = cal(RUS) - C_i", "#f7dfe7")
    box(8.7, 3.4, 1.1, 1.0, "Trust\ninternal", "#eef3fb")
    box(8.7, 1.4, 1.1, 1.0, "Use\nretrieval", "#f7efdf")

    arrow(1.8, 3.0, 2.2, 3.9); arrow(1.8, 2.9, 2.2, 1.8)
    arrow(4.4, 4.0, 4.8, 4.1); arrow(4.4, 1.7, 4.8, 1.9)
    arrow(6.7, 4.1, 7.4, 3.6); arrow(6.7, 1.9, 7.4, 2.6)
    arrow(8.4, 3.2, 8.7, 3.6); arrow(8.4, 2.6, 8.7, 2.0)
    ax.text(5.0, 5.6, "SMART-LLM: Confidence-Driven Knowledge Arbitration",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(7.7, 0.7, "no double inference: retrieval pass runs only if ΔC>0",
            ha="center", fontsize=8, style="italic", color="#666")
    return _save(plt, cfg, "figure1_architecture")


def figure2_rbe(cfg: Config, df: pd.DataFrame) -> str:
    plt = _mpl()
    d = _slice(df, cfg.pooling.default, "clean")
    bt, bp = d.B_true.to_numpy(), d.B_pred.to_numpy()
    r2 = M.r2_score(bp, bt); r = M.pearson_r(bp, bt)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(bt, bp, s=10, alpha=0.4, edgecolor="none")
    lim = [min(bt.min(), bp.min()), max(bt.max(), bp.max())]
    ax.plot(lim, lim, "k--", lw=1, label="ideal")
    ax.set_xlabel("B_true  (ground-truth retrieval benefit)")
    ax.set_ylabel("B_pred  (RBE estimate)")
    ax.set_title(f"RBE prediction quality\n$R^2$={r2:.3f}, r={r:.3f}")
    ax.legend(frameon=False)
    return _save(plt, cfg, "figure2_rbe")


def figure3_robustness(cfg: Config, df: pd.DataFrame) -> str:
    plt = _mpl()
    pool = cfg.pooling.default
    conds = sorted(df.condition.unique())
    smart_acc, rag_acc, freq = [], [], []
    for c in conds:
        d = _slice(df, pool, c); y = d.label.to_numpy()
        smart_acc.append(M.accuracy(d.smart_pred.to_numpy(), y))
        rag_acc.append(M.accuracy(d.pred_r.to_numpy(), y))
        freq.append(M.retrieval_frequency(d.smart_decision.to_numpy()))
    x = np.arange(len(conds))
    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    w = 0.35
    ax1.bar(x - w / 2, rag_acc, w, label="Always-RAG acc", color="#c44")
    ax1.bar(x + w / 2, smart_acc, w, label="SMART acc", color="#48c")
    ax1.set_xticks(x); ax1.set_xticklabels(conds)
    ax1.set_ylabel("Accuracy"); ax1.set_ylim(0, 1)
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(x, freq, "o-", color="#2a2", label="SMART retrieval freq.")
    ax2.set_ylabel("SMART retrieval frequency"); ax2.set_ylim(0, 1)
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, frameon=False, loc="lower left", fontsize=8)
    ax1.set_title("Retrieval robustness across conditions")
    return _save(plt, cfg, "figure3_robustness")


def figure4_pareto(cfg: Config, df: pd.DataFrame) -> str:
    """Accuracy-vs-latency frontier by sweeping the routing threshold on ΔC."""
    plt = _mpl()
    d = _slice(df, cfg.pooling.default, "clean")
    y = d.label.to_numpy()
    delta = d.delta_C.to_numpy()
    pred_p, pred_r = d.pred_p.to_numpy(), d.pred_r.to_numpy()
    t_p, t_r = d.t_p.to_numpy(), d.t_r.to_numpy()

    taus = np.quantile(delta, np.linspace(0, 1, 21))
    accs, lats, freqs = [], [], []
    for tau in taus:
        dec = (delta > tau).astype(int)
        pred = np.where(dec == 1, pred_r, pred_p)
        accs.append(M.accuracy(pred, y))
        lats.append(float(np.mean(t_p + dec * t_r)) * 1e3)
        freqs.append(float(np.mean(dec)))

    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.plot(lats, accs, "-", color="#48c", alpha=0.6, label="SMART frontier (ΔC sweep)")
    sc = ax.scatter(lats, accs, c=freqs, cmap="viridis", s=28, zorder=3)
    fig.colorbar(sc, ax=ax, label="retrieval frequency")
    # baselines
    ax.scatter([np.mean(t_p) * 1e3], [M.accuracy(pred_p, y)], marker="s",
               s=70, color="#888", label="No retrieval", zorder=4)
    ax.scatter([np.mean(t_r) * 1e3], [M.accuracy(pred_r, y)], marker="^",
               s=70, color="#c44", label="Always RAG", zorder=4)
    ax.set_xlabel("Latency (ms/sample, LLM passes)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy-computation Pareto frontier")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(plt, cfg, "figure4_pareto")


def figure5_uncertainty(cfg: Config, df: pd.DataFrame, n_bins: int = 8) -> str:
    plt = _mpl()
    d = _slice(df, cfg.pooling.default, "clean")
    u = uncertainty(d.entropy.to_numpy(), d.C_i.to_numpy(),
                    cfg.router.lam_uncertainty)
    dec = d.smart_decision.to_numpy()
    orc = d.oracle_decision.to_numpy()
    bins = np.linspace(u.min(), u.max() + 1e-9, n_bins + 1)
    centers, sm_freq, or_freq = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (u >= lo) & (u < hi)
        if not np.any(m):
            continue
        centers.append((lo + hi) / 2)
        sm_freq.append(float(dec[m].mean()))
        or_freq.append(float(orc[m].mean()))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(centers, sm_freq, "o-", color="#48c", label="SMART retrieval freq.")
    ax.plot(centers, or_freq, "s--", color="#2a2", label="Oracle retrieval freq.")
    ax.set_xlabel("Uncertainty  U(x)")
    ax.set_ylabel("Retrieval frequency")
    ax.set_ylim(0, 1)
    ax.set_title("Uncertainty vs retrieval frequency")
    ax.legend(frameon=False)
    return _save(plt, cfg, "figure5_uncertainty")


def build_all(cfg: Config, df: pd.DataFrame = None) -> List[str]:
    from .tables import load_master
    if df is None:
        df = load_master(cfg)
    return [
        figure1_architecture(cfg),
        figure2_rbe(cfg, df),
        figure3_robustness(cfg, df),
        figure4_pareto(cfg, df),
        figure5_uncertainty(cfg, df),
    ]
