"""Manuscript figures — each answers one scientific question, no decoration.

F1 architecture              (schematic)                      : what is SMART?
F2 router decision process   C_i vs calibrated RUS + boundary : how does it decide?
F3 reliability diagram       confidence vs accuracy + ECE     : is C_i calibrated?
F4 RBE prediction + residual B_pred vs B_true, residuals      : can benefit be predicted?
F5 routing-margin histogram  ΔC distribution + threshold      : how often / how decisively?
F6 noise robustness          acc + freq across conditions     : does it suppress bad retrieval?
F7 ablation                  agreement/acc per variant        : what does each module add?
F8 case study                per-case C_i / decision / correct: why does it succeed/fail?

Figures 2-8 are computed from results/master_<dataset>.csv (+ ablation / case-study
CSVs). Saved as PNG and PDF.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..config import Config
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


def _save(plt, cfg, name) -> str:
    base = Path(cfg.paths.figures_dir) / f"{name}_{cfg.data.dataset}"
    plt.savefig(str(base) + ".png", bbox_inches="tight")
    plt.savefig(str(base) + ".pdf", bbox_inches="tight")
    plt.close()
    return str(base) + ".png"


def _slice(df, pooling, condition, split="test"):
    return df[(df.pooling == pooling) & (df.condition == condition)
             & (df.split == split)]


# --------------------------------------------------------------------------- #
def figure1_architecture(cfg) -> str:
    plt = _mpl()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                     linewidth=1.2, edgecolor="#333", facecolor=color))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=12, color="#555", lw=1.2))

    box(0.2, 2.4, 1.6, 1.2, "Input x", "#eef3fb")
    box(2.2, 3.4, 2.2, 1.2, "Frozen LLM\n(parametric pass)\n-> h_L, C_i, Loss_p", "#dfe9f7")
    box(2.2, 1.2, 2.2, 1.0, "FAISS retrieval\n-> K, mu_K, sim", "#f7efdf")
    box(4.8, 3.6, 1.9, 1.0, "Confidence probe\nC_i = max softmax", "#e7f7df")
    box(4.8, 1.4, 1.9, 1.0, "RBE  (key)\nB_pred", "#d8f0cf")
    box(7.0, 2.4, 1.5, 1.2, "Arbiter\nΔC = cal(RUS) - C_i", "#f7dfe7")
    box(8.8, 3.4, 1.0, 1.0, "Trust\ninternal", "#eef3fb")
    box(8.8, 1.4, 1.0, 1.0, "Retrieve", "#f7efdf")

    arrow(1.8, 3.0, 2.2, 3.9); arrow(1.8, 2.9, 2.2, 1.8)
    arrow(4.4, 4.0, 4.8, 4.1); arrow(4.4, 1.7, 4.8, 1.9)
    arrow(6.7, 4.1, 7.5, 3.6); arrow(6.7, 1.9, 7.5, 2.6)
    arrow(8.5, 3.2, 8.8, 3.6); arrow(8.5, 2.6, 8.8, 2.0)
    ax.text(5.0, 5.6, "SMART-LLM: decision-time retrieval arbitration",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(7.75, 0.7, "retrieval executed only if ΔC>0 (no double inference)",
            ha="center", fontsize=8, style="italic", color="#666")
    return _save(plt, cfg, "figure1_architecture")


def figure2_router_decision(cfg, df) -> str:
    plt = _mpl()
    d = _slice(df, cfg.pooling.default, "clean")
    ci = d.C_i.to_numpy(); cr = d.calibrated_RUS.to_numpy()
    dec = d.smart_decision.to_numpy()
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.scatter(ci[dec == 0], cr[dec == 0], s=12, alpha=0.5, color="#48c",
               label="trust internal (ΔC≤0)")
    ax.scatter(ci[dec == 1], cr[dec == 1], s=12, alpha=0.5, color="#e07",
               label="retrieve (ΔC>0)")
    lim = [0, 1]
    ax.plot(lim, lim, "k--", lw=1, label="decision boundary ΔC=0")
    ax.set_xlabel("internal confidence  C_i")
    ax.set_ylabel("calibrated retrieval utility  cal(RUS)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("Router decision process")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(plt, cfg, "figure2_router_decision")


def figure3_reliability(cfg, df) -> str:
    plt = _mpl()
    d = _slice(df, cfg.pooling.default, "clean")
    y = d.label.to_numpy()
    correct = (d.probe_pred.to_numpy() == y).astype(float)
    conf = d.C_i.to_numpy()
    bc, ba, cnt = M.reliability_curve(conf, correct, n_bins=10)
    ece = M.expected_calibration_error(conf, correct)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(bc, ba, "o-", color="#48c", label="probe C_i")
    ax.bar(bc, ba, width=0.08, alpha=0.15, color="#48c")
    ax.set_xlabel("confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.legend(frameon=False, loc="upper left")
    return _save(plt, cfg, "figure3_reliability")


def figure4_rbe(cfg, df) -> str:
    plt = _mpl()
    d = _slice(df, cfg.pooling.default, "clean")
    bt, bp = d.B_true.to_numpy(), d.B_pred.to_numpy()
    r2, r = M.r2_score(bp, bt), M.pearson_r(bp, bt)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.4))
    lim = [min(bt.min(), bp.min()), max(bt.max(), bp.max())]
    a1.scatter(bt, bp, s=10, alpha=0.4, edgecolor="none")
    a1.plot(lim, lim, "k--", lw=1)
    a1.set_xlabel("B_true"); a1.set_ylabel("B_pred")
    a1.set_title(f"RBE prediction ($R^2$={r2:.3f}, r={r:.3f})")
    resid = bp - bt
    a2.scatter(bt, resid, s=10, alpha=0.4, edgecolor="none", color="#c44")
    a2.axhline(0, color="k", lw=1, ls="--")
    a2.set_xlabel("B_true"); a2.set_ylabel("residual (B_pred - B_true)")
    a2.set_title(f"Residuals (MAE={M.mae(bp, bt):.3f})")
    return _save(plt, cfg, "figure4_rbe")


def figure5_margin_hist(cfg, df) -> str:
    plt = _mpl()
    d = _slice(df, cfg.pooling.default, "clean")
    delta = d.delta_C.to_numpy()
    freq = float(np.mean(d.smart_decision.to_numpy()))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(delta[delta <= 0], bins=30, color="#48c", alpha=0.7,
            label="trust internal")
    ax.hist(delta[delta > 0], bins=30, color="#e07", alpha=0.7, label="retrieve")
    ax.axvline(0, color="k", lw=1.2, ls="--", label="threshold ΔC=0")
    ax.set_xlabel("routing margin  ΔC = cal(RUS) - C_i")
    ax.set_ylabel("count")
    ax.set_title(f"Routing-margin distribution (retrieval freq. = {freq:.2f})")
    ax.legend(frameon=False, fontsize=8)
    return _save(plt, cfg, "figure5_margin_hist")


def figure6_noise(cfg, df) -> str:
    plt = _mpl()
    pool = cfg.pooling.default
    conds = sorted(df.condition.unique())
    smart_acc, rag_acc, freq = [], [], []
    for c in conds:
        d = _slice(df, pool, c); y = d.label.to_numpy()
        smart_acc.append(M.accuracy(d.smart_pred.to_numpy(), y))
        rag_acc.append(M.accuracy(d.pred_r.to_numpy(), y))
        freq.append(M.retrieval_frequency(d.smart_decision.to_numpy()))
    x = np.arange(len(conds)); w = 0.35
    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.bar(x - w / 2, rag_acc, w, label="Always-RAG acc", color="#c44")
    ax1.bar(x + w / 2, smart_acc, w, label="SMART acc", color="#48c")
    ax1.set_xticks(x); ax1.set_xticklabels(conds)
    ax1.set_ylabel("Accuracy"); ax1.set_ylim(0, 1)
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(x, freq, "o-", color="#2a2", label="SMART retrieval freq.")
    ax2.set_ylabel("SMART retrieval frequency"); ax2.set_ylim(0, 1)
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, frameon=False, loc="lower left", fontsize=8)
    ax1.set_title("Retrieval robustness across conditions")
    return _save(plt, cfg, "figure6_noise")


def figure7_ablation(cfg) -> str:
    plt = _mpl()
    p = Path(cfg.paths.tables_dir) / f"ablation_{cfg.data.dataset}.csv"
    if not p.exists():
        return ""
    a = pd.read_csv(p)
    cols = [c for c in ("oracle_agreement", "accuracy") if c in a.columns]
    if not cols:
        return ""
    x = np.arange(len(a)); w = 0.8 / max(1, len(cols))
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(a)), 4.4))
    for i, c in enumerate(cols):
        ax.bar(x + i * w, a[c].to_numpy(), w, label=c.replace("_", " "))
    ax.set_xticks(x + w * (len(cols) - 1) / 2)
    ax.set_xticklabels(a["Variant"].tolist(), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title("Ablation: contribution of each routing component")
    ax.legend(frameon=False, fontsize=8)
    return _save(plt, cfg, "figure7_ablation")


def figure8_case_study(cfg) -> str:
    plt = _mpl()
    p = Path(cfg.paths.results_dir) / f"case_study_{cfg.data.dataset}.csv"
    if not p.exists():
        return ""
    c = pd.read_csv(p)
    y = np.arange(len(c))
    colors = ["#2a2" if v else "#c44" for v in c["correct"].to_numpy()]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(c) + 1.5))
    ax.barh(y, c["C_i"].to_numpy(), color=colors, alpha=0.8)
    for i, row in c.reset_index().iterrows():
        tag = "RET" if int(row["smart_decision"]) == 1 else "INT"
        ax.text(row["C_i"] + 0.01, i, f"{tag} | pred={int(row['smart_pred'])} "
                f"gold={int(row['label'])}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['category']}:{r['id']}" for _, r in c.iterrows()],
                       fontsize=7)
    ax.set_xlabel("confidence  C_i"); ax.set_xlim(0, 1.3)
    ax.set_title("Case study (green=correct, red=wrong; RET=retrieve, INT=internal)")
    return _save(plt, cfg, "figure8_case_study")


def build_all(cfg: Config, df: pd.DataFrame = None) -> List[str]:
    from .tables import load_master
    if df is None:
        df = load_master(cfg)
    figs = [
        figure1_architecture(cfg),
        figure2_router_decision(cfg, df),
        figure3_reliability(cfg, df),
        figure4_rbe(cfg, df),
        figure5_margin_hist(cfg, df),
        figure6_noise(cfg, df),
        figure7_ablation(cfg),
        figure8_case_study(cfg),
    ]
    return [f for f in figs if f]
