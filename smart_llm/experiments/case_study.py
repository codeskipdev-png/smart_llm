"""Analysis 9 — qualitative case study.

Selects successful and failure cases from the master log and, for each, records:
input, retrieved documents (with their displayed labels), confidence C_i,
entropy, routing decision (+ oracle), predictions, benefit estimate, a generated
explanation, and an attribution-faithfulness score. Emits both a machine CSV
(for Figure 8) and a human-readable Markdown report.

The selection is deterministic and covers the scientifically interesting regimes:
retrieval-helped, internal-sufficed, retrieval-hurt, and missed-retrieval.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import add_config_args, config_from_args
from ..data.datasets import load_corpus
from ..explain.attribution import AttributionExplainer, faithfulness_score
from ..llm.prompts import VerbalizerSpec
from ..utils.io import atomic_write_csv
from ..utils.logging import get_logger
from ..utils.seed import seed_everything
from .cache import load_features

_log = get_logger("smart_llm.case_study")


def _pick(d: pd.DataFrame, n_each: int) -> pd.DataFrame:
    """Deterministic selection covering four decision regimes."""
    d = d.copy()
    d["correct"] = (d.smart_pred == d.label).astype(int)
    d = d.sort_values("id")
    succ = d[d.correct == 1]
    fail = d[d.correct == 0]
    # prefer, among successes: retrieval helped then internal sufficed
    helped = succ[(succ.smart_decision == 1) & (succ.pred_p != succ.label)]
    internal = succ[(succ.smart_decision == 0)]
    succ_sel = pd.concat([helped, internal, succ]).drop_duplicates("id").head(n_each)
    # prefer, among failures: retrieval hurt then missed retrieval
    hurt = fail[(fail.smart_decision == 1) & (fail.pred_p == fail.label)]
    missed = fail[(fail.smart_decision == 0) & (fail.oracle_decision == 1)]
    fail_sel = pd.concat([hurt, missed, fail]).drop_duplicates("id").head(n_each)
    succ_sel = succ_sel.assign(category="success")
    fail_sel = fail_sel.assign(category="failure")
    return pd.concat([succ_sel, fail_sel], ignore_index=True)


def run(cfg, n_each: int = 5) -> pd.DataFrame:
    corpus = load_corpus(cfg)
    verbalizer = VerbalizerSpec(corpus.label_names)
    feats = load_features(cfg.paths.cache_dir, cfg.data.dataset, conditions=["clean"])
    id_to_idx = {sid: i for i, sid in enumerate(feats.meta["id"].tolist())}
    retr_idx = feats.conds["clean"]["retr_idx"]
    demo_lab = feats.conds["clean"]["demo_label_ids"]

    master = pd.read_csv(f"{cfg.paths.results_dir}/master_{cfg.data.dataset}.csv")
    d = master[(master.pooling == cfg.pooling.default)
               & (master.condition == "clean") & (master.split == "test")]
    picks = _pick(d, n_each)

    explainer = AttributionExplainer(cfg)
    rows, md = [], ["# SMART-LLM case study\n"]
    for _, r in picks.iterrows():
        idx = id_to_idx[r["id"]]
        text = corpus.eval_texts[idx]
        pred_class = int(r["smart_pred"])
        attr = explainer.attribute(text, verbalizer)
        expl = explainer.generate_explanation(text, verbalizer, pred_class)
        faith = faithfulness_score(attr.top_tokens, expl)

        docs = [(corpus.pool_texts[int(j)][:180], corpus.label_names[int(l)])
                for j, l in zip(retr_idx[idx][:cfg.retrieval.k],
                                demo_lab[idx][:cfg.retrieval.k])]
        rows.append({
            "id": r["id"], "category": r["category"],
            "label": int(r["label"]), "smart_pred": pred_class,
            "pred_p": int(r["pred_p"]), "pred_r": int(r["pred_r"]),
            "correct": int(pred_class == int(r["label"])),
            "C_i": float(r["C_i"]), "entropy": float(r["entropy"]),
            "smart_decision": int(r["smart_decision"]),
            "oracle_decision": int(r["oracle_decision"]),
            "B_pred": float(r["B_pred"]), "B_true": float(r["B_true"]),
            "delta_C": float(r["delta_C"]),
            "top_tokens": "|".join(attr.top_tokens),
            "faithfulness": faith,
        })
        md.append(_render_md(r, corpus, text, docs, expl, attr.top_tokens, faith))
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    df = pd.DataFrame(rows)
    atomic_write_csv(df, f"{cfg.paths.results_dir}/case_study_{cfg.data.dataset}.csv")
    Path(f"{cfg.paths.results_dir}/case_study_{cfg.data.dataset}.md").write_text(
        "\n".join(md), encoding="utf-8")
    _log.info("Wrote %d case studies (%d success / %d failure)",
              len(df), int((df.category == "success").sum()),
              int((df.category == "failure").sum()))
    return df


def _render_md(r, corpus, text, docs, expl, tokens, faith) -> str:
    dec = "RETRIEVE" if int(r["smart_decision"]) == 1 else "TRUST INTERNAL"
    orc = "retrieve" if int(r["oracle_decision"]) == 1 else "internal"
    ln = corpus.label_names
    out = [f"\n## [{r['category']}] {r['id']}",
           f"- **Input:** {text[:400]}...",
           f"- **Gold:** {ln[int(r['label'])]} | **SMART pred:** {ln[int(r['smart_pred'])]} "
           f"(parametric={ln[int(r['pred_p'])]}, retrieval={ln[int(r['pred_r'])]})",
           f"- **C_i:** {r['C_i']:.3f} | entropy {r['entropy']:.3f} | "
           f"B_pred {r['B_pred']:.3f} (B_true {r['B_true']:.3f}) | ΔC {r['delta_C']:.3f}",
           f"- **Decision:** {dec}  (oracle: {orc})",
           f"- **Top attribution tokens:** {', '.join(tokens)}",
           f"- **Explanation:** {expl.strip()[:400]}",
           f"- **Faithfulness:** {faith:.2f}",
           "- **Retrieved (top-3):**"]
    for i, (t, lab) in enumerate(docs[:3], 1):
        out.append(f"    {i}. ({lab}) {t}...")
    # short automatic diagnosis
    if int(r["smart_pred"]) == int(r["label"]):
        why = ("retrieval supplied the correct signal" if int(r["smart_decision"]) == 1
               else "parametric knowledge was sufficient and retrieval was withheld")
    else:
        why = ("retrieval introduced misleading context" if int(r["smart_decision"]) == 1
               else "beneficial retrieval was missed (over-confidence)")
    out.append(f"- **Why:** {why}.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="SMART-LLM qualitative case study")
    add_config_args(ap)
    ap.add_argument("--n-each", type=int, default=5)
    args = ap.parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg.seed)
    run(cfg, n_each=args.n_each)


if __name__ == "__main__":
    main()
