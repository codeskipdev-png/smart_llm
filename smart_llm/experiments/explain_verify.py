"""Contribution 3 experiment: attribution-guided explanation verification.

For a subset of eval samples, compute IG attributions, generate an explanation,
and score how well the explanation covers the top-attribution tokens. Reports the
distribution of faithfulness scores (mean, median, fraction below a threshold =
"unfaithful" explanations).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..config import add_config_args, config_from_args
from ..data.datasets import load_corpus
from ..explain.attribution import AttributionExplainer, faithfulness_score
from ..llm.prompts import VerbalizerSpec
from ..utils.io import atomic_write_csv
from ..utils.logging import get_logger
from ..utils.seed import seed_everything

_log = get_logger("smart_llm.explain_exp")


def run(cfg, n_samples: int = None) -> pd.DataFrame:
    corpus = load_corpus(cfg)
    verbalizer = VerbalizerSpec(corpus.label_names)
    n = min(n_samples or cfg.explain.n_samples, len(corpus.eval_texts))
    expl = AttributionExplainer(cfg)

    rows = []
    try:
        from tqdm import tqdm
        it = tqdm(range(n), desc="explain")
    except ImportError:
        it = range(n)

    for i in it:
        text = corpus.eval_texts[i]
        label = int(corpus.eval_labels[i])
        attr = expl.attribute(text, verbalizer)
        explanation = expl.generate_explanation(text, verbalizer, attr.pred_class)
        score = faithfulness_score(attr.top_tokens, explanation)
        rows.append({
            "id": corpus.eval_ids[i], "label": label, "pred": attr.pred_class,
            "correct": int(attr.pred_class == label),
            "top_tokens": "|".join(attr.top_tokens),
            "faithfulness": score,
            "explanation": explanation.replace("\n", " ")[:500],
        })

    df = pd.DataFrame(rows)
    atomic_write_csv(df, f"{cfg.paths.results_dir}/explain_{cfg.data.dataset}.csv")

    valid = df["faithfulness"].dropna().to_numpy()
    summary = pd.DataFrame([{
        "n": len(valid),
        "mean_faithfulness": float(np.mean(valid)) if len(valid) else float("nan"),
        "median_faithfulness": float(np.median(valid)) if len(valid) else float("nan"),
        "frac_unfaithful(<0.2)": float(np.mean(valid < 0.2)) if len(valid) else float("nan"),
        "mean_faithfulness_correct": float(
            df.loc[df.correct == 1, "faithfulness"].dropna().mean()),
        "mean_faithfulness_wrong": float(
            df.loc[df.correct == 0, "faithfulness"].dropna().mean()),
    }])
    atomic_write_csv(summary, f"{cfg.paths.tables_dir}/table_explain_{cfg.data.dataset}.csv")
    _log.info("\n%s", summary.to_string(index=False))
    return summary


def main():
    ap = argparse.ArgumentParser(description="SMART-LLM explanation verification")
    add_config_args(ap)
    ap.add_argument("--n-samples", type=int, default=None)
    args = ap.parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg.seed)
    run(cfg, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
