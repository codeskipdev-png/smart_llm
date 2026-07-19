"""Contribution 2 experiment: Uncertainty-Aware Adapter Scaling.

Trains LoRA adapters at several ranks on the frozen backbone, then compares:

  * static LoRA r in {4, 16, 32}            (fixed capacity for every input)
  * SMART-UAAS  r*(x) = bucket(uncertainty)  (per-input capacity)

Uncertainty U(x) reuses the CDKA signals cached in Stage 1
(U = lam*entropy + (1-lam)*(1-C_i), with C_i the frozen-LLM verbalizer confidence
by default). Requires the Stage-1 cache (Phase 1 precedes Phase 2).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..analysis import metrics as M
from ..config import add_config_args, config_from_args
from ..data.datasets import load_corpus
from ..llm.prompts import VerbalizerSpec
from ..uaas.lora import UAASLoRA
from ..uaas.scaling import continuous_rank, bucket_rank
from ..utils.io import atomic_write_csv
from ..utils.logging import get_logger
from ..utils.seed import seed_everything
from .cache import load_features

_log = get_logger("smart_llm.uaas_exp")


def _trainable_params(model, name: str) -> int:
    return int(sum(p.numel() for n, p in model.named_parameters()
                   if f".{name}." in n and "lora_" in n))


def run(cfg, train_samples: int = 1000) -> pd.DataFrame:
    corpus = load_corpus(cfg)
    verbalizer = VerbalizerSpec(corpus.label_names)

    # uncertainty from the Stage-1 cache (aligned to eval order)
    feats = load_features(cfg.paths.cache_dir, cfg.data.dataset,
                          conditions=["clean"])
    assert list(feats.meta["id"]) == list(corpus.eval_ids[:feats.n]), \
        "cache eval order must match corpus eval order (same seed/config)."
    lam = cfg.uaas.lam
    conf = feats.meta["conf_llm"].to_numpy()
    ent = feats.meta["entropy_llm"].to_numpy()
    U = lam * ent + (1 - lam) * (1 - conf)
    r_cont = continuous_rank(U, cfg.uaas.r_min, cfg.uaas.r_max)
    r_star = bucket_rank(r_cont, cfg.uaas.rank_buckets)

    eval_texts = corpus.eval_texts[:feats.n]
    eval_labels = feats.labels

    # ---- train adapters ----
    ua = UAASLoRA(cfg)
    tr_texts = corpus.pool_texts[:train_samples]
    tr_labels = corpus.pool_labels[:train_samples]
    examples = ua.build_examples(tr_texts, tr_labels, verbalizer)

    ranks = sorted(set(cfg.uaas.static_ranks) | set(cfg.uaas.rank_buckets))
    names = {r: f"r{r}" for r in ranks}
    for r in ranks:
        ua.add_adapter(names[r], r)
    for r in ranks:
        ua.train_adapter(names[r], examples)
    params = {r: _trainable_params(ua.peft, names[r]) for r in ranks}

    rows, per_sample = [], []

    def summarize(method, preds, losses, ranks_used):
        rows.append({
            "Method": method,
            "Accuracy": M.accuracy(preds, eval_labels),
            "Macro-F1": M.macro_f1(preds, eval_labels),
            "Mean loss": float(np.mean(losses)),
            "Avg rank": float(np.mean(ranks_used)),
            "Avg trainable params": float(np.mean([params[int(r)] for r in ranks_used])),
        })

    # static baselines
    for r in cfg.uaas.static_ranks:
        preds, losses, ru = ua.evaluate(eval_texts, eval_labels, verbalizer, names[r])
        summarize(f"Static LoRA r={r}", preds, losses, ru)

    # adaptive
    adapter_of = [names[int(r)] for r in r_star]
    preds, losses, ru = ua.evaluate(eval_texts, eval_labels, verbalizer, adapter_of)
    summarize("SMART-UAAS (adaptive)", preds, losses, ru)
    for i in range(len(eval_texts)):
        per_sample.append({"id": feats.meta["id"].iloc[i], "label": int(eval_labels[i]),
                           "U": float(U[i]), "r_cont": float(r_cont[i]),
                           "r_star": int(r_star[i]), "pred": int(preds[i]),
                           "loss": float(losses[i])})

    df = pd.DataFrame(rows)
    atomic_write_csv(df, f"{cfg.paths.tables_dir}/table_uaas_{cfg.data.dataset}.csv")
    atomic_write_csv(pd.DataFrame(per_sample),
                     f"{cfg.paths.results_dir}/uaas_master_{cfg.data.dataset}.csv")
    _log.info("\n%s", df.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser(description="SMART-LLM UAAS (Contribution 2)")
    add_config_args(ap)
    ap.add_argument("--train-samples", type=int, default=1000)
    args = ap.parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg.seed)
    run(cfg, train_samples=args.train_samples)


if __name__ == "__main__":
    main()
