"""Stage 1 (GPU, heavy): extract frozen features + ground-truth retrieval benefit.

For every evaluation sample this runs the frozen LLM ``1 + |conditions|`` times:

* one **no-retrieval** pass (with hidden states)  -> h_last/h_mean/h_tokens, Loss_p,
* one pass **per retrieval condition** (clean / random / adversarial) -> Loss_r.

Ground truth (per condition):
    B_true = (Loss_p - Loss_r) / (|Loss_p| + eps)
    oracle = 1[Loss_r < Loss_p]

Everything is written to the sharded feature cache; no training happens here.
This is the ONLY place full RAG is run per sample, and it is offline supervision
— the deployed router never does this (see Contribution 1 / no double inference).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ..config import add_config_args, config_from_args
from ..data.datasets import load_corpus
from ..data.fewshot import build_demo_block
from ..embeddings.encoder import TextEncoder
from ..embeddings.faiss_index import Retriever
from ..llm.backbone import FrozenLLM
from ..llm.prompts import VerbalizerSpec
from ..utils.logging import get_logger
from ..utils.seed import seed_everything
from .cache import FeatureWriter, cache_exists

_log = get_logger("smart_llm.stage1")
EPS = 1e-6


def _now() -> float:
    """perf_counter with a CUDA sync so GPU timings are accurate."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    return time.perf_counter()


def _demo_block(retr_idx_row, demo_lab_row, pool_texts, label_names, n_demos):
    pairs = [(pool_texts[int(idx)], label_names[int(lab)])
             for idx, lab in zip(retr_idx_row[:n_demos], demo_lab_row[:n_demos])]
    return build_demo_block(pairs)


def main():
    ap = argparse.ArgumentParser(description="SMART-LLM Stage-1 feature extraction")
    add_config_args(ap)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: only process the first N eval samples")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg.seed)

    if cache_exists(cfg.paths.cache_dir, cfg.data.dataset) and not args.overwrite:
        _log.warning("Cache already exists for %s (use --overwrite). Exiting.",
                     cfg.data.dataset)
        return

    corpus = load_corpus(cfg)
    verbalizer = VerbalizerSpec(corpus.label_names)
    conditions = cfg.retrieval.conditions
    n_demos = cfg.n_demos

    # ---- embeddings + FAISS (with amortised timing for the efficiency table) ----
    timing = {"search_time": {}}
    encoder = TextEncoder(cfg)
    _log.info("Encoding %d pool docs …", len(corpus.pool_texts))
    t = _now(); pool_emb = encoder.encode_documents(corpus.pool_texts)
    timing["embed_pool_time"] = _now() - t
    _log.info("Encoding %d eval queries …", len(corpus.eval_texts))
    t = _now(); query_emb = encoder.encode_queries(corpus.eval_texts)
    timing["embed_query_time"] = _now() - t
    t = _now()
    retriever = Retriever(cfg).fit(pool_emb, corpus.pool_labels, corpus.pool_ids)
    timing["index_build_time"] = _now() - t

    # precompute retrieval for each condition (vectorised)
    retr = {}
    for cond in conditions:
        t = _now()
        retr[cond] = retriever.retrieve(
            query_emb, k=cfg.retrieval.k, condition=cond,
            query_ids=corpus.eval_ids, query_labels=corpus.eval_labels)
        timing["search_time"][cond] = _now() - t
    timing["n_eval_for_amortization"] = len(corpus.eval_texts)

    # ---- frozen LLM ----
    llm = FrozenLLM(cfg)

    writer = FeatureWriter(cfg.paths.cache_dir, cfg.data.dataset, conditions,
                           corpus.label_names, shard_size=args.shard_size,
                           config_snapshot=cfg.to_dict())

    n_eval = len(corpus.eval_texts)
    if args.limit:
        n_eval = min(n_eval, args.limit)
    _log.info("Processing %d eval samples × (1 + %d conditions) LLM passes …",
              n_eval, len(conditions))

    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_eval), desc="stage1")
    except ImportError:
        iterator = range(n_eval)

    t0 = time.time()
    for i in iterator:
        text = corpus.eval_texts[i]
        label = int(corpus.eval_labels[i])

        # no-retrieval pass (parametric only) + hidden states  [timed: t_p]
        t = _now()
        op = llm.classify(text, verbalizer, demo_block=None,
                          true_label=label, want_hidden=True)
        t_p = _now() - t
        loss_p = op.loss

        scalars = dict(id=corpus.eval_ids[i], label=label, pred_p=op.pred,
                       loss_p=loss_p, conf_llm=op.confidence,
                       entropy_llm=op.entropy, n_prompt_tokens=op.n_prompt_tokens,
                       t_p=t_p)
        vecs = dict(h_last=op.h_last, h_mean=op.h_mean, query_emb=query_emb[i])
        tokens = dict(h_tokens=op.h_tokens, token_mask=op.token_mask)

        conds = {}
        for cond in conditions:
            r = retr[cond]
            demo_block = _demo_block(r.indices[i], r.demo_label_ids[i],
                                     corpus.pool_texts, corpus.label_names, n_demos)
            t = _now()
            orr = llm.classify(text, verbalizer, demo_block=demo_block,
                               true_label=label, want_hidden=False)
            t_r = _now() - t
            loss_r = orr.loss
            btrue = (loss_p - loss_r) / (abs(loss_p) + EPS)
            conds[cond] = dict(
                centroid=r.centroid[i], sim=float(r.sims[i].mean()),
                loss_r=loss_r, pred_r=orr.pred, btrue=btrue,
                oracle=int(loss_r < loss_p),
                retr_idx=r.indices[i], demo_label_ids=r.demo_label_ids[i],
                t_r=t_r, n_tokens_r=orr.n_prompt_tokens)

        writer.add(scalars, vecs, tokens, conds)

    writer.set_timing(timing)
    out_dir = writer.finalize()
    dt = time.time() - t0
    _log.info("Stage-1 done: %d samples in %.1fs -> %s", n_eval, dt, out_dir)


if __name__ == "__main__":
    main()
