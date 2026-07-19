"""Phase-1A orchestrator (CDKA validation).

Assumes Stage-1 features already exist (run ``generate_features`` on the GPU box
first — it is the only heavy step). Then:

    1. Stage-2 CDKA training  -> master CSV + metrics JSON  (Experiments 1 & 2)
    2. Router-signal ablation  -> supplementary Table 6
    3. Tables 1-5 + Figures 1-5 (Experiment 3 lives in Tables 1 & 4)
"""
from __future__ import annotations

import argparse

from ..config import add_config_args, config_from_args
from ..utils.logging import get_logger
from ..utils.seed import seed_everything
from . import ablation, train_cdka
from .cache import cache_exists

_log = get_logger("smart_llm.phase1a")


def main():
    ap = argparse.ArgumentParser(description="SMART-LLM Phase-1A (CDKA)")
    add_config_args(ap)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg.seed)

    if not cache_exists(cfg.paths.cache_dir, cfg.data.dataset):
        _log.error("No Stage-1 cache for '%s'. Run first:\n"
                   "  python -m smart_llm.experiments.generate_features "
                   "--config <cfg> --dataset %s", cfg.data.dataset, cfg.data.dataset)
        raise SystemExit(1)

    _log.info("== Stage 2: CDKA training ==")
    train_cdka.run(cfg, device=args.device)
    _log.info("== Router-signal ablation ==")
    ablation.run(cfg, device=args.device)
    _log.info("== Tables + figures ==")
    from ..analysis import make_all
    make_all.run(cfg)
    _log.info("Phase-1A complete. See %s / %s / %s",
              cfg.paths.results_dir, cfg.paths.tables_dir, cfg.paths.figures_dir)


if __name__ == "__main__":
    main()
