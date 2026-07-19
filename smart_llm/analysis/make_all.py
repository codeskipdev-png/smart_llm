"""Build every table and figure from the master results CSV."""
from __future__ import annotations

import argparse

from ..config import add_config_args, config_from_args
from ..utils.logging import get_logger
from . import figures, tables

_log = get_logger("smart_llm.analysis")


def run(cfg) -> dict:
    df = tables.load_master(cfg)
    tbls = tables.build_all(cfg)
    figs = figures.build_all(cfg, df)
    for name, t in tbls.items():
        _log.info("\n[%s]\n%s", name, t.to_string(index=False))
    _log.info("Figures: %s", ", ".join(figs))
    return {"tables": tbls, "figures": figs}


def main():
    ap = argparse.ArgumentParser(description="Build SMART-LLM tables + figures")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = config_from_args(args)
    run(cfg)


if __name__ == "__main__":
    main()
