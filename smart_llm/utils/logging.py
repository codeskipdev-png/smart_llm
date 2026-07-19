"""Small logging helper: consistent format, optional file sink, singleton-ish."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONFIGURED = set()


def get_logger(name: str = "smart_llm",
               level: int = logging.INFO,
               log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger
    logger.setLevel(level)
    logger.propagate = False

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(stream)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(fh)

    _CONFIGURED.add(name)
    return logger
