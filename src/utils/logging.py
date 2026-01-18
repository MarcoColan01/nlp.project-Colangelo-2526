# src/utils/logging.py
from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str = "emotions-map", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
