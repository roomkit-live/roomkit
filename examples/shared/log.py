"""Shared logging setup for RoomKit examples."""

from __future__ import annotations

import logging


def setup_logging(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Configure logging and return a named logger.

    Sets up ``basicConfig`` with a standard format, then returns
    ``logging.getLogger(name)``.  Safe to call multiple times — the
    ``basicConfig`` call is a no-op after the first invocation.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return logging.getLogger(name)
