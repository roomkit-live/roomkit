"""Argparse ``type=`` validators shared by the examples.

Each example owns its own flags — those are part of what it teaches — but
the validation plumbing behind them is not. Using these keeps a bad value
reported by argparse (``--workspace: no such directory: /nope``) instead of
surfacing as a traceback from inside ``main()``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def non_negative_int(value: str) -> int:
    """``type=`` for a count or budget that may be zero but never negative."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def existing_directory(value: str) -> Path:
    """``type=`` for a directory the example will work in.

    Expands ``~``, resolves to an absolute path (what channels asking for a
    ``cwd`` require), and refuses anything that is not an existing directory.
    """
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"no such directory: {path}")
    return path


__all__ = ["existing_directory", "non_negative_int"]
