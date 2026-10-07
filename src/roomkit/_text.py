"""Text helpers with no dependency inside RoomKit, so any module can import them
(the orchestration and tasks packages load the AI channel as they import).

Text RoomKit did not write and places in a model's context goes through them
(RFC §6.4): quoted inline with :func:`quoted`, or set apart as a block with
:func:`roomkit.tools.fence.fence`. What the runtime gives outside both carries no
text of its own: an :func:`identifier`, a value :func:`one_of` a known set, a
number, or a :func:`person_name`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection
from typing import Any


def bounded_text(text: str, limit: int) -> str:
    """*text* within *limit* characters: cut at a word, the cut ending in "…".

    Never in the middle of a word or a number: a forecast's "14,9 °C" cut to "1"
    was read back as a temperature (RMK-556).
    """
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip() + "…"


def one_line(text: Any) -> str:
    """*text* on one line: each run of whitespace, line breaks included, one space."""
    return " ".join(str(text).split())


_DOUBLE_QUOTES = str.maketrans(dict.fromkeys('"“”„‟«»＂〝〞〟', "'"))
"""Every double quote mark, made a single one inside a quoted text: none is then
left to close the quote, a plain ``"`` included, which a model reads as closing
“ as readily as ”."""


def quoted(text: Any, limit: int) -> str:
    """External *text* quoted inline (RFC §6.4): on one line, within *limit*
    characters, between “ and ” it cannot close."""
    return f"“{bounded_text(one_line(text).translate(_DOUBLE_QUOTES), limit)}”"


_NOT_IN_AN_IDENTIFIER = re.compile(r"[^\w.@+-]+")


def identifier(value: Any, fallback: str, limit: int = 64) -> str:
    """*value* as the runtime gives an identifier outside a quote: letters,
    digits, ``_ . @ + -``, within *limit* characters; *fallback* when nothing is
    left."""
    kept = _NOT_IN_AN_IDENTIFIER.sub("-", str(value or "")).strip("-")[:limit]
    return kept or fallback


def one_of(value: Any, allowed: Collection[str], fallback: str) -> str:
    """*value* when it is one of *allowed*, else *fallback*: a value the runtime
    gives outside a quote comes from a known set."""
    text = str(value)
    return text if text in allowed else fallback


_NOT_IN_A_NAME = re.compile(r"[^\w .'’-]+")


def person_name(value: Any, limit: int = 64) -> str:
    """A person's name as the runtime gives it outside a quote: letters, digits,
    spaces, ``. - '``, on one line, within *limit* characters; ``""`` when
    nothing is left.

    Unquoted, it cannot open a line, a quote or a bracket, nor end the name
    (``:``) where a transcript reads it.
    """
    composed = unicodedata.normalize("NFC", str(value or ""))
    return one_line(_NOT_IN_A_NAME.sub(" ", composed))[:limit].strip()
