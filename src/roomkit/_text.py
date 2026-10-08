"""Text helpers with no dependency inside RoomKit, so any module can import them
(the orchestration and tasks packages load the AI channel as they import).

Text RoomKit did not write and places in a model's context goes through them
(RFC §6.4): set apart as a block with :func:`fence`, or quoted inline with
:func:`quoted`. What the runtime gives outside both carries no text of its own:
an :func:`identifier`, a value :func:`one_of` a known set, a number, or a
:func:`person_name`.
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


INVISIBLE = "\u00ad\u200b-\u200f\u2060-\u2064\ufeff"
"""Characters a text holds without showing them, as the body of a regular
expression's class: a soft hyphen, zero-width spaces and joiners, direction
marks, word joiners, a byte order mark. A model reads past them."""


def one_line(text: Any) -> str:
    """*text* on one line: each run of whitespace, line breaks included, one space."""
    return " ".join(str(text).split())


# -- Blocks --------------------------------------------------------------------

CONVERSATION_SUMMARY_TAG = "conversation_summary"
"""The tag a memory's summary of the conversation is set apart in."""

FENCED_TAGS = ("tool_result", "worker_output", "knowledge", CONVERSATION_SUMMARY_TAG, "context")
"""The tags RoomKit fences external data in: a tool's result, a worker's output,
a knowledge passage, a memory's summary of the conversation, and content a
realtime provider adds to its prompt."""


def _tag_name(tag: str) -> str:
    """*tag*'s name as a model reads it: any invisible character between its
    letters."""
    return f"[{INVISIBLE}]*".join(map(re.escape, tag))


def fence(tag: str, text: str) -> str:
    """*text* inside ``<tag>`` … ``</tag>``, with no closing tag of its own.

    Any closing tag of that name in *text*, in any case, with any spacing,
    invisible character or trailing attributes (``</TOOL_RESULT >``,
    ``< / tool_result>``, ``</tool_\u200bresult>``, ``</tool_result foo>``,
    ``</tool_result/>``), is neutralised, so the data cannot close the block.
    """
    gap = rf"[\s{INVISIBLE}]*"
    closing = re.compile(rf"<{gap}/{gap}{_tag_name(tag)}\b[^>]*>", re.IGNORECASE)
    neutral = f"</{tag}_>"
    body = closing.sub(lambda _match: neutral, text)
    return f"<{tag}>\n{body}\n</{tag}>"


def named_blocks(text: str, tags: tuple[str, ...] = FENCED_TAGS) -> str:
    """*text* with each block of *tags* replaced by its tag in brackets
    (``[tool_result]``), a block cut off before its end included.

    For text about to be cut short, such as a summary: quoting part of a block
    could leave it open, and what follows would then read as data.
    """
    gap = rf"[\s{INVISIBLE}]*"
    for tag in tags:
        name = _tag_name(tag)
        block = re.compile(
            rf"<{gap}{name}\b[^>]*>.*?(?:<{gap}/{gap}{name}\b[^>]*>|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        text = block.sub(f"[{tag}]", text)
    return text


# -- Inline quotes -------------------------------------------------------------

_DOUBLE_QUOTES = str.maketrans(dict.fromkeys('"“”„‟«»＂〝〞〟⹂❝❞❠🙶🙷🙸″‶ʺ˝', "'"))
"""Every double quote mark, made a single one inside a quoted text: none is then
left to close the quote, a plain ``"`` included, which a model reads as closing
“ as readily as ”."""


def quoted(text: Any, limit: int) -> str:
    """External *text* quoted inline (RFC §6.4): on one line, within *limit*
    characters, between “ and ” it cannot close; cut short, it names the blocks
    it holds rather than leave part of one open."""
    flat = one_line(text).translate(_DOUBLE_QUOTES)
    if len(flat) > limit:
        flat = named_blocks(flat)
    return f"“{bounded_text(flat, limit)}”"


# -- What is given unquoted ----------------------------------------------------


def _kept(text: str, extra: str, other: str) -> str:
    """*text* with each character that is neither a letter, one of its marks, a
    digit nor in *extra* made *other*; *text* composed first, so that an accent
    written apart rides its letter."""
    return "".join(
        char if char in extra or unicodedata.category(char)[0] in "LMN" else other
        for char in unicodedata.normalize("NFC", text)
    )


def identifier(value: Any, fallback: str, limit: int = 64) -> str:
    """*value* as the runtime gives an identifier outside a quote: letters and
    their marks, digits, ``_ . @ + -``, within *limit* characters; *fallback*
    when nothing is left."""
    kept = re.sub(r"-{2,}", "-", _kept(str(value or ""), "_.@+-", "-"))
    return kept.strip("-")[:limit] or fallback


def one_of(value: Any, allowed: Collection[str], fallback: str) -> str:
    """*value* when it is one of *allowed*, else *fallback*: a value the runtime
    gives outside a quote comes from a known set."""
    text = str(value)
    return text if text in allowed else fallback


def person_name(value: Any, limit: int = 64) -> str:
    """A person's name as the runtime gives it outside a quote: letters and their
    marks, digits, spaces, ``. - _ #`` and apostrophes, on one line, within
    *limit* characters; ``""`` when nothing is left.

    Unquoted, it cannot open a line, a quote or a bracket, nor end the name
    (``:``) where a transcript reads it.
    """
    return one_line(_kept(str(value or ""), " .-_#'’", " "))[:limit].strip()
