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


INVISIBLE = (
    "\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180b-\u180f\u200b-\u200f"
    "\u202a-\u202e\u2060-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0\ufff0-\ufff8"
    "\U0001bca0-\U0001bca3\U0001d173-\U0001d17a\U000e0000-\U000e0fff"
)
"""The characters Unicode marks as ignorable by default (Default_Ignorable_Code_Point),
as the body of a regular expression's class: soft hyphen, zero-width spaces and
joiners, direction marks, bidirectional embeddings, overrides and isolates,
variation selectors, tag characters and the like. A text holds them without
showing them, and a model reads past them."""


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
    """*tag*'s name as a model reads it, up to its end: any invisible character
    between its letters, and no further letter of a name after it. Its end is
    read without consuming anything (not ``\\b``, which a Hangul filler, an
    invisible character Unicode counts a letter, would defeat), so what
    follows it is read once: no two quantifiers compete for the same
    characters, and the pattern stays linear."""
    return f"[{INVISIBLE}]*".join(map(re.escape, tag)) + "(?![A-Za-z0-9_])"


def fence(tag: str, text: str) -> str:
    """*text* inside ``<tag>`` … ``</tag>``, with no closing tag of its own.

    Any closing tag of that name in *text*, in any case, with any spacing,
    invisible character (:data:`INVISIBLE`, a zero-width space inside the name
    included) or trailing attributes (``</TOOL_RESULT >``, ``< / tool_result>``,
    ``</tool_result foo>``, ``</tool_result/>``), is neutralised, so the data
    cannot close the block.
    """
    gap = rf"[\s{INVISIBLE}]*"
    closing = re.compile(rf"<{gap}/{gap}{_tag_name(tag)}[^>]*>", re.IGNORECASE)
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
            rf"<{gap}{name}[^>]*>.*?(?:<{gap}/{gap}{name}[^>]*>|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        text = block.sub(f"[{tag}]", text)
    return text


# -- A frame across a cut ----------------------------------------------------

_TAGS = "|".join(map(re.escape, FENCED_TAGS))
_FRAME_EDGE = re.compile(rf"<({_TAGS})>\n|</({_TAGS})>|“|”")
"""Where a frame :func:`fence` or :func:`quoted` wrote opens or ends: a block's
opening exactly as written (its tag, then a line break), a block's closing
tag, an opening or closing quote mark."""

_LEAD_LIMIT = 120
"""The characters before a quote on its line that a reopened quote repeats: an
author's name or the instruction that quotes it; a longer lead is not one."""


def open_frame(text: str) -> tuple[str, str]:
    """The frame left open at the end of *text*, as what closes it there and
    what opens it again (RFC §6.4, §12.4.1); ``("", "")`` when none is.

    A block of one of :data:`FENCED_TAGS`, opened as :func:`fence` writes it
    (``<tag>`` then a line break) with no ``</tag>`` after it, inside which only
    that closing tag counts, what the block holds being data; or a quote “
    with no ” after it, opened again after what precedes it on its line
    (``Marie · sms: “``). A tag named in running text (``<tool_result> is
    data``) opens nothing.
    """
    tag: str | None = None
    quote_at: int | None = None
    for edge in _FRAME_EDGE.finditer(text):
        mark = edge.group(0)
        if tag is not None:
            tag = None if mark == f"</{tag}>" else tag
        elif quote_at is not None:
            quote_at = None if mark == "”" else quote_at
        elif mark == "“":
            quote_at = edge.start()
        elif edge.group(1):
            tag = edge.group(1)
    if tag is not None:
        return f"\n</{tag}>", f"<{tag}>\n"
    if quote_at is None:
        return "", ""
    lead = text[text.rfind("\n", 0, quote_at) + 1 : quote_at]
    return "”", f"{lead if len(lead) <= _LEAD_LIMIT else ''}“"


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
