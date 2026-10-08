"""Text helpers with no dependency inside RoomKit, so any module can import them
(the orchestration and tasks packages load the AI channel as they import).

Text RoomKit did not write and places in a model's context goes through them
(RFC §6.4): set apart as a block with :func:`fence`, or quoted inline with
:func:`quoted`. What the runtime gives outside both carries no text of its own:
an :func:`identifier`, a value :func:`one_of` a known set, a number, or a
:func:`person_name`.
"""

from __future__ import annotations

import functools
import json
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
    "\x00-\x08\x0e-\x1f\x7f-\x9f\ud800-\udfff"
)
"""The characters a text holds without showing them, as the body of a regular
expression's class: those Unicode marks as ignorable by default
(Default_Ignorable_Code_Point: soft hyphen, zero-width spaces and joiners,
direction marks, bidirectional embeddings, overrides and isolates, variation
selectors, tag characters and the like), and the control characters other than
whitespace and the lone surrogates a provider may strip before it sends a
text. A model reads past them, and a stripped one leaves its neighbours
side by side."""


def one_line(text: Any) -> str:
    """*text* on one line: each run of whitespace, line breaks included, one space."""
    return " ".join(str(text).split())


def json_line(value: Any) -> str:
    """*value* as JSON on one line: the line separators a JSON string keeps raw
    (U+2028, U+2029, U+0085) escaped, so a text inside cannot start a line of
    its own (RFC §6.4)."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
        .replace("\x85", "\\u0085")
    )


# -- Blocks --------------------------------------------------------------------

CONVERSATION_SUMMARY_TAG = "conversation_summary"
"""The tag a memory's summary of the conversation is set apart in."""

FENCED_TAGS = (
    "tool_result",
    "worker_output",
    "knowledge",
    CONVERSATION_SUMMARY_TAG,
    "context",
    "task",
    "vision",
)
"""The tags RoomKit fences external data in: a tool's result, a worker's output,
a knowledge passage, a memory's summary of the conversation, content a
realtime provider adds to its prompt, the goal or task an orchestration
strategy copies into another model's prompt, and what a vision provider saw."""


_OPEN = "<\u02c2\u2039\u2329\u27e8\u3008"
_SLASH = "/\u2044\u2215"
_CLOSE = ">\u02c3\u203a\u232a\u27e9\u3009"
"""A tag's angle brackets and slash as a model reads them beyond what NFKC
folds onto them (fullwidth and small forms, see :func:`_lookalikes`): their
modifier, quotation, angle and mathematical look-alikes."""

_GAP = rf"[\s{INVISIBLE}\u0300-\u036f\u2800\ufff9-\ufffb]*"
"""Room between a tag's brackets, slash and name: spacing, the invisible
characters, combining marks, a braille blank, interlinear annotation marks."""


_CONFUSABLES = {
    "a": "\u0430\u0251\u03b1",
    "b": "\u044c",
    "c": "\u0441\u03f2\u1d04",
    "d": "\u0501",
    "e": "\u0435\u04bd",
    "g": "\u0261\u0581",
    "h": "\u04bb\u0570",
    "i": "\u0456\u03b9\u0269\u04cf\u0131",
    "j": "\u0458\u03f3",
    "k": "\u03ba\u043a",
    "l": "\u04cf\u01c0",
    "m": "\u043c",
    "n": "\u0578",
    "o": "\u043e\u03bf\u03c3\u0585\u1d0f",
    "p": "\u0440\u03c1",
    "q": "\u051b\u0566",
    "r": "\u0433",
    "s": "\u0455\ua731",
    "t": "\u0442\u03c4",
    "u": "\u03c5\u057d\u1d1c",
    "v": "\u03bd\u0475\u1d20",
    "w": "\u051d\u0461\u1d21",
    "x": "\u0445\u03c7",
    "y": "\u0443\u04af\u03b3",
    "z": "\u1d22",
}
"""Letters of other scripts a model reads as a Latin one beyond what NFKC folds:
the Cyrillic, Greek and Armenian homoglyphs (``о``, ``ο``, ``օ`` for ``o``), and
small capitals; their capitals match through case folding."""


@functools.cache
def _lookalikes() -> dict[str, str]:
    """Each lowercase ASCII letter, digit, underscore, bracket and slash, with
    every character NFKC folds onto it under case folding (fullwidth,
    mathematical, circled, small forms: ``ｔ``, ``𝐭``, ``ⓣ``, ``＜``) and its
    homoglyphs in other scripts (:data:`_CONFUSABLES`), as the body of a
    regular expression's class. Read once, on first use."""
    found = {char: [re.escape(char)] for char in "abcdefghijklmnopqrstuvwxyz0123456789_<>/"}
    for code in range(0x80, 0x110000):
        if 0xD800 <= code <= 0xDFFF:
            continue
        folded = unicodedata.normalize("NFKC", chr(code)).casefold()
        if folded in found:
            found[folded].append(re.escape(chr(code)))
    for char, homoglyphs in _CONFUSABLES.items():
        found[char] += map(re.escape, homoglyphs)
    return {char: "".join(forms) for char, forms in found.items()}


def _class(char: str, more: str = "") -> str:
    """A regular expression class of *char* as a model reads it."""
    return f"[{_lookalikes()[char]}{re.escape(more)}]"


def _tag_name(tag: str, *, exact: bool = False) -> str:
    """*tag*'s name as a model reads it: each letter in any of its forms (case,
    NFKC look-alikes), any invisible character between them, and no letter,
    digit or underscore of a longer name after it. A hyphen, a dot or any
    other mark ends it: read as a closing tag, such a name closes the block
    for a model, so a closing tag errs toward one. *exact*, for an opening
    tag, which names a block only as written: the name as written, ended by a
    space, an invisible character, a slash or the bracket (``<task-list>`` is
    another tag). The end is read without consuming anything."""
    if exact:
        letters = f"[{INVISIBLE}]*".join(map(re.escape, tag))
        return letters + rf"(?=[\s{INVISIBLE}/>])"
    letters = f"[{INVISIBLE}]*".join(_class(char) for char in tag)
    return letters + "(?![A-Za-z0-9_])"


@functools.cache
def _closing_tag(tag: str) -> re.Pattern[str]:
    """Where a closing tag of *tag* starts, as a model reads it: a bracket, one
    slash or more (an escaped one included, ``<\\/``), the name."""
    slashes = rf"(?:\\?{_class('/', _SLASH)}{_GAP})+"
    return re.compile(f"{_class('<', _OPEN)}{_GAP}{slashes}{_tag_name(tag)}", re.IGNORECASE)


@functools.cache
def _loose_opening_tag(tag: str) -> re.Pattern[str]:
    """Where an opening tag of *tag* starts, as a model reads it."""
    return re.compile(f"{_class('<', _OPEN)}{_GAP}{_tag_name(tag)}", re.IGNORECASE)


@functools.cache
def _opening_tag(tag: str) -> re.Pattern[str]:
    """Where an opening tag of *tag* starts, as written."""
    return re.compile(f"<{_GAP}{_tag_name(tag, exact=True)}", re.IGNORECASE)


@functools.cache
def _tag_end() -> re.Pattern[str]:
    """The bracket that ends a tag, in any of its forms."""
    return re.compile(_class(">", _CLOSE))


def _next_tag(text: str, start: re.Pattern[str], pos: int) -> tuple[int, int] | None:
    """The first tag *start* opens at or after *pos*, through the first bracket
    that ends it, whatever its attributes hold; ``None`` when none is ended.

    Read from *pos* forward once: no bracket after a tag's start means none
    after any later start either, so a text of tags never ended costs one
    pass, however long."""
    begun = start.search(text, pos)
    if begun is None:
        return None
    ended = _tag_end().search(text, begun.end())
    return None if ended is None else (begun.start(), ended.end())


def fence(tag: str, text: str) -> str:
    """*text* inside ``<tag>`` … ``</tag>``, with no closing tag of its own.

    Any closing tag of that name in *text* a model could read as one, its
    end bracket there or not, is neutralised where it starts (``</tag`` made
    ``</tag_``, what follows kept): in any case and spacing, with an invisible
    or control character anywhere in it, with NFKC look-alikes of its letters,
    brackets and slash (``＜／ｔｏｏｌ_result＞``, ``𝐭𝐨𝐨𝐥_result``), with
    several slashes or an escaped one (``<\\/tool_result>``), with attributes
    of any length or a mark after the name (``</tool_result.>``). An opening
    tag of that name is neutralised the same way (``<tag_``), so no reader
    tracking nesting reads the runtime's text after the block as data. Read in
    linear time.
    """
    body = _closing_tag(tag).sub(f"</{tag}_", text)
    body = _loose_opening_tag(tag).sub(f"<{tag}_", body)
    return f"<{tag}>\n{body}\n</{tag}>"


def named_blocks(text: str, tags: tuple[str, ...] = FENCED_TAGS) -> str:
    """*text* with each block of *tags* replaced by its tag in brackets
    (``[tool_result]``), a block cut off before its end included.

    For text about to be cut short, such as a summary: quoting part of a block
    could leave it open, and what follows would then read as data.
    """
    for tag in tags:
        text = _named(text, tag)
    return text


def _named(text: str, tag: str) -> str:
    """*text* with each *tag* block, from its opening through its closing or
    the text's end, replaced by ``[tag]``; read in linear time."""
    opening, closing = _opening_tag(tag), _closing_tag(tag)
    parts: list[str] = []
    pos = 0
    while (span := _next_tag(text, opening, pos)) is not None:
        ended = _next_tag(text, closing, span[1])
        parts += [text[pos : span[0]], f"[{tag}]"]
        pos = ended[1] if ended is not None else len(text)
    return "".join(parts) + text[pos:]


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

_DOUBLE_QUOTES = str.maketrans(
    {
        **dict.fromkeys('"“”„‟«»＂〝〞〟⹂❝❞❠🙶🙷🙸″‶ʺ˝ˮ״〃‴⁗\U000e0022', "'"),
        **dict.fromkeys("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"),
    }
)
"""Every double quote mark, made a single one inside a quoted text: none is then
left to close the quote, a plain ``"`` included, which a model reads as closing
“ as readily as ”. The bidirectional embeddings, overrides and isolates are
dropped: one left open would show the quote's end, and what follows, reversed."""


def quoted(text: Any, limit: int) -> str:
    """External *text* quoted inline (RFC §6.4): on one line, within *limit*
    characters, between “ and ” it cannot close; cut short, it names the blocks
    it holds rather than leave part of one open."""
    flat = one_line(text).translate(_DOUBLE_QUOTES)
    if len(flat) > limit:
        flat = named_blocks(flat)
    return f"“{bounded_text(flat, limit)}”"


# -- What is given unquoted ----------------------------------------------------


_IMPOSTORS = frozenset("ːꓽˮʺ")
"""Letters that read as a colon or a double quote: kept in a name, they would end
it (``Adminː refund approved. Bob``) or open a quote."""


def _kept(text: str, extra: str, other: str) -> str:
    """*text* with each character that is neither a letter, one of its marks, a
    digit nor in *extra* made *other*, a letter that reads as a colon or a quote
    included; *text* composed first, so that an accent written apart rides its
    letter."""
    return "".join(
        char
        if char not in _IMPOSTORS and (char in extra or unicodedata.category(char)[0] in "LMN")
        else other
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
