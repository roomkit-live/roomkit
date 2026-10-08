"""TTS text filters — strip internal prompt markers before synthesis."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import Literal

from roomkit.telemetry.redaction import redact

logger = logging.getLogger("roomkit.voice.tts.filters")


def _tidy(text: str) -> str:
    """*text* without the double spaces a removal leaves, nor surrounding blanks."""
    return re.sub(r"  +", " ", text).strip()


class TTSStreamFilter(ABC):
    """Base class for TTS text filters.

    Supports both streaming (chunk-by-chunk via :meth:`feed`/:meth:`flush`)
    and non-streaming (full text via :meth:`__call__`) usage.

    Subclasses must implement :meth:`feed`, :meth:`flush`, and
    :meth:`reset`.  The default :meth:`__call__` delegates to feed+flush
    but subclasses may override it with a more efficient implementation
    (e.g. a single regex pass).

    A voice channel streams each response through its own copy of its filter
    (``copy.deepcopy``), so responses streamed at once in several rooms never
    share what a filter buffers: a filter holding something that cannot be
    copied defines ``__deepcopy__``.
    """

    @abstractmethod
    def feed(self, chunk: str) -> str:
        """Process one streaming token/chunk. Return cleaned text (may be empty)."""

    @abstractmethod
    def flush(self) -> str:
        """Flush any buffered text at end-of-stream. Return remaining cleaned text."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new utterance."""

    def __call__(self, text: str) -> str:
        """Non-streaming convenience: filter a complete text string."""
        self.reset()
        result = self.feed(text)
        result += self.flush()
        return result


# ---------------------------------------------------------------------------
# StripInternalTags — removes [internal]...[/internal] and [internal: ...] blocks
# ---------------------------------------------------------------------------

# Paired tags: [internal]...[/internal]
# Single bracket: [internal: ...] or [internal ...] (AI "thinking" style)
_INTERNAL_RE = re.compile(
    r"\[internal\].*?\[/internal\]"  # paired tags
    r"|\[internal[:\s][^\]]*\]"  # single bracket with colon or space
    r"|\[/internal\]",  # stray closing tag (AI mixed formats)
    re.DOTALL | re.IGNORECASE,
)


class StripInternalTags(TTSStreamFilter):
    """Strip ``[internal]...[/internal]`` and ``[internal: ...]`` blocks.

    Handles two formats that AI models commonly produce:

    - **Paired tags**: ``[internal]reasoning here[/internal] spoken text``
    - **Single bracket**: ``[internal: reasoning here] spoken text``

    In streaming mode, buffers text when ``[internal`` is detected and
    discards everything up to the matching close.  Text outside tags is
    passed through immediately.

    In non-streaming mode (``__call__``), a single regex removes all
    tagged blocks.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False
        # "paired" = [internal]...[/internal], "single" = [internal: ...]
        self._mode: str = ""

    def reset(self) -> None:
        self._buf = ""
        self._inside = False
        self._mode = ""

    def __call__(self, text: str) -> str:
        return _tidy(_INTERNAL_RE.sub("", text))

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []

        while True:
            if not self._inside:
                lower = self._buf.lower()

                # Look for "[internal" (common prefix for both formats)
                idx = lower.find("[internal")
                # Also look for stray "[/internal]" (AI mixed formats)
                stray_idx = lower.find("[/internal]")

                # Handle stray closing tag if it comes first
                if stray_idx != -1 and (idx == -1 or stray_idx < idx):
                    if stray_idx > 0:
                        out.append(self._buf[:stray_idx])
                    self._buf = self._buf[stray_idx + len("[/internal]") :]
                    continue

                if idx == -1:
                    # No opening tag found.  Emit everything except a
                    # trailing partial that *could* be the start of a tag.
                    safe = self._safe_prefix(self._buf)
                    if safe:
                        out.append(safe)
                        self._buf = self._buf[len(safe) :]
                    break

                # Emit text before the tag
                if idx > 0:
                    out.append(self._buf[:idx])

                # Determine format from the character after "[internal"
                rest = self._buf[idx + len("[internal") :]
                if not rest:
                    # Need more input to determine format
                    self._buf = self._buf[idx:]
                    break

                if rest[0] == "]":
                    # Paired tag: [internal]...[/internal]
                    self._buf = rest[1:]
                    self._inside = True
                    self._mode = "paired"
                elif rest[0] in (":", " "):
                    # Single bracket: [internal: ...] or [internal ...]
                    self._buf = rest[1:]
                    self._inside = True
                    self._mode = "single"
                else:
                    # Not a tag (e.g. "[internally]") — emit "[" and retry
                    out.append(self._buf[idx])
                    self._buf = self._buf[idx + 1 :]
            else:
                if self._mode == "paired":
                    # Look for closing tag [/internal]
                    close_idx = self._buf.lower().find("[/internal]")
                    if close_idx == -1:
                        break
                    self._buf = self._buf[close_idx + len("[/internal]") :]
                else:
                    # Single bracket — look for ]
                    close_idx = self._buf.find("]")
                    if close_idx == -1:
                        break
                    self._buf = self._buf[close_idx + 1 :]
                self._inside = False
                self._mode = ""

        return "".join(out)

    def flush(self) -> str:
        if self._inside:
            # Unclosed tag — discard the buffered content
            self._buf = ""
            self._inside = False
            self._mode = ""
            return ""
        remaining = self._buf
        self._buf = ""
        return remaining

    @staticmethod
    def _safe_prefix(text: str) -> str:
        """Return the prefix of *text* that cannot be the start of a tag."""
        # Hold back partial matches for both "[internal" and "[/internal".
        lower = text.lower()
        for tag in ("[/internal", "[internal"):
            for i in range(1, len(tag)):
                if lower.endswith(tag[:i]):
                    return text[:-i]
        return text


# ---------------------------------------------------------------------------
# StripBrackets — removes all [...] content
# ---------------------------------------------------------------------------

_BRACKET_RE = re.compile(r"\[[^\]]*\]")


class StripBrackets(TTSStreamFilter):
    """Strip ``[...]`` bracketed content from TTS text, but the tags in *keep*.

    Catches markers like ``[Respond in French]``, ``[laughs]``,
    ``[thinking]``, etc. A TTS that reads some tags as sounds keeps those:
    ``StripBrackets(keep=("laugh", "sigh"))`` passes ``[laugh]`` and
    ``[Sigh]`` through and drops ``[smiles]``.

    Args:
        keep: Tags passed through, without brackets, matched ignoring case
            and surrounding spaces.
    """

    def __init__(self, keep: Iterable[str] = ()) -> None:
        self._keep = frozenset(tag.strip().lower() for tag in keep)
        self._buf = ""
        self._inside = False

    def reset(self) -> None:
        self._buf = ""
        self._inside = False

    def _kept(self, tag: str) -> bool:
        return tag.strip().lower() in self._keep

    def _replace(self, match: re.Match[str]) -> str:
        return match.group(0) if self._kept(match.group(0)[1:-1]) else ""

    def __call__(self, text: str) -> str:
        return _tidy(_BRACKET_RE.sub(self._replace, text))

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []

        while True:
            if not self._inside:
                idx = self._buf.find("[")
                if idx == -1:
                    out.append(self._buf)
                    self._buf = ""
                    break
                if idx > 0:
                    out.append(self._buf[:idx])
                self._buf = self._buf[idx + 1 :]
                self._inside = True
            else:
                idx = self._buf.find("]")
                if idx == -1:
                    break
                tag = self._buf[:idx]
                if self._kept(tag):
                    out.append(f"[{tag}]")
                self._buf = self._buf[idx + 1 :]
                self._inside = False

        return "".join(out)

    def flush(self) -> str:
        if self._inside:
            # Unclosed bracket — discard buffered content
            self._buf = ""
            self._inside = False
            return ""
        remaining = self._buf
        self._buf = ""
        return remaining


# ---------------------------------------------------------------------------
# StripEmoji — removes emoji, which a TTS reads aloud or garbles
# ---------------------------------------------------------------------------

_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, emoticons, flags, supplemental symbols
    "\U00002300-\U000023ff"  # technical symbols (⌚ ⏰)
    "\U00002600-\U000027bf"  # miscellaneous symbols and dingbats (☀ ❤ ✅)
    "\U00002b00-\U00002bff"  # arrows and stars (⬆ ⭐)
    "\U0000fe0f"  # emoji presentation selector
    "\U0000200d"  # zero-width joiner of composed emoji
    "\U000020e3"  # keycap
    "\U000e0020-\U000e007f"  # tag sequences (subdivision flags)
    "]+"
)


class StripEmoji(TTSStreamFilter):
    """Strip emoji from TTS text.

    Language models add emoji to replies even when told not to; a TTS then
    names them or produces a stray sound. Each code point is removed on its
    own, so a streamed emoji is caught whichever chunks it spans. The text
    stored in the conversation is left as the model wrote it.
    """

    def reset(self) -> None:
        pass

    def __call__(self, text: str) -> str:
        return _tidy(_EMOJI_RE.sub("", text))

    def feed(self, chunk: str) -> str:
        return _EMOJI_RE.sub("", chunk)

    def flush(self) -> str:
        return ""


# ---------------------------------------------------------------------------
# StripTechnicalText — never read a tool call, a note to self or a separator
# ---------------------------------------------------------------------------

_TECHNICAL_START = re.compile(r"[{(-]")

_JSON_OPENING = re.compile(r'\{\s*"')
"""A JSON object as a model writes one: its first key is quoted."""

_JSON_OPENING_SO_FAR = re.compile(r"\{\s*$")

_NOTE_OPENING = re.compile(r"\(\s*(?:note|nb)\s*:", re.IGNORECASE)

_NOTE_OPENING_SO_FAR = re.compile(r"\(\s*(?:n|no|not|note\s*|nb\s*)?$", re.IGNORECASE)

_SEPARATOR_DASHES = 3

_Removed = Literal["json", "note", "separator"]

_OPENINGS: dict[str, tuple[re.Pattern[str], re.Pattern[str], Literal["json", "note"]]] = {
    "{": (_JSON_OPENING, _JSON_OPENING_SO_FAR, "json"),
    "(": (_NOTE_OPENING, _NOTE_OPENING_SO_FAR, "note"),
}
"""What a ``{`` or a ``(`` may open: the opening that decides it, the start of
one still too short to tell, and what the buffer then holds open."""


class StripTechnicalText(TTSStreamFilter):
    """Strip the technical text a model may write into a spoken reply.

    - **A JSON object**, recognised by its quoted first key (``{"``, spaces
      allowed): a tool call or a tool result the model wrote as words instead
      of calling the tool. Nested objects go with it, and a brace inside one
      of its strings does not end it.
    - **A note to itself**: ``(Note: ...)`` or ``(NB: ...)``, any case.
    - **A separator**: three dashes or more.

    Whatever the model, a reply may carry them: a model may write its next
    tool call as text, with a result it made up, or leave a note for itself.
    Removing it from the voice does not make the call happen. The text stored
    in the conversation is left as the model wrote it, so the slip stays
    visible; it is just not heard. Each JSON object or note removed is logged
    as a warning with its length (its text at DEBUG, through
    :func:`~roomkit.telemetry.redaction.redact`); a separator at DEBUG.

    A parenthesis, a brace or a dash of ordinary text passes: ``(about ten
    minutes)``, ``{x}``, ``well-known``, ``--``. A JSON object or a note still
    open when the stream ends is removed. The rules do not know who a note is
    for: ``(Note: holidays may delay it)`` written for the listener goes too,
    and ``3---2`` reads ``32``. What they do not cover stays: reasoning in
    plain words ("Need no tool."), which no rule tells from an answer, and the
    fence or brackets around a removed object (a Markdown code block, a JSON
    array: chain :class:`StripBrackets` for the brackets).

    In streaming mode it works on the tokens, before a reply is cut into
    sentences, so an object holding a sentence's full stop is removed whole.
    """

    _mode: Literal["", "json", "note"]
    """What the buffer holds open: nothing, a JSON object or a note."""
    _depth: int
    _scanned: int
    """How much of the open object or note is scanned."""
    _in_string: bool
    _escaped: bool
    _buf: str
    _blank: bool
    """What was spoken so far ends on a blank (or nothing was)."""
    _trim: bool
    """The spacing that follows a removal is not spoken."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = ""
        self._blank = True
        self._trim = False
        self._settle()

    def __call__(self, text: str) -> str:
        # Its own state: a whole text never resets a reply this instance streams.
        whole = type(self)()
        return _tidy(whole.feed(text) + whole.flush())

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        while self._buf and self._step(out, ended=False):
            pass
        return "".join(out)

    def flush(self) -> str:
        out: list[str] = []
        while self._buf and self._step(out, ended=True):
            pass
        if self._mode:
            self._drop(len(self._buf))  # still open at the end
        self.reset()
        return "".join(out)

    def _step(self, out: list[str], *, ended: bool) -> bool:
        """Move the buffer on by one decision; ``False`` when it needs more text."""
        if self._mode == "json":
            return self._close_json()
        if self._mode == "note":
            return self._close_note()
        found = _TECHNICAL_START.search(self._buf)
        if found is None:
            self._emit(out, self._buf)
            self._buf = ""
            return False
        self._emit(out, self._buf[: found.start()])
        self._buf = self._buf[found.start() :]
        return self._open(out, ended=ended)

    def _open(self, out: list[str], *, ended: bool) -> bool:
        """Decide what the ``{``, ``(`` or ``-`` the buffer starts with opens."""
        buf = self._buf
        if buf[0] == "-":
            dashes = len(buf) - len(buf.lstrip("-"))
            if dashes == len(buf) and not ended:
                return False  # more dashes may come
            if dashes >= _SEPARATOR_DASHES:
                self._removed("separator", buf[:dashes])
                self._trim = self._blank
            else:
                self._emit(out, buf[:dashes])
            self._buf = buf[dashes:]
            return True
        opening, so_far, mode = _OPENINGS[buf[0]]
        if opening.match(buf):
            self._mode = mode
            return True
        if so_far.match(buf) and not ended:
            return False  # too early to tell
        self._emit(out, buf[0])
        self._buf = buf[1:]
        return True

    def _emit(self, out: list[str], text: str) -> None:
        """Add plain *text* to what is spoken, without doubling the spacing
        around a removal."""
        if self._trim:
            text = text.lstrip()
            self._trim = not text
        if text:
            self._blank = text[-1].isspace()
        out.append(text)

    def _close_json(self) -> bool:
        """Scan the open object to its closing brace; ``False`` while it is open."""
        for i in range(self._scanned, len(self._buf)):
            char = self._buf[i]
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._in_string = False
            elif char == '"':
                self._in_string = True
            elif char in "{}":
                self._depth += 1 if char == "{" else -1
                if self._depth == 0:
                    return self._drop(i + 1)
        self._scanned = len(self._buf)
        return False

    def _close_note(self) -> bool:
        """Scan the open note to its closing parenthesis; ``False`` while it is open."""
        for i in range(self._scanned, len(self._buf)):
            if self._buf[i] in "()":
                self._depth += 1 if self._buf[i] == "(" else -1
                if self._depth == 0:
                    return self._drop(i + 1)
        self._scanned = len(self._buf)
        return False

    def _drop(self, end: int) -> bool:
        """Remove the open object or note, which ends at *end*."""
        if self._mode:
            self._removed(self._mode, self._buf[:end])
        self._buf = self._buf[end:]
        self._settle()
        self._trim = self._blank
        return True

    def _settle(self) -> None:
        """Nothing open any more: the buffer reads as plain text."""
        self._mode, self._depth, self._scanned = "", 0, 0
        self._in_string = self._escaped = False

    @staticmethod
    def _removed(kind: _Removed, text: str) -> None:
        if kind == "separator":
            logger.debug("Separator kept out of speech")
            return
        what = "JSON object" if kind == "json" else "note"
        logger.warning("A %s (%d chars) was kept out of speech", what, len(text))
        logger.debug("Kept out of speech: %s", redact(text))


# ---------------------------------------------------------------------------
# TTSFilterChain — several filters as one
# ---------------------------------------------------------------------------


class TTSFilterChain(TTSStreamFilter):
    """Run several filters in order, as one ``tts_filter``.

    ``VoiceChannel`` takes one filter: ``TTSFilterChain(StripEmoji(),
    StripBrackets(keep=...))`` gives it both. Each chunk goes through every
    filter in turn; at the end of the stream each filter's flush goes through
    the filters after it.
    """

    def __init__(self, *filters: TTSStreamFilter) -> None:
        self._filters = filters

    def __call__(self, text: str) -> str:
        for f in self._filters:
            text = f(text)
        return text

    def reset(self) -> None:
        for f in self._filters:
            f.reset()

    def feed(self, chunk: str) -> str:
        for f in self._filters:
            chunk = f.feed(chunk)
        return chunk

    def flush(self) -> str:
        out = ""
        for i, f in enumerate(self._filters):
            tail = f.flush()
            for later in self._filters[i + 1 :]:
                tail = later.feed(tail)
            out += tail
        return out


# ---------------------------------------------------------------------------
# filtered_stream — wrap an async token stream through a TTSStreamFilter
# ---------------------------------------------------------------------------


async def filtered_stream(
    source: AsyncIterator[str],
    tts_filter: TTSStreamFilter,
) -> AsyncIterator[str]:
    """Wrap an async token stream through a :class:`TTSStreamFilter`.

    Yields non-empty cleaned chunks.  Calls :meth:`~TTSStreamFilter.reset`
    at the start and :meth:`~TTSStreamFilter.flush` at the end.
    """
    tts_filter.reset()
    async for chunk in source:
        cleaned = tts_filter.feed(chunk)
        if cleaned:
            yield cleaned
    remaining = tts_filter.flush()
    if remaining:
        yield remaining
