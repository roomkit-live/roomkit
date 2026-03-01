"""TTS text filters — strip internal prompt markers before synthesis."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class TTSStreamFilter(ABC):
    """Base class for TTS text filters.

    Supports both streaming (chunk-by-chunk via :meth:`feed`/:meth:`flush`)
    and non-streaming (full text via :meth:`__call__`) usage.

    Subclasses must implement :meth:`feed`, :meth:`flush`, and
    :meth:`reset`.  The default :meth:`__call__` delegates to feed+flush
    but subclasses may override it with a more efficient implementation
    (e.g. a single regex pass).
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
    r"|\[internal[:\s][^\]]*\]",  # single bracket with colon or space
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
        result = _INTERNAL_RE.sub("", text)
        # Collapse multiple spaces left by removal and strip
        return re.sub(r"  +", " ", result).strip()

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []

        while True:
            if not self._inside:
                # Look for "[internal" (common prefix for both formats)
                lower = self._buf.lower()
                idx = lower.find("[internal")
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
        """Return the prefix of *text* that cannot be the start of ``[internal``."""
        # If the text ends with a partial match for "[internal", hold it back.
        tag = "[internal"
        for i in range(1, len(tag)):
            if text.lower().endswith(tag[:i]):
                return text[:-i]
        return text


# ---------------------------------------------------------------------------
# StripBrackets — removes all [...] content
# ---------------------------------------------------------------------------

_BRACKET_RE = re.compile(r"\[[^\]]*\]")


class StripBrackets(TTSStreamFilter):
    """Strip all ``[...]`` bracketed content from TTS text.

    A simpler variant that catches markers like ``[Respond in French]``,
    ``[laughs]``, ``[thinking]``, etc.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def reset(self) -> None:
        self._buf = ""
        self._inside = False

    def __call__(self, text: str) -> str:
        result = _BRACKET_RE.sub("", text)
        return re.sub(r"  +", " ", result).strip()

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
