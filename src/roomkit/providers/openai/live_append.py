"""Append chunking for the OpenAI Live API (GPT-Live).

A context append is bounded in tokens (:data:`MAX_APPEND_TOKENS`); the API
refuses a longer one and closes the session. This module measures appends
with the model's byte-pair encoding, or by UTF-8 bytes when it cannot be
loaded, and splits a longer text into appends within the bound, a framed
text closing and reopening its frame at each cut (RFC §6.4, §12.4.1).
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Any, Protocol

from roomkit._text import open_frame
from roomkit.providers.openai.live_events import MAX_APPEND_TOKENS

logger = logging.getLogger("roomkit.providers.openai.live")

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_SPACES = re.compile(r"\s*")

#: The byte-pair encoding of the GPT-5 generation GPT-Live belongs to.
TOKENIZER_ENCODING = "o200k_base"


class Tokenizer(Protocol):
    """What the chunker measures with: a text's tokens, and the bytes a token prefix spells."""

    def encode(self, text: str) -> list[int]: ...

    def decode_bytes(self, tokens: list[int]) -> bytes: ...


class ByteTokenizer:
    """One token per UTF-8 byte: the bound that holds for any byte-pair encoding.

    A four-characters-per-token estimate undercounts identifiers, JSON,
    punctuation and some scripts, and an append the API refuses closes the
    session. Every byte can be its own token, so this bound holds without a
    model-specific tokenizer or a network lookup — at the price of splitting
    ordinary prose about four times more often than the model's own count
    would, and a split spoken append is spoken once per piece.
    """

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode_bytes(self, tokens: list[int]) -> bytes:
        return bytes(tokens)


class _Encoding:
    """A tiktoken encoding as the chunker reads it: special-token text is ordinary text."""

    def __init__(self, encoding: Any) -> None:
        self._encoding = encoding

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, disallowed_special=())

    def decode_bytes(self, tokens: list[int]) -> bytes:
        return self._encoding.decode_bytes(tokens)


BYTES = ByteTokenizer()
_tokenizer: Tokenizer | None = None
_tokenizer_lock = threading.Lock()


def _load_tokenizer() -> Tokenizer:
    """Blocking: import tiktoken and load the encoding, fetched over the network on first use."""
    global _tokenizer
    with _tokenizer_lock:
        if _tokenizer is not None:
            return _tokenizer
        try:
            import tiktoken
        except ImportError:
            logger.warning(
                "GPT-Live context appends are bounded by UTF-8 bytes: tiktoken is not "
                "installed (pip install 'roomkit[realtime-openai]')"
            )
            _tokenizer = BYTES
            return _tokenizer
        try:
            _tokenizer = _Encoding(tiktoken.get_encoding(TOKENIZER_ENCODING))
        except Exception as exc:  # the table could not be fetched; try again next session
            logger.warning(
                "GPT-Live context appends are bounded by UTF-8 bytes this session: %s could "
                "not be loaded (%s: %s)",
                TOKENIZER_ENCODING,
                type(exc).__name__,
                exc,
            )
            return BYTES
        return _tokenizer


async def tokenizer() -> Tokenizer:
    """The tokenizer appends are measured with, loaded off the event loop once per process.

    ``o200k_base`` through tiktoken when it is installed; its table is fetched
    on first use, which is why the load runs in a thread. Without tiktoken, or
    while the table cannot be fetched, :data:`BYTES` bounds the count instead.
    """
    if _tokenizer is not None:
        return _tokenizer
    return await asyncio.to_thread(_load_tokenizer)


def estimated_tokens(text: str) -> int:
    """Bound byte-pair tokens by the number of UTF-8 bytes (see :class:`ByteTokenizer`)."""
    return len(BYTES.encode(text))


def token_count(text: str, tok: Tokenizer | None = None) -> int:
    """A text's cost in tokens, by ``tok`` or the tokenizer loaded for the process."""
    return len((tok or _tokenizer or BYTES).encode(text))


def _window(text: str, start: int, token_limit: int, tok: Tokenizer) -> tuple[int, list[int]]:
    """Where a prefix of *text* from *start* holding more than *token_limit*
    tokens ends, or the end of *text*, with its tokens: what a cut reads,
    without encoding the rest of a long text at every cut (the prefix doubles
    from four characters a token)."""
    span = token_limit * 4
    while True:
        stop = min(start + span, len(text))
        tokens = tok.encode(text[start:stop])
        if len(tokens) > token_limit or stop == len(text):
            return stop, tokens
        span *= 2


def _fits(text: str, start: int, token_limit: int, tok: Tokenizer, lead: str = "") -> bool:
    """Whether *lead* and *text* from *start* are one append within *token_limit*."""
    stop, tokens = _window(text, start, token_limit, tok)
    if stop < len(text) or len(tokens) > token_limit:
        return False
    return not lead or len(tok.encode(lead + text[start:])) <= token_limit


def _sendable(text: str) -> str:
    """*text* as UTF-8 can carry it: a lone surrogate, which no encoding sends,
    made a replacement character."""
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")


def _cut(text: str, start: int, token_limit: int, tok: Tokenizer) -> int:
    """Where the longest piece of *text* from *start* within ``token_limit``
    ends, preferring a sentence or space boundary. Read with a cursor, so a
    long text is never copied once per cut."""
    _, tokens = _window(text, start, token_limit, tok)
    if len(tokens) <= token_limit:
        return len(text)
    # The first ``token_limit`` tokens spell a byte prefix of the text; a
    # multibyte character they cut in half is dropped, which only shortens it.
    end = start + len(tok.decode_bytes(tokens[:token_limit]).decode("utf-8", errors="ignore"))
    if end == start:
        raise ValueError("token_limit cannot fit one UTF-8 character")
    boundaries = [b for b in _SENTENCE_BOUNDARY.finditer(text, start, end) if b.start() > start]
    if boundaries:
        end = boundaries[-1].end()
    else:
        space = text.rfind(" ", start, end)
        if space >= start:
            end = space + 1
    # Re-encoded on its own, a prefix can cost a token more than the slice it
    # came from (a trailing space no longer merges with the word after it).
    while len(tok.encode(text[start:end])) > token_limit:
        space = text.rfind(" ", start, end - 1)
        end = space + 1 if space > start else end - 1
        if end == start:
            raise ValueError("token_limit cannot fit one UTF-8 character")
    return end


def chunk_text(
    text: str, token_limit: int = MAX_APPEND_TOKENS, *, tok: Tokenizer | None = None
) -> list[str]:
    """Split appends within the token bound, preserving inner text.

    Measured with ``tok``, else the tokenizer loaded for the process, else by
    UTF-8 bytes. A text within the bound is one append — what a spoken append
    needs, since the model voices each piece. UTF-8 characters stay whole.
    Splits prefer sentences, then spaces, then character boundaries. RFC
    §12.4.1: bounded appends split rather than truncate or refuse content.
    """
    if token_limit <= 0:
        raise ValueError("token_limit must be positive")
    tok = tok or _tokenizer or BYTES
    text = _sendable(text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = _cut(text, start, token_limit, tok)
        chunks.append(text[start:end])
        start = end
    return chunks


_FRAME_RESERVE = 32
"""Tokens a cut keeps for closing the frame it leaves open (``</tool_result>``
on its own line, measured in bytes when no tokenizer is loaded)."""


def chunk_framed_text(
    text: str, token_limit: int = MAX_APPEND_TOKENS, *, tok: Tokenizer | None = None
) -> list[str]:
    """:func:`chunk_text` for an injected text, which the runtime may have set
    apart in a frame (RFC §6.4, §12.4.1): each cut closes the frame it leaves
    open, and the next append opens it again before the text that follows.

    A text within the bound is one append. Every cut takes text from what is
    left, never from the frame's opening, so each one advances.
    """
    if token_limit <= 2 * _FRAME_RESERVE:
        raise ValueError("token_limit leaves no room for a frame")
    tok = tok or _tokenizer or BYTES
    text, opening, start = _sendable(text).strip(), "", 0
    chunks: list[str] = []
    while start < len(text):
        if _fits(text, start, token_limit, tok, lead=opening):
            chunks.append(opening + text[start:])
            break
        opening = _affordable(opening, token_limit, tok)
        room = token_limit - _FRAME_RESERVE - len(tok.encode(opening))
        end = _cut(text, start, room, tok)
        piece, opening, start = _close_at_cut(opening + text[start:end], text, end)
        if piece:
            chunks.append(piece)
    return chunks


def _affordable(opening: str, token_limit: int, tok: Tokenizer) -> str:
    """*opening*, or its quote mark alone when its lead would take more than a
    quarter of an append."""
    if len(tok.encode(opening)) * 4 <= token_limit or not opening.endswith("“"):
        return opening
    return "“"


def _close_at_cut(piece: str, text: str, end: int) -> tuple[str, str, int]:
    """*piece* closing the frame it leaves open, the opening the next append
    starts with, and where in *text* that append resumes after *end*: a block
    ends and starts over, a quote too, after its author or the instruction
    that quotes it. A frame the cut falls right after the opening of, or right
    before the end of, stays whole on one side rather than leave an empty one
    on the other."""
    closing, opening = open_frame(piece)
    if not closing:
        return piece, "", end
    piece, end = piece.rstrip(), _past_spaces(text, end)
    if text.startswith(closing.strip(), end):
        return piece + closing, "", _past_spaces(text, end + len(closing.strip()))
    if piece.endswith(opening.strip()) and open_frame(piece[: -len(opening.strip())])[0] == "":
        return piece[: -len(opening.strip())].rstrip(), opening, end
    return piece + closing, opening, end


def _past_spaces(text: str, pos: int) -> int:
    """The first position of *text* from *pos* that is not a space."""
    match = _SPACES.match(text, pos)
    return match.end() if match else pos
