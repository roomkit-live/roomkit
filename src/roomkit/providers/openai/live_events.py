"""Wire vocabulary and helpers for the OpenAI Live API (GPT-Live).

The Live API is a full-duplex speech-to-speech protocol: one ``session.start``,
then audio and transcript deltas in both directions, delegations the model
hands to a backend, and context appends the client returns. This module holds
what the provider needs to speak it without a session of its own: the event
names, the turn grouper that synthesizes the response and speech boundaries
the wire lacks (RFC §12.4.1), and the append chunker that honours the
per-append token bound.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from roomkit._text import open_frame
from roomkit.providers.ai.tool_declaration import declared_parameters

logger = logging.getLogger("roomkit.providers.openai.live")

# --- Client events -----------------------------------------------------------

EVT_SESSION_START = "session.start"
EVT_SESSION_UPDATE = "session.update"
EVT_SESSION_CLOSE = "session.close"
EVT_INPUT_AUDIO_APPEND = "session.input_audio.append"
EVT_INSTRUCTIONS_APPEND = "session.instructions.append"
EVT_THINKING_APPEND = "session.thinking.append"
EVT_COMMENTARY_APPEND = "session.commentary.append"
EVT_RESPONSE_ITEM_CREATE = "response.item.create"
EVT_RESPONSE_CREATE = "response.create"

# --- Server events -----------------------------------------------------------

EVT_SESSION_STARTED = "session.started"
EVT_SESSION_UPDATED = "session.updated"
EVT_SESSION_CLOSED = "session.closed"
EVT_SESSION_USAGE_UPDATED = "session.usage.updated"
EVT_OUTPUT_AUDIO_DELTA = "session.output_audio.delta"
EVT_INPUT_TRANSCRIPT_DELTA = "session.input_transcript.delta"
EVT_OUTPUT_TRANSCRIPT_DELTA = "session.output_transcript.delta"
EVT_DELEGATION_CREATED = "session.delegation.created"
EVT_RESPONSE_EVENT = "response.event"
EVT_ERROR = "error"

# Bulk-data events kept out of the protocol-level debug log.
NOISY_EVENTS = frozenset(
    {EVT_OUTPUT_AUDIO_DELTA, EVT_INPUT_TRANSCRIPT_DELTA, EVT_OUTPUT_TRANSCRIPT_DELTA}
)

# Delegation targets on the wire → the RFC §12.4.1 vocabulary.
DELEGATION_TARGETS: dict[str, str] = {"responses": "hosted", "client": "integrator"}

#: Startup ``input`` history accepts at most this many text messages.
MAX_INPUT_ITEMS = 128

#: The API bounds one context append at 500 tokens. Chunks are measured with
#: the model's own tokenizer when it is installed, else by UTF-8 bytes (see
#: ``tokenizer``), against this headroom below the limit.
MAX_APPEND_TOKENS = 450

#: Correlation key for a wrapped Responses event whose envelope names no delegation.
UNCORRELATED_DELEGATION = "uncorrelated"


# --- Turn grouping -----------------------------------------------------------


class TurnGrouper:
    """One speaker's in-progress turn, assembled from transcript fragments.

    The wire emits transcript deltas on frame boundaries and never declares a
    turn. A grouper opens a turn on the first delta after quiet and closes it
    once ``gap_s`` elapses without another — the synthesized boundary RFC
    §12.4.1 asks a full-duplex provider for. ``on_open`` runs when the turn
    opens; ``on_close`` receives the accumulated text when it closes. Both run
    under the grouper's lock, so one turn's close never lands among the next
    turn's start.
    """

    def __init__(
        self,
        gap_s: float,
        *,
        on_open: Callable[[], Awaitable[None]],
        on_close: Callable[[str], Awaitable[None]],
    ) -> None:
        self._gap_s = gap_s
        self._on_open = on_open
        self._on_close = on_close
        self.text = ""
        self.open = False
        self._timer: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def feed(self, delta: str) -> None:
        """Add a fragment, opening the turn if quiet, and rearm the gap timer."""
        async with self._lock:
            if not self.open:
                self.open = True
                self.text = ""
                await self._on_open()
            self.text += delta
            self._rearm_timer()

    def _rearm_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = asyncio.create_task(self._close_after_gap())

    async def _close_after_gap(self) -> None:
        await asyncio.sleep(self._gap_s)
        async with self._lock:
            # Running inside the turn's own timer: it must not cancel itself.
            self._timer = None
            await self._close_locked()

    async def close(self) -> None:
        """Close the turn now, if one is open, emitting its end."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        async with self._lock:
            await self._close_locked()

    def cancel(self) -> None:
        """Drop the turn without emitting anything (connection torn down)."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self.open = False
        self.text = ""

    async def _close_locked(self) -> None:
        if not self.open:
            return
        self.open = False
        text, self.text = self.text, ""
        await self._on_close(text)


# --- Payload shaping ---------------------------------------------------------


def format_backend_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project RoomKit tool dicts onto the Responses API tool shape.

    A function tool keeps ``name``/``description``/``parameters`` under
    ``type: function``; keys the caller uses elsewhere (``tags`` for Tool
    Search, ``strict`` which the Live session schema rejects) are dropped.
    Hosted tools carrying a non-function ``type`` pass through unchanged.
    """
    formatted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type", "function") != "function":
            formatted.append(dict(tool))
            continue
        shaped: dict[str, Any] = {"type": "function"}
        for key in ("name", "description"):
            if key in tool:
                shaped[key] = tool[key]
        shaped["parameters"] = declared_parameters(tool.get("parameters"))
        formatted.append(shaped)
    return formatted


def history_items(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render prior text messages as the startup ``input`` history.

    Only text crosses: ``system``/``developer`` become developer messages
    (the role the history accepts), ``user`` and ``assistant`` keep theirs.
    Anything else, or a message without text, is skipped. The most recent
    :data:`MAX_INPUT_ITEMS` are kept.
    """
    items: list[dict[str, Any]] = []
    for message in history:
        shape = _HISTORY_ROLES.get(str(message.get("role")))
        text = message.get("text") or message.get("content")
        if shape is None or not isinstance(text, str) or not text.strip():
            continue
        wire_role, content_type = shape
        items.append(_input_item(wire_role, content_type, text))
    return items[-MAX_INPUT_ITEMS:]


# Message role → (wire role, content type) for the startup history.
_HISTORY_ROLES: dict[str, tuple[str, str]] = {
    "system": ("developer", "input_text"),
    "developer": ("developer", "input_text"),
    "user": ("user", "input_text"),
    "assistant": ("assistant", "output_text"),
}


def _input_item(role: str, content_type: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": [{"type": content_type, "text": text}]}


def build_audio_format(rate: int, codec: str) -> tuple[dict[str, Any], str | None]:
    """Map the channel's output rate to the one wire format a session uses.

    The Live API takes ``audio/pcm`` at 16 or 24 kHz, or G.711 (``audio/pcmu``,
    ``audio/pcma``) at 8 kHz, and applies it to both directions. Returns the
    format object and the G.711 law to build a codec for, if any.
    """
    if rate in (16000, 24000):
        if codec not in ("pcm", ""):
            raise ValueError(f"GPT-Live {rate} Hz audio is PCM; codec={codec!r} is only for 8 kHz")
        return {"type": "audio/pcm", "rate": rate}, None
    if rate == 8000:
        if codec not in ("pcmu", "pcma"):
            raise ValueError(f"GPT-Live 8 kHz requires codec='pcmu' or 'pcma', got {codec!r}")
        return {"type": f"audio/{codec}", "rate": 8000}, ("mulaw" if codec == "pcmu" else "alaw")
    raise ValueError(f"GPT-Live accepts 16000 or 24000 Hz (PCM) or 8000 Hz (G.711), got {rate}")


# --- Append chunking ---------------------------------------------------------

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
