"""Wire vocabulary and helpers for the OpenAI Live API (GPT-Live).

The Live API is a full-duplex speech-to-speech protocol: one ``session.start``,
then audio and transcript deltas in both directions, delegations the model
hands to a backend, and context appends the client returns. This module holds
what the provider needs to speak it without a session of its own: the event
names, the turn grouper that synthesizes the response and speech boundaries
the wire lacks (RFC §12.4.1), the append chunker that honours the per-append
token bound, and the bookkeeping of a hosted delegation's function calls.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

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

#: The API bounds one context append at 500 tokens. Chunks are measured
#: against a lower budget because what counts them is an estimate.
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


# --- Hosted delegation bookkeeping -----------------------------------------


@dataclass
class PendingResponse:
    """A hosted delegation's Responses run whose function calls are being collected.

    The hosted service rejects ``response.create`` while any call of the run
    is unanswered, so the run resumes only once ``finished`` is set and
    ``call_ids`` is empty — and only if it asked for a call at all.
    """

    call_ids: set[str] = field(default_factory=set)
    had_calls: bool = False
    finished: bool = False

    @property
    def ready_to_continue(self) -> bool:
        return self.finished and self.had_calls and not self.call_ids


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
        for key in ("name", "description", "parameters"):
            if key in tool:
                shaped[key] = tool[key]
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


def _token_quarters(text: str) -> int:
    """A text's cost in quarter-tokens: one per ASCII character, four otherwise.

    A byte-pair encoding packs about four ASCII characters into a token;
    other scripts cost more per character (CJK runs near one token each), so
    anything outside ASCII is counted as a whole token.
    """
    return sum(1 if char.isascii() else 4 for char in text)


def estimated_tokens(text: str) -> int:
    """Estimate how many tokens a text costs, rounding up."""
    return (_token_quarters(text) + 3) // 4


def _split_at_token_limit(text: str, token_limit: int) -> tuple[str, str]:
    """Split off the longest prefix within ``token_limit``, on a space if one exists."""
    budget = token_limit * 4
    end = len(text)
    cost = 0
    for index, char in enumerate(text):
        cost += _token_quarters(char)
        if cost > budget:
            end = max(index, 1)
            break
    space = text.rfind(" ", 0, end)
    if space > 0:
        end = space
    return text[:end].strip(), text[end:].strip()


def chunk_text(text: str, token_limit: int = MAX_APPEND_TOKENS) -> list[str]:
    """Split text into appends of at most ``token_limit`` estimated tokens.

    Chunks break on sentence boundaries; a sentence longer than the limit is
    split at a space. RFC §12.4.1: a bounded append splits rather than
    truncates or refuses.
    """
    text = text.strip()
    if not text:
        return []
    if estimated_tokens(text) <= token_limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for piece in _SENTENCE_BOUNDARY.split(text):
        piece = piece.strip()
        while estimated_tokens(piece) > token_limit:
            head, piece = _split_at_token_limit(piece, token_limit)
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if not piece:
            continue
        if current and estimated_tokens(f"{current} {piece}") > token_limit:
            chunks.append(current)
            current = ""
        current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks
