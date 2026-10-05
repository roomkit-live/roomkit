"""Per-session state of the Gemini Live provider.

The dataclass every other ``realtime_*`` module reads and writes, the
transcription chunk buffer, and the exception that turns a server GoAway into
a reconnect. Nothing here talks to the network: the state is what the
connection, the handlers and the tool bookkeeping share, kept apart so each
of them can be read without the others.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from roomkit.voice.base import VoiceSession

__all__ = ["_GeminiSessionState", "_GoAwayError", "_TranscriptionBuffer"]


class _TranscriptionBuffer:
    """Accumulates Gemini transcription chunks until finished=True.

    Gemini sends transcription text in incremental chunks.  This buffer
    collects them per (session_id, role) and emits the concatenated
    result when ``flush`` is called or when ``append`` receives a
    ``finished=True`` chunk.
    """

    def __init__(self) -> None:
        self._buffers: dict[tuple[str, str], list[str]] = {}

    def append(self, session_id: str, role: str, text: str, finished: bool) -> str | None:
        """Add a chunk. Returns the full text if ``finished``, else None."""
        key = (session_id, role)
        self._buffers.setdefault(key, []).append(text)
        if finished:
            full = "".join(self._buffers.pop(key, []))
            return full if full.strip() else None
        return None

    def flush(self, session_id: str, role: str) -> str | None:
        """Flush the buffer for a (session, role) pair. Returns text or None."""
        chunks = self._buffers.pop((session_id, role), [])
        if chunks:
            full = "".join(chunks)
            return full if full.strip() else None
        return None

    def clear_session(self, session_id: str) -> None:
        """Remove all buffers for a session."""
        for key in [k for k in self._buffers if k[0] == session_id]:
            del self._buffers[key]


@dataclass
class _GeminiSessionState:
    """Consolidated per-session state for Gemini Live provider."""

    session: VoiceSession
    live_session: Any = None
    ctxmgr: Any = None
    malformed_call_nudged: bool = False
    """The model was told its function call could not be parsed, since the
    user last spoke: once until they speak again (RFC §12.4)."""
    live_config: Any = None
    receive_task: asyncio.Task[None] | None = None
    resumption_handle: str | None = None
    audio_chunk_count: int = 0
    response_started: bool = False
    user_speech_active: bool = False
    audio_buffer: deque[tuple[float, bytes]] = field(default_factory=lambda: deque(maxlen=100))
    error_suppressed: bool = False
    started_at: float = 0.0
    turn_count: int = 0
    tool_result_bytes: int = 0
    input_sample_rate: int = 16000
    # Tools this session declared BLOCKING. From 3.8 the model runs its calls
    # in the background by default, but a single tool can still ask to block
    # where the model allows it, so the mode is a property of the call and not
    # of the model: deriving it from the model's default let a blocking call
    # slip past the injection queue that exists precisely for it.
    blocking_tool_names: set[str] = field(default_factory=set)
    # The calls still open (on the provider's book, each with the function
    # it named) the API is waiting on. Non-empty means it refuses
    # client_content.
    blocking_call_ids: set[str] = field(default_factory=set)
    queued_injections: list[tuple[bytes, str, str, bool]] = field(default_factory=list)
    realtime_input_sent: bool = False
    queued_text_injections: list[tuple[str, str, bool]] = field(default_factory=list)
    # Last final transcription emitted per role — see _is_duplicate_final.
    last_final_text: dict[str, str] = field(default_factory=dict)
    # Gemini does not always emit VAD boundaries. turn_complete then marks
    # that the next input transcription belongs to a new user utterance.
    awaiting_new_user_utterance: bool = False
    # output_transcription can precede model_turn or share its server message.
    # Track that boundary independently from response_started so the duplicate
    # guard is reset exactly once, before the first assistant chunk is handled.
    assistant_response_observed: bool = False
    # Whether this session's server reports ``interaction_status``. From the
    # 3.8 generation the model may speak several times inside one request, so
    # ``turn_complete`` stops meaning "the model is done" and only ``IDLE``
    # does. Recorded from what the stream actually carries rather than from
    # the model id, so a preview this build never heard of is handled right.
    reports_interaction_status: bool = False
    # Setup fields already reported as dropped or downgraded for this
    # session. One line per session is what an operator can act on; kept on
    # the provider, it went out for the first call of the process and never
    # again.
    warned_unsupported: set[str] = field(default_factory=set)
    # The interruption already ended this response: the turn_complete (and
    # IDLE) that closes the interrupted request must not end it again.
    response_ended_by_interrupt: bool = False
    # Whether anything was said in this session yet: text sent, the user
    # transcribed, the model heard or seen calling a tool. A session with
    # nothing in it has no context to resume, and ``gemini-3.8-live`` resumes
    # a session under its original instruction, ignoring a new one.
    has_conversation: bool = False
    # The instruction a resumption did not apply: it rides the session's
    # next non-silent injection, then clears.
    pending_instructions: str | None = None
    # Effective config values, kept in sync across connect + reconfigure
    # so partial reconfigures (e.g. system_prompt-only) preserve the
    # other fields. Without these, ``_build_config`` (which treats
    # ``None`` as "absent") would silently wipe the unspecified fields
    # — most notably the tools list, leaving the model with no
    # functions to call after a skill activation.
    system_prompt: str | None = None
    voice: str | None = None
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    server_vad: bool = True
    provider_config: dict[str, Any] = field(default_factory=dict)

    def buffer_audio(self, audio: bytes) -> None:
        """Keep at most two seconds of recent mono PCM16 during a reconnect."""
        now = time.monotonic()
        limit = self.input_sample_rate * 2 * 2
        self.audio_buffer.append((now, audio[-limit:]))
        size = sum(len(chunk) for _, chunk in self.audio_buffer)
        while self.audio_buffer and (size > limit or now - self.audio_buffer[0][0] > 2.0):
            _, removed = self.audio_buffer.popleft()
            size -= len(removed)

    def pop_audio(self) -> bytes | None:
        """Consume without iterating across network awaits; discard expired speech."""
        now = time.monotonic()
        while self.audio_buffer:
            captured_at, audio = self.audio_buffer.popleft()
            if now - captured_at <= 2.0:
                return audio
        return None


class _GoAwayError(Exception):
    """Raised when the server sends a GoAway signal to trigger proactive reconnection."""
