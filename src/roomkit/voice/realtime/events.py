"""Realtime voice event types for hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from roomkit.voice.base import VoiceSession


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class RealtimeTranscriptionEvent:
    """Transcription produced by the realtime provider.

    Fired through ON_TRANSCRIPTION hooks. For final transcriptions,
    the channel emits a RoomEvent so other channels see the conversation.
    """

    session: VoiceSession
    """The realtime session that produced this transcription."""

    text: str
    """The transcribed text."""

    role: Literal["user", "assistant"]
    """Who spoke: 'user' (input) or 'assistant' (AI output)."""

    is_final: bool
    """True if this is a final transcription (not interim)."""

    item_id: str | None = None
    """Provider-specific item ID for correlation."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the transcription was received."""


@dataclass(frozen=True)
class RealtimeToolCallEvent:
    """The realtime provider is requesting a function call.

    Fired through ON_REALTIME_TOOL_CALL hooks (sync). The hook
    must return a result that gets submitted back to the provider.
    """

    session: VoiceSession
    """The realtime session requesting the tool call."""

    tool_call_id: str
    """Provider-assigned ID for this tool call."""

    name: str
    """The function name being called."""

    arguments: dict[str, Any]
    """Parsed arguments for the function call."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the tool call was received."""


@dataclass(frozen=True)
class RealtimeSpeechEvent:
    """Speech activity detected by the realtime provider's server-side VAD."""

    session: VoiceSession
    """The realtime session where speech activity changed."""

    type: Literal["start", "end"]
    """Whether speech started or ended."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the speech event was detected."""


@dataclass(frozen=True)
class RealtimeErrorEvent:
    """Error from the realtime provider."""

    session: VoiceSession
    """The realtime session that encountered the error."""

    code: str
    """Error code from the provider."""

    message: str
    """Human-readable error description."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the error occurred."""
