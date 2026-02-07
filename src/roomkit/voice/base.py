"""Base models for voice support."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Flag, StrEnum, auto, unique
from typing import Any


@unique
class VoiceSessionState(StrEnum):
    """State of a voice session."""

    CONNECTING = "connecting"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class VoiceCapability(Flag):
    """Capabilities a VoiceBackend can support.

    Backends declare their capabilities via the `capabilities` property.
    This allows RoomKit to know which features are available and
    enables integrators to choose backends based on their needs.

    Example:
        class MyBackend(VoiceBackend):
            @property
            def capabilities(self) -> VoiceCapability:
                return (
                    VoiceCapability.INTERRUPTION |
                    VoiceCapability.BARGE_IN
                )
    """

    NONE = 0
    """No optional capabilities (default)."""

    INTERRUPTION = auto()
    """Backend can cancel ongoing audio playback (cancel_audio)."""

    BARGE_IN = auto()
    """Backend detects and handles barge-in (user interrupts TTS)."""

    NATIVE_AEC = auto()
    """Backend provides its own Acoustic Echo Cancellation."""

    NATIVE_AGC = auto()
    """Backend provides its own Automatic Gain Control."""

    DTMF_INBAND = auto()
    """Backend can detect DTMF tones from the audio stream."""

    DTMF_SIGNALING = auto()
    """Backend receives DTMF via out-of-band signaling (e.g. SIP INFO)."""


@dataclass
class AudioChunk:
    """A chunk of audio data for streaming (used for outbound TTS)."""

    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    format: str = "pcm_s16le"
    timestamp_ms: int | None = None
    is_final: bool = False


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


@dataclass
class VoiceSession:
    """Active voice connection for a participant."""

    id: str
    room_id: str
    participant_id: str
    channel_id: str
    state: VoiceSessionState = VoiceSessionState.CONNECTING
    created_at: datetime = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""

    text: str
    is_final: bool = True
    confidence: float | None = None
    language: str | None = None
    words: list[dict[str, Any]] = field(default_factory=list)


# Type aliases for voice callbacks
BargeInCallback = Callable[[VoiceSession], Any]
"""Callback for barge-in detection: (session)."""
