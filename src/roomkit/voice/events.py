"""Voice event types for enhanced voice support (RFC §19)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from roomkit.voice.base import VoiceSession


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class BargeInEvent:
    """User started speaking while TTS was playing.

    This event is fired when the VAD detects speech starting while
    audio is being sent to the user. This allows the system to:
    - Cancel the current TTS playback
    - Adjust response strategy (e.g., acknowledge interruption)
    - Track conversation dynamics
    """

    session: VoiceSession
    """The voice session where barge-in occurred."""

    interrupted_text: str
    """The text that was being spoken when interrupted."""

    audio_position_ms: int
    """How far into the TTS audio playback (in milliseconds)."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the barge-in was detected."""


@dataclass(frozen=True)
class TTSCancelledEvent:
    """TTS playback was cancelled.

    This event is fired when TTS synthesis or playback is stopped
    before completion. Reasons include:
    - barge_in: User started speaking
    - explicit: Application called interrupt()
    - disconnect: Session ended
    - error: TTS or playback error
    """

    session: VoiceSession
    """The voice session where TTS was cancelled."""

    reason: Literal["barge_in", "explicit", "disconnect", "error"]
    """Why the TTS was cancelled."""

    text: str
    """The text that was being synthesized."""

    audio_position_ms: int
    """How far into playback (0 if not started)."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the cancellation occurred."""


@dataclass(frozen=True)
class PartialTranscriptionEvent:
    """Interim transcription result during speech.

    This event is fired by backends that support streaming STT,
    providing real-time transcription updates before the final
    result. Use cases include:
    - Live captions/subtitles
    - Early intent detection
    - Visual feedback during speech
    """

    session: VoiceSession
    """The voice session being transcribed."""

    text: str
    """The current transcription (may change in subsequent events)."""

    confidence: float
    """Confidence score (0.0 to 1.0)."""

    is_stable: bool
    """True if this portion is unlikely to change significantly."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When this transcription was received."""


@dataclass(frozen=True)
class VADSilenceEvent:
    """Silence detected after speech.

    This event is fired when the VAD detects a period of silence
    following speech. It can be used for:
    - Early end-of-utterance detection (before full speech_end)
    - Adaptive silence thresholds
    - Turn-taking management
    """

    session: VoiceSession
    """The voice session where silence was detected."""

    silence_duration_ms: int
    """Duration of silence in milliseconds."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the silence was detected."""


@dataclass(frozen=True)
class VADAudioLevelEvent:
    """Periodic audio level update for UI feedback.

    This event is fired periodically (typically 10Hz) to provide
    audio level information for UI visualization. Use cases include:
    - Audio level meters
    - Speaking indicators
    - Noise detection
    """

    session: VoiceSession
    """The voice session."""

    level_db: float
    """Audio level in dB (typically -60 to 0, where 0 is max)."""

    is_speech: bool
    """VAD's determination if this audio contains speech."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When this measurement was taken."""
