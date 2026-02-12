"""Voice event types for enhanced voice support (RFC §19)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.realtime.base import RealtimeSession


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


@dataclass(frozen=True)
class AudioLevelEvent:
    """Audio level update for input or output.

    Fired per audio frame from the pipeline (input) or backend (output),
    regardless of whether VAD is enabled.  Use for VU meters.
    """

    session: VoiceSession | RealtimeSession
    """The voice session (VoiceSession or RealtimeSession)."""

    level_db: float
    """Audio level in dB (typically -60 to 0, where 0 is max)."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When this measurement was taken."""


@dataclass(frozen=True)
class SpeakerChangeEvent:
    """Speaker change detected by diarization.

    This event is fired when the audio pipeline's diarization
    stage detects a different speaker than the previous frame.
    """

    session: VoiceSession
    """The voice session where the change was detected."""

    speaker_id: str
    """The new speaker's identifier."""

    confidence: float
    """Confidence score for the speaker identification (0.0 to 1.0)."""

    is_new_speaker: bool
    """True if this speaker has not been seen before in this session."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the speaker change was detected."""


@dataclass(frozen=True)
class DTMFDetectedEvent:
    """DTMF tone detected in the audio stream."""

    session: VoiceSession
    """The voice session where the DTMF was detected."""

    digit: str
    """The DTMF digit ('0'-'9', '*', '#', 'A'-'D')."""

    duration_ms: float
    """Duration of the tone in milliseconds."""

    confidence: float = 1.0
    """Detection confidence (0.0 to 1.0)."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the DTMF was detected."""


@dataclass(frozen=True)
class TurnCompleteEvent:
    """User's conversational turn is considered complete."""

    session: VoiceSession
    """The voice session."""

    text: str
    """Full accumulated text of the completed turn."""

    confidence: float = 1.0
    """Confidence in the turn completion decision."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the turn was determined complete."""


@dataclass(frozen=True)
class TurnIncompleteEvent:
    """User's conversational turn is not yet complete (accumulating)."""

    session: VoiceSession
    """The voice session."""

    text: str
    """Text accumulated so far in this potential turn."""

    confidence: float = 1.0
    """Confidence in the incomplete decision."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the decision was made."""


@dataclass(frozen=True)
class BackchannelEvent:
    """Backchannel utterance detected (e.g. 'uh-huh', 'yeah')."""

    session: VoiceSession
    """The voice session."""

    text: str
    """The backchannel text."""

    confidence: float = 1.0
    """Detection confidence (0.0 to 1.0)."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the backchannel was detected."""


@dataclass(frozen=True)
class RecordingStartedEvent:
    """Audio recording has started for a session."""

    session: VoiceSession
    """The voice session being recorded."""

    id: str
    """Unique identifier for this recording."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the recording started."""


@dataclass(frozen=True)
class RecordingStoppedEvent:
    """Audio recording has stopped for a session."""

    session: VoiceSession
    """The voice session that was being recorded."""

    id: str
    """Unique identifier for this recording."""

    urls: tuple[str, ...] = ()
    """URLs or paths to the completed recording file(s)."""

    duration_seconds: float = 0.0
    """Duration of the recording in seconds."""

    timestamp: datetime = field(default_factory=_utcnow)
    """When the recording stopped."""
