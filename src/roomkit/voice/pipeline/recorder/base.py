"""Audio recorder ABC and related data types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, unique
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.base import VoiceSession


@unique
class RecordingMode(StrEnum):
    """Which audio directions to record."""

    INBOUND_ONLY = "inbound_only"
    """Record only inbound (microphone) audio."""

    OUTBOUND_ONLY = "outbound_only"
    """Record only outbound (TTS/speaker) audio."""

    BOTH = "both"
    """Record both inbound and outbound audio."""

    @classmethod
    def _missing_(cls, value: object) -> RecordingMode | None:
        """Backwards-compat: map legacy ``"always"``/``"speech_only"`` to ``BOTH``."""
        if isinstance(value, str):
            lower = value.lower()
            if lower in ("always", "speech_only"):
                return cls.BOTH
        return None


@unique
class RecordingTrigger(StrEnum):
    """When recording is temporally active."""

    ALWAYS = "always"
    """Record for the entire session."""

    SPEECH_ONLY = "speech_only"
    """Record only during detected speech segments."""


@unique
class RecordingChannelMode(StrEnum):
    """How audio channels are recorded."""

    MIXED = "mixed"
    """Mix inbound and outbound into a single channel."""

    SEPARATE = "separate"
    """Record inbound and outbound as separate channels."""

    STEREO = "stereo"
    """Record inbound and outbound as left/right stereo channels."""


@dataclass
class RecordingConfig:
    """Configuration for audio recording."""

    mode: RecordingMode = RecordingMode.BOTH
    """Which audio directions to record."""

    trigger: RecordingTrigger = RecordingTrigger.ALWAYS
    """When to temporally activate recording."""

    channels: RecordingChannelMode = RecordingChannelMode.MIXED
    """How to mix audio channels."""

    format: str = "wav"
    """Output file format."""

    storage: str = ""
    """Integrator-defined storage identifier, resolved at runtime."""

    retention_days: int | None = None
    """Optional retention period in days (None = indefinite)."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Provider-specific configuration."""


@dataclass
class RecordingHandle:
    """Handle to an active recording."""

    id: str
    """Unique identifier for this recording."""

    session_id: str
    """The voice session being recorded."""

    state: str = "recording"
    """Current state: ``"recording"`` or ``"stopped"``."""

    started_at: datetime | None = None
    """When the recording started (UTC)."""

    path: str = ""
    """File path where the recording is being written."""


@dataclass
class RecordingResult:
    """Result returned when a recording is stopped."""

    id: str
    """Unique identifier for this recording."""

    urls: list[str] = field(default_factory=list)
    """URLs or paths to the completed recording file(s)."""

    duration_seconds: float = 0.0
    """Total duration of the recording in seconds."""

    format: str = "wav"
    """File format of the recording."""

    mode: RecordingChannelMode = RecordingChannelMode.MIXED
    """Channel mode used for the recording."""

    size_bytes: int = 0
    """File size in bytes."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Provider-specific result metadata."""


class AudioRecorder(ABC):
    """Abstract base class for audio recording providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @abstractmethod
    def start(self, session: VoiceSession, config: RecordingConfig) -> RecordingHandle:
        """Start recording a session.

        Args:
            session: The voice session to record.
            config: Recording configuration.

        Returns:
            A handle to the active recording.
        """
        ...

    @abstractmethod
    def stop(self, handle: RecordingHandle) -> RecordingResult:
        """Stop an active recording.

        Args:
            handle: The recording handle from start().

        Returns:
            Result with file path and duration.
        """
        ...

    @abstractmethod
    def tap_inbound(self, handle: RecordingHandle, frame: AudioFrame) -> None:
        """Feed an inbound audio frame to the recorder.

        Args:
            handle: The active recording handle.
            frame: The inbound audio frame.
        """
        ...

    @abstractmethod
    def tap_outbound(self, handle: RecordingHandle, frame: AudioFrame) -> None:
        """Feed an outbound audio frame to the recorder.

        Args:
            handle: The active recording handle.
            frame: The outbound audio frame.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset internal state."""

    def close(self) -> None:  # noqa: B027
        """Release resources."""
