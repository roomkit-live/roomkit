"""Audio recorder ABC and related data types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.base import VoiceSession


@unique
class RecordingMode(StrEnum):
    """When recording is active."""

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


@dataclass
class RecordingConfig:
    """Configuration for audio recording."""

    mode: RecordingMode = RecordingMode.ALWAYS
    """When to record."""

    channel_mode: RecordingChannelMode = RecordingChannelMode.MIXED
    """How to mix audio channels."""

    output_format: str = "wav"
    """Output file format."""

    output_dir: str = ""
    """Directory for recording files (empty = system temp)."""

    extra: dict[str, object] = field(default_factory=dict)
    """Provider-specific configuration."""


@dataclass
class RecordingHandle:
    """Handle to an active recording."""

    recording_id: str
    """Unique identifier for this recording."""

    session_id: str
    """The voice session being recorded."""

    path: str = ""
    """File path where the recording is being written."""


@dataclass
class RecordingResult:
    """Result returned when a recording is stopped."""

    recording_id: str
    """Unique identifier for this recording."""

    path: str
    """File path of the completed recording."""

    duration_ms: float
    """Total duration of the recording in milliseconds."""

    size_bytes: int = 0
    """File size in bytes."""


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
