"""Voice Activity Detection provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


@unique
class VADEventType(StrEnum):
    """Types of VAD events."""

    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    SILENCE = "silence"
    AUDIO_LEVEL = "audio_level"


@dataclass
class VADEvent:
    """Event produced by a VAD provider."""

    type: VADEventType
    """The type of VAD event."""

    audio_bytes: bytes | None = None
    """Accumulated speech audio (set on SPEECH_END)."""

    confidence: float | None = None
    """Confidence score (0.0 to 1.0)."""

    duration_ms: float | None = None
    """Duration in milliseconds (speech or silence)."""

    level_db: float | None = None
    """Audio level in dB (set on AUDIO_LEVEL)."""


@dataclass
class VADConfig:
    """Configuration for VAD processing."""

    silence_threshold_ms: int = 500
    """Milliseconds of silence before triggering SPEECH_END."""

    speech_pad_ms: int = 300
    """Padding added around detected speech segments."""

    min_speech_duration_ms: int = 250
    """Minimum speech duration to trigger events."""

    extra: dict[str, object] = field(default_factory=dict)
    """Provider-specific configuration."""


class VADProvider(ABC):
    """Abstract base class for Voice Activity Detection providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'silero', 'webrtc')."""
        ...

    @abstractmethod
    def process(self, frame: AudioFrame) -> VADEvent | None:
        """Process an audio frame and optionally return a VAD event.

        Args:
            frame: The audio frame to analyse.

        Returns:
            A VADEvent if a state transition occurred, else None.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset internal state (e.g. between utterances)."""

    def close(self) -> None:  # noqa: B027
        """Release resources."""
