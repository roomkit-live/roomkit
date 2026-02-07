"""Speaker diarization provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


@dataclass
class DiarizationResult:
    """Result from a diarization provider."""

    speaker_id: str
    """Identified speaker label (e.g. 'speaker_0')."""

    confidence: float
    """Confidence score (0.0 to 1.0)."""

    is_new_speaker: bool
    """True if this is the first time this speaker has been seen."""


class DiarizationProvider(ABC):
    """Abstract base class for speaker diarization providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'pyannote', 'resemblyzer')."""
        ...

    @abstractmethod
    def process(self, frame: AudioFrame) -> DiarizationResult | None:
        """Analyse an audio frame for speaker identity.

        Args:
            frame: The audio frame to analyse.

        Returns:
            A DiarizationResult if a speaker was identified, else None.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset internal state."""

    def close(self) -> None:  # noqa: B027
        """Release resources."""
