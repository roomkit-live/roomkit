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
    def process(self, frame: AudioFrame, stream: str) -> DiarizationResult | None:
        """Analyse an audio frame for speaker identity.

        Args:
            frame: The audio frame to analyse.
            stream: Identity of the audio stream this frame belongs to. A
                provider keeps its state per stream: a voice session and a
                conference track are separate speakers, and letting one advance
                the other's detection state makes silence from one close the
                other's utterance.

        Returns:
            A DiarizationResult if a speaker was identified, else None.
        """
        ...

    def reset(self, stream: str) -> None:  # noqa: B027
        """Drop a stream's state.

        Called when the stream ends, so a long-running room does not accumulate
        the state of every speaker that ever joined.
        """

    def clear_speakers(self) -> None:  # noqa: B027
        """Forget every enrolled speaker.

        Unlike :meth:`reset` (which clears transient clustering state), this
        drops the enrollment set so a provider reused across sessions does not
        carry speakers from a previous conversation into the next one.
        Providers with no enrollment concept may leave this as a no-op.
        """

    def close(self) -> None:  # noqa: B027
        """Release resources."""
