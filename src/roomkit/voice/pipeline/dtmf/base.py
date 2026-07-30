"""DTMF tone detector ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


@dataclass
class DTMFEvent:
    """A detected DTMF tone."""

    digit: str
    """The DTMF digit ('0'-'9', '*', '#', 'A'-'D')."""

    duration_ms: float
    """Duration of the tone in milliseconds."""

    confidence: float = 1.0
    """Detection confidence (0.0 to 1.0)."""


class DTMFDetector(ABC):
    """Abstract base class for DTMF tone detection providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'goertzel')."""
        ...

    @abstractmethod
    def process(self, frame: AudioFrame, stream: str) -> DTMFEvent | None:
        """Analyse an audio frame for DTMF tones.

        Args:
            frame: The audio frame to analyse.
            stream: Identity of the audio stream this frame belongs to. A
                provider keeps its state per stream: a voice session and a
                conference track are separate speakers, and letting one advance
                the other's detection state makes silence from one close the
                other's utterance.

        Returns:
            A DTMFEvent if a tone was detected, else None.
        """
        ...

    def reset(self, stream: str) -> None:  # noqa: B027
        """Drop a stream's state.

        Called when the stream ends, so a long-running room does not accumulate
        the state of every speaker that ever joined.
        """

    def close(self) -> None:  # noqa: B027
        """Release resources."""
