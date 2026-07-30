"""Automatic Gain Control provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


@dataclass
class AGCConfig:
    """Configuration for Automatic Gain Control."""

    target_level_dbfs: float = -3.0
    """Target output level in dBFS."""

    max_gain_db: float = 30.0
    """Maximum gain applied in dB."""

    attack_ms: float = 10.0
    """Attack time in milliseconds (how quickly gain increases)."""

    release_ms: float = 100.0
    """Release time in milliseconds (how quickly gain decreases)."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Provider-specific configuration."""


class AGCProvider(ABC):
    """Abstract base class for Automatic Gain Control providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'webrtc_agc')."""
        ...

    @abstractmethod
    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        """Apply gain control to an audio frame.

        Args:
            frame: The audio frame to normalise.
            stream: Identity of the audio stream this frame belongs to. A
                provider keeps its state per stream: a voice session and a
                conference track are separate speakers, and letting one advance
                the other's detection state makes silence from one close the
                other's utterance.

        Returns:
            A new or modified AudioFrame with gain applied.
        """
        ...

    def reset(self, stream: str) -> None:  # noqa: B027
        """Drop a stream's state.

        Called when the stream ends, so a long-running room does not accumulate
        the state of every speaker that ever joined.
        """

    def close(self) -> None:  # noqa: B027
        """Release resources."""
