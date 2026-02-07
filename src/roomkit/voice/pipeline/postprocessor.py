"""Audio postprocessor ABC (interface only, implementation deferred)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


class AudioPostProcessor(ABC):
    """Abstract base class for audio postprocessors.

    Postprocessors transform audio frames after the main pipeline
    stages. Implementation of concrete postprocessors is deferred.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Postprocessor name."""
        ...

    @abstractmethod
    def process(self, frame: AudioFrame) -> AudioFrame:
        """Process an audio frame.

        Args:
            frame: The audio frame to transform.

        Returns:
            A new or modified AudioFrame.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset internal state."""

    def close(self) -> None:  # noqa: B027
        """Release resources."""
