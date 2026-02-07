"""Audio denoiser provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


class DenoiserProvider(ABC):
    """Abstract base class for audio denoising providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'rnnoise', 'deepfilter')."""
        ...

    @abstractmethod
    def process(self, frame: AudioFrame) -> AudioFrame:
        """Denoise an audio frame.

        Args:
            frame: The noisy audio frame.

        Returns:
            A new or modified AudioFrame with reduced noise.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset internal state."""

    def close(self) -> None:  # noqa: B027
        """Release resources."""
