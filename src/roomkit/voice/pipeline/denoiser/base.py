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
    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        """Denoise an audio frame.

        Args:
            frame: The noisy audio frame.
            stream: Identity of the audio stream this frame belongs to. A
                provider keeps its state per stream: a voice session and a
                conference track are separate speakers, and letting one advance
                the other's detection state makes silence from one close the
                other's utterance.

        Returns:
            A new or modified AudioFrame with reduced noise.
        """
        ...

    def reset(self, stream: str) -> None:  # noqa: B027
        """Drop a stream's state.

        Called when the stream ends, so a long-running room does not accumulate
        the state of every speaker that ever joined.
        """

    def close(self) -> None:  # noqa: B027
        """Release resources."""
