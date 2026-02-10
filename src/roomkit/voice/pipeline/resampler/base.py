"""Resampler provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


class ResamplerProvider(ABC):
    """Abstract base class for audio resampling providers.

    The ``resample()`` method accepts target format parameters rather than
    fixing them at construction time because the pipeline calls it in two
    directions: inbound (transport -> internal) and outbound (internal ->
    transport) with different targets.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'linear', 'libsamplerate')."""
        ...

    @abstractmethod
    def resample(
        self,
        frame: AudioFrame,
        target_rate: int,
        target_channels: int,
        target_width: int,
    ) -> AudioFrame:
        """Resample an audio frame to the target format.

        Returns the original frame unchanged when the format already matches.

        Args:
            frame: The audio frame to resample.
            target_rate: Target sample rate in Hz.
            target_channels: Target number of channels.
            target_width: Target bytes per sample.

        Returns:
            A new or modified AudioFrame in the target format.
        """
        ...

    def flush(
        self,
        target_rate: int,
        target_channels: int,
        target_width: int,
    ) -> AudioFrame | None:  # noqa: B027
        """Flush any buffered audio remaining after end-of-stream.

        Subclasses that hold a pending frame (e.g. for look-ahead context)
        should override this to emit that frame using silence as look-ahead.

        Returns ``None`` when there is nothing to flush.
        """
        return None

    def reset(self) -> None:  # noqa: B027
        """Reset internal state."""

    def close(self) -> None:  # noqa: B027
        """Release resources."""
