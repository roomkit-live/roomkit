"""Audio mixer provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


class MixerProvider(ABC):
    """Abstract base class for PCM audio mixing providers.

    A mixer combines multiple int16 PCM audio frames into a single frame.
    All input frames are expected to share the same sample rate, channels,
    and sample width (the bridge resamples before calling ``mix``).

    Implementations may use pure Python, NumPy, or native C/Rust libraries
    for performance.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. ``'python'``, ``'numpy'``)."""
        ...

    @abstractmethod
    def mix(self, frames: list[AudioFrame]) -> AudioFrame:
        """Mix multiple audio frames into one.

        Combines PCM samples from all input frames.  For 2 sources the
        samples are summed directly.  For 3+ sources, headroom scaling
        is applied to prevent clipping.  Results are clamped to the
        int16 range ``[-32768, 32767]``.

        If frames have different byte lengths, the result uses the
        length of the shortest frame.

        Args:
            frames: One or more audio frames to mix.  A single frame
                is returned unchanged.  Must share the same sample
                rate, channels, and sample width.

        Returns:
            A new AudioFrame containing the mixed audio.  Metadata
            (sample_rate, channels, sample_width, timestamp_ms) is
            taken from the first frame.
        """
        ...
