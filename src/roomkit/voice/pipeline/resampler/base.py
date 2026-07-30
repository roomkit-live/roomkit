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

    The resampler is stage 1 of the inbound pipeline, so it is bound by the
    same stream identity contract as every other stage (RFC 12.3): one
    provider instance serves many streams, and any state it holds between
    frames MUST be kept under the stream key it was given. A resampler that
    buffers a frame for look-ahead and keys that buffer on format alone hands
    one speaker's audio to the next stream that asks — which is the whole
    reason the key is threaded down here rather than stopping at the VAD.
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
        stream: str,
    ) -> AudioFrame:
        """Resample an audio frame to the target format.

        Returns the original frame unchanged when the format already matches.

        Args:
            frame: The audio frame to resample.
            target_rate: Target sample rate in Hz.
            target_channels: Target number of channels.
            target_width: Target bytes per sample.
            stream: Identity of the audio stream this frame belongs to. A
                stateless resampler ignores it; one that buffers across frames
                keeps that buffer per stream, because a voice session and a
                conference lane are different speakers sharing one instance.

        Returns:
            A new or modified AudioFrame in the target format.
        """
        ...

    def flush(
        self,
        target_rate: int,
        target_channels: int,
        target_width: int,
        stream: str,
    ) -> AudioFrame | None:  # noqa: B027
        """Flush audio buffered for one stream after its end-of-stream.

        Subclasses that hold a pending frame (e.g. for look-ahead context)
        should override this to emit that frame using silence as look-ahead.

        Returns ``None`` when that stream has nothing to flush.
        """
        return None

    def reset(self, stream: str | None = None) -> None:  # noqa: B027
        """Drop buffered state — one stream's, or every stream's.

        ``None`` means all of them: it is what a blanket pipeline reset asks
        for. A stream key drops just that speaker's buffer, which is what the
        end of a session or a conference lane asks for.
        """

    def close(self) -> None:  # noqa: B027
        """Release resources."""
