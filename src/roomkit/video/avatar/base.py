"""AvatarProvider ABC — generate lip-synced video frames from audio.

An avatar takes a static reference image (portrait photo) and TTS
audio chunks, producing video frames with realistic lip movements
synchronized to the speech.  This is the video equivalent of TTS
for audio — it makes the AI "visible" to the user.

Pipeline integration::

    AI text → TTS → audio chunks ──┬── VoiceBackend (send audio)
                                   └── AvatarProvider (audio → video)
                                            │
                                       VideoBackend (send video)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame


class AvatarProvider(ABC):
    """Generate lip-synced video frames from audio input.

    Lifecycle:
        1. ``start(reference_image)`` — load model, preprocess face
        2. ``feed_audio(pcm)`` → video frames (called per TTS chunk)
        3. ``get_idle_frame()`` → idle animation when not speaking
        4. ``flush()`` → remaining frames after speech ends
        5. ``stop()`` — release GPU/resources
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    @property
    @abstractmethod
    def fps(self) -> int:
        """Output video frame rate."""

    @property
    def is_started(self) -> bool:
        """Whether the provider has been started with a reference image."""
        return False

    @abstractmethod
    async def start(
        self,
        reference_image: bytes,
        *,
        width: int = 512,
        height: int = 512,
    ) -> None:
        """Initialize with a portrait image.

        Preprocesses the face (landmarks, base latent) for real-time
        synthesis.  Must be called before ``feed_audio``.

        Args:
            reference_image: PNG or JPEG portrait image bytes.
            width: Output video frame width.
            height: Output video frame height.
        """

    @abstractmethod
    def feed_audio(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
    ) -> list[VideoFrame]:
        """Feed an audio chunk and get back lip-synced video frames.

        May return 0 frames (buffering audio) or multiple frames
        (catching up).  Audio-to-video timing: at 16kHz audio and
        30fps video, each 20ms audio chunk (320 samples) produces
        roughly 0.6 frames — so expect ~2 frames per 3 audio chunks.

        Args:
            pcm_data: Raw PCM-16 LE audio bytes.
            sample_rate: Audio sample rate in Hz.

        Returns:
            List of ``VideoFrame`` with ``codec="raw_rgb24"``.
        """

    def get_idle_frame(self) -> VideoFrame | None:
        """Return an idle animation frame (blinking, slight movement).

        Called at ``fps`` rate when no audio is being processed.
        The default implementation returns ``None`` (static image —
        the caller should use the last frame or the reference image).
        """
        return None

    def flush(self) -> list[VideoFrame]:
        """Flush remaining frames after speech ends.

        Returns closing animation frames (mouth closing, return to
        neutral expression).  Default returns empty list.
        """
        return []

    @abstractmethod
    async def stop(self) -> None:
        """Release model and GPU resources."""

    async def close(self) -> None:
        """Alias for ``stop()``."""
        await self.stop()
