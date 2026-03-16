"""AvatarProvider ABC — generate lip-synced video frames from audio.

An avatar takes TTS audio chunks and produces video frames with
realistic lip movements synchronized to the speech.  This is the
video equivalent of TTS for audio — it makes the AI "visible".

Two delivery modes are supported:

**Synchronous** (local inference — MuseTalk, WebSocket):
    ``feed_audio(pcm)`` returns ``list[VideoFrame]`` immediately.

**Asynchronous** (cloud inference — Anam):
    ``feed_audio(pcm)`` returns ``[]``, video frames arrive later
    via ``on_video()`` callbacks.  ``is_async`` property returns True.

Pipeline integration::

    AI text → TTS → audio chunks ──┬── VoiceBackend (send audio)
                                   └── AvatarProvider (audio → video)
                                            │
                                       VideoBackend (send video)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame


class AvatarProvider(ABC):
    """Generate lip-synced video frames from audio input.

    Lifecycle:
        1. ``start(reference_image)`` — load model / connect to cloud
        2. ``feed_audio(pcm)`` → video frames (sync) or ``[]`` (async)
        3. ``end_turn()`` — signal end of TTS response
        4. ``get_idle_frame()`` → idle animation when not speaking
        5. ``flush()`` → remaining frames after speech ends
        6. ``stop()`` — release resources / disconnect
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

    @property
    def is_async(self) -> bool:
        """Whether this provider delivers frames via callbacks.

        If True, ``feed_audio()`` returns ``[]`` and frames arrive
        asynchronously via ``on_video()`` callbacks.  The channel
        should wire the callback instead of processing the return value.

        Default: False (synchronous delivery via return value).
        """
        return False

    @abstractmethod
    async def start(
        self,
        reference_image: bytes,
        *,
        width: int = 512,
        height: int = 512,
    ) -> None:
        """Initialize the avatar provider.

        For local providers: load model and preprocess face from the
        reference image.  For cloud providers: connect to the service
        (reference image may be ignored if the service uses a pre-configured
        avatar model).

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
        """Feed an audio chunk for lip-sync processing.

        **Synchronous providers** return video frames immediately.
        **Async providers** return ``[]`` — frames arrive later via
        ``on_video()`` callbacks.

        Args:
            pcm_data: Raw PCM-16 LE audio bytes.
            sample_rate: Audio sample rate in Hz.

        Returns:
            List of ``VideoFrame`` (sync) or empty list (async).
        """

    def end_turn(self) -> None:  # noqa: B027
        """Signal end of a TTS/response turn.

        Cloud providers (Anam) require this to stop the avatar from
        freezing while waiting for more audio.  Local providers can
        ignore this (default is a no-op).
        """

    def on_video(self, callback: Callable[[VideoFrame], Any]) -> None:  # noqa: B027
        """Register a callback for async video frame delivery.

        Only used by async providers (``is_async == True``).  The callback
        is called with each ``VideoFrame`` as it becomes available.

        Args:
            callback: Called with a ``VideoFrame`` for each avatar frame.
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
        neutral expression).  Async providers should call ``end_turn()``
        internally and return ``[]``.
        """
        return []

    @abstractmethod
    async def stop(self) -> None:
        """Release model/GPU resources or disconnect from cloud service."""

    async def close(self) -> None:
        """Alias for ``stop()``."""
        await self.stop()
