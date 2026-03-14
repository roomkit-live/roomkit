"""Mock avatar provider for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.video.avatar.base import AvatarProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame


class MockAvatarProvider(AvatarProvider):
    """Test avatar that generates solid-color frames from audio.

    Produces one frame per ``feed_audio`` call with a configurable
    color.  Tracks call counts and audio bytes for assertions.

    Args:
        fps: Output frame rate (default 30).
        color: RGB tuple for generated frames (default green).
        idle_color: RGB tuple for idle frames (default gray),
            or None to return None from ``get_idle_frame``.
    """

    def __init__(
        self,
        *,
        fps: int = 30,
        color: tuple[int, int, int] = (0, 200, 0),
        idle_color: tuple[int, int, int] | None = (128, 128, 128),
    ) -> None:
        self._fps = fps
        self._color = color
        self._idle_color = idle_color
        self._started = False
        self._width = 512
        self._height = 512
        self._reference_image: bytes | None = None

        # Tracking for test assertions
        self.feed_count = 0
        self.total_audio_bytes = 0
        self.flush_count = 0
        self.idle_count = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def is_started(self) -> bool:
        return self._started

    async def start(
        self,
        reference_image: bytes,
        *,
        width: int = 512,
        height: int = 512,
    ) -> None:
        self._reference_image = reference_image
        self._width = width
        self._height = height
        self._started = True

    def feed_audio(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
    ) -> list[VideoFrame]:
        if not self._started:
            return []
        self.feed_count += 1
        self.total_audio_bytes += len(pcm_data)
        return [self._make_frame(self._color)]

    def get_idle_frame(self) -> VideoFrame | None:
        if not self._started or self._idle_color is None:
            return None
        self.idle_count += 1
        return self._make_frame(self._idle_color)

    def flush(self) -> list[VideoFrame]:
        if not self._started:
            return []
        self.flush_count += 1
        return [self._make_frame(self._color)]

    async def stop(self) -> None:
        self._started = False

    def _make_frame(self, color: tuple[int, int, int]) -> VideoFrame:
        from roomkit.video.video_frame import VideoFrame

        w, h = self._width, self._height
        pixel = bytes(color)
        data = pixel * (w * h)
        return VideoFrame(
            data=data,
            codec="raw_rgb24",
            width=w,
            height=h,
            keyframe=True,
            sequence=self.feed_count,
        )
