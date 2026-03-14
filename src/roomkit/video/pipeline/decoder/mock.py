"""Mock video decoder for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.video.pipeline.decoder.base import VideoDecoderProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame


class MockVideoDecoderProvider(VideoDecoderProvider):
    """Pass-through decoder for testing.

    Returns the frame unchanged, recording every call for assertions.

    Example::

        decoder = MockVideoDecoderProvider()
        result = decoder.decode(frame)
        assert result is frame
        assert decoder.call_count == 1
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.frames: list[VideoFrame] = []

    @property
    def name(self) -> str:
        return "MockVideoDecoder"

    def decode(self, frame: VideoFrame) -> VideoFrame | None:
        self.call_count += 1
        self.frames.append(frame)
        return frame

    def reset(self) -> None:
        self.call_count = 0
        self.frames.clear()
