"""Mock video resizer for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.video.pipeline.resizer.base import VideoResizerProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame


class MockVideoResizerProvider(VideoResizerProvider):
    """Pass-through resizer for testing.

    Returns the frame unchanged, recording every call for assertions.

    Example::

        resizer = MockVideoResizerProvider()
        result = resizer.resize(frame)
        assert result is frame
        assert resizer.call_count == 1
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.frames: list[VideoFrame] = []

    @property
    def name(self) -> str:
        return "MockVideoResizer"

    def resize(self, frame: VideoFrame) -> VideoFrame:
        self.call_count += 1
        self.frames.append(frame)
        return frame

    def reset(self) -> None:
        self.call_count = 0
        self.frames.clear()

    def close(self) -> None:
        pass
