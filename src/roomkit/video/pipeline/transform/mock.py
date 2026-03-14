"""Mock video transform for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.video.pipeline.transform.base import VideoTransformProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame


class MockVideoTransformProvider(VideoTransformProvider):
    """Pass-through transform that tracks calls for testing."""

    @property
    def name(self) -> str:
        return "mock"

    def __init__(self) -> None:
        self.call_count = 0
        self.last_frame: VideoFrame | None = None

    def transform(self, frame: VideoFrame) -> VideoFrame:
        self.call_count += 1
        self.last_frame = frame
        return frame

    def reset(self) -> None:
        self.call_count = 0
        self.last_frame = None
