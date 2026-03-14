"""Mock video filter for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.video.pipeline.filter.base import FilterContext, VideoFilterProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame


class MockVideoFilterProvider(VideoFilterProvider):
    """Pass-through filter that tracks calls for testing."""

    @property
    def name(self) -> str:
        return "mock"

    def __init__(self) -> None:
        self.call_count = 0
        self.last_frame: VideoFrame | None = None
        self.last_context: FilterContext | None = None

    def filter(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        self.call_count += 1
        self.last_frame = frame
        self.last_context = context
        return frame
