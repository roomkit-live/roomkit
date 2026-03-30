"""Mock face touch filter for testing without MediaPipe."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.video.pipeline.filter.base import FilterContext, FilterEvent, VideoFilterProvider

if TYPE_CHECKING:
    from roomkit.video.events import VideoDetectionEvent
    from roomkit.video.video_frame import VideoFrame


class MockFaceTouchFilter(VideoFilterProvider):
    """Mock filter that emits pre-configured detection events at specific frames.

    Useful for testing the detection → hook pipeline without requiring
    MediaPipe or a real camera feed.

    Args:
        events_at: Mapping of frame sequence numbers to lists of
            :class:`~roomkit.video.events.VideoDetectionEvent` instances
            to emit at that frame.

    Example::

        from roomkit.video.events import VideoDetectionEvent

        mock = MockFaceTouchFilter(events_at={
            5: [VideoDetectionEvent(
                session=None,
                kind="face_touch",
                labels=["left_cheek"],
                confidence=0.85,
                metadata={"zone": "left_cheek", "hand": "right"},
            )],
        })
    """

    def __init__(self, events_at: dict[int, list[VideoDetectionEvent]] | None = None) -> None:
        self._events_at: dict[int, list[VideoDetectionEvent]] = events_at or {}
        self._frame_count = 0

    @property
    def name(self) -> str:
        return "mock_face_touch"

    def filter(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        self._frame_count += 1
        seq = frame.sequence
        events = self._events_at.get(seq)
        if events:
            for event in events:
                context.events.append(FilterEvent(kind="face_touch", data=event))
        return frame

    def reset(self) -> None:
        self._frame_count = 0

    def close(self) -> None:
        pass
