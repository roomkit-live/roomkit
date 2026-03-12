"""Mock vision provider for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.video.vision.base import VisionProvider, VisionResult

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame


class MockVisionProvider(VisionProvider):
    """Mock vision provider for testing.

    Returns pre-configured descriptions in round-robin order and
    records every frame submitted for analysis.

    Example::

        provider = MockVisionProvider(
            descriptions=["A person waving", "An empty room"],
        )
        result = await provider.analyze_frame(frame)
        assert result.description == "A person waving"
        assert len(provider.calls) == 1
    """

    def __init__(
        self,
        descriptions: list[str] | None = None,
        *,
        labels: list[list[str]] | None = None,
    ) -> None:
        self.descriptions = descriptions or [
            "A video frame",
            "A person in a room",
        ]
        self.labels = labels or [["person"], ["room"]]
        self.calls: list[VideoFrame] = []
        self._index = 0

    async def analyze_frame(self, frame: VideoFrame) -> VisionResult:
        self.calls.append(frame)
        desc = self.descriptions[self._index % len(self.descriptions)]
        frame_labels = self.labels[self._index % len(self.labels)]
        self._index += 1
        return VisionResult(
            description=desc,
            labels=frame_labels,
        )
