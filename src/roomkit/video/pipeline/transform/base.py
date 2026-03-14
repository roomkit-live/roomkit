"""VideoTransformProvider ABC — pixel-level frame transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame


class VideoTransformProvider(ABC):
    """Transform video frame pixels in the pipeline.

    Transforms run on **every** raw frame between resizer and filters.
    They modify pixel data (e.g., grayscale, sepia, blur) without
    inspecting vision results — that is the filter's job.

    Implementations must be fast (no I/O, no async).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable transform name."""

    @abstractmethod
    def transform(self, frame: VideoFrame) -> VideoFrame:
        """Transform frame pixels. Returns a new frame with modified data.

        Args:
            frame: The current video frame (raw pixels after decode/resize).

        Returns:
            A new ``VideoFrame`` with transformed pixel data, or the
            original frame unchanged if the codec is not supported.
        """

    def reset(self) -> None:  # noqa: B027
        """Reset internal state."""

    def close(self) -> None:  # noqa: B027
        """Release resources."""
