"""VideoResizerProvider abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame


class VideoResizerProvider(ABC):
    """Resize video frames to target dimensions.

    Implementations may use PyAV, NumPy/Pillow, or other libraries
    to scale frames. Only raw (uncompressed) frames can be resized.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'pyav-resizer', 'pillow-resizer')."""
        ...

    @abstractmethod
    def resize(self, frame: VideoFrame) -> VideoFrame:
        """Resize a raw video frame.

        Args:
            frame: A raw video frame (rgb24, bgr24, yuv420p, etc.).

        Returns:
            A new VideoFrame with resized pixel data and updated
            width/height. If the frame already fits within target
            dimensions, it may be returned unchanged.
        """
        ...

    def close(self) -> None:  # noqa: B027
        """Release resources."""
