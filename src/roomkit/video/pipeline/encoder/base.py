"""VideoEncoderProvider ABC — encode raw video frames to codec bitstream."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame


class VideoEncoderProvider(ABC):
    """Encode raw video frames to a compressed codec bitstream.

    Used for outbound video: avatar raw frames → H.264 NAL units → RTP.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable encoder name."""

    @abstractmethod
    def encode(self, frame: VideoFrame) -> list[bytes]:
        """Encode a raw frame, return codec-specific data units.

        For H.264: returns individual NAL units (without start codes).
        For VP9: returns complete frame bitstream.

        May return empty list if the encoder is buffering.
        """

    def flush(self) -> list[bytes]:
        """Flush remaining buffered frames."""
        return []

    def reset(self) -> None:  # noqa: B027
        """Reset encoder state."""

    def close(self) -> None:  # noqa: B027
        """Release resources."""
