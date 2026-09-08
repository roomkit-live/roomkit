"""VideoDecoderProvider abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame


class VideoDecoderProvider(ABC):
    """Decode encoded video frames to raw pixels.

    Implementations handle specific codecs (H.264, VP8, VP9) and
    produce raw pixel data (RGB24, YUV420P, etc.) suitable for
    downstream pipeline stages such as resizing or vision analysis.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'pyav', 'ffmpeg')."""
        ...

    @abstractmethod
    def decode(self, frame: VideoFrame) -> VideoFrame | None:
        """Decode an encoded frame to raw pixels.

        Args:
            frame: Encoded video frame (H.264, VP8, VP9, etc.).

        Returns:
            A new VideoFrame with raw pixel data and updated codec/dimensions,
            or None if the frame cannot be decoded (e.g., missing keyframe).
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset decoder state (e.g., after seek or packet loss)."""

    def decode_for_session(self, session_id: str, frame: VideoFrame) -> VideoFrame | None:
        """Decode a session's frame. Stateful multi-session decoders override this.

        The default preserves the existing decode contract for stateless or
        single-session providers.
        """
        return self.decode(frame)

    def reset_session(self, session_id: str) -> None:  # noqa: B027
        """Release only this session's state, preserving other active sessions.

        Legacy providers receive a global reset when the final session ends.
        """

    def close(self) -> None:  # noqa: B027
        """Release resources."""
