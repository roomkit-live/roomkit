"""PyAV-based video frame resizer."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from roomkit.video.pipeline.resizer.base import VideoResizerProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.pipeline.resizer")

# Map raw codec names to PyAV pixel format identifiers.
_RAW_CODEC_TO_PIX_FMT: dict[str, str] = {
    "raw_rgb24": "rgb24",
    "raw_bgr24": "bgr24",
    "raw_yuv420p": "yuv420p",
    "raw_nv12": "nv12",
}


class PyAVVideoResizer(VideoResizerProvider):
    """Resize raw video frames using PyAV.

    Only resizes if the frame exceeds the target dimensions.
    Aspect ratio is preserved by default (letterboxing is not
    applied — the frame is scaled to fit within the bounding box).

    Args:
        width: Maximum output width in pixels.
        height: Maximum output height in pixels.
        keep_aspect: Whether to preserve aspect ratio when scaling.

    Example::

        resizer = PyAVVideoResizer(width=320, height=240)
        small_frame = resizer.resize(large_frame)
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        *,
        keep_aspect: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be positive, got {width}x{height}")
        self._target_width = width
        self._target_height = height
        self._keep_aspect = keep_aspect

    @property
    def name(self) -> str:
        return "PyAVVideoResizer"

    def resize(self, frame: VideoFrame) -> VideoFrame:
        """Resize a raw frame if it exceeds the target dimensions."""
        # Only resize raw frames.
        pix_fmt = _RAW_CODEC_TO_PIX_FMT.get(frame.codec)
        if pix_fmt is None:
            logger.warning(
                "Cannot resize encoded frame (codec=%s); pass through decoder first",
                frame.codec,
            )
            return frame

        # Skip if already within target dimensions.
        if frame.width <= self._target_width and frame.height <= self._target_height:
            return frame

        new_w, new_h = self._compute_dimensions(frame.width, frame.height)
        if new_w == frame.width and new_h == frame.height:
            return frame

        try:
            return self._resize_with_pyav(frame, new_w, new_h, pix_fmt)
        except Exception:
            logger.exception(
                "Resize error for %dx%d -> %dx%d; returning original",
                frame.width,
                frame.height,
                new_w,
                new_h,
            )
            return frame

    def _compute_dimensions(self, src_w: int, src_h: int) -> tuple[int, int]:
        """Compute target dimensions, preserving aspect ratio if requested."""
        if not self._keep_aspect:
            return self._target_width, self._target_height

        scale_w = self._target_width / src_w
        scale_h = self._target_height / src_h
        scale = min(scale_w, scale_h)

        # Ensure dimensions are even (required by many codecs).
        new_w = max(2, int(src_w * scale) & ~1)
        new_h = max(2, int(src_h * scale) & ~1)
        return new_w, new_h

    def _resize_with_pyav(
        self,
        frame: VideoFrame,
        new_w: int,
        new_h: int,
        pix_fmt: str,
    ) -> VideoFrame:
        """Perform the actual resize via PyAV."""
        import av

        av_frame = av.VideoFrame(frame.width, frame.height, pix_fmt)
        av_frame.planes[0].update(frame.data)

        resized = av_frame.reformat(width=new_w, height=new_h, format=pix_fmt)
        raw_bytes = resized.to_ndarray().tobytes()

        return replace(
            frame,
            data=raw_bytes,
            width=new_w,
            height=new_h,
            metadata={**frame.metadata, "resizer": self.name},
        )
