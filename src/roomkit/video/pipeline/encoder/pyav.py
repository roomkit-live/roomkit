"""PyAV H.264 encoder for outbound video."""

from __future__ import annotations

import logging
from typing import Any

from roomkit.video.pipeline.encoder.base import VideoEncoderProvider
from roomkit.video.video_frame import ENCODED_CODECS, VideoFrame

logger = logging.getLogger("roomkit.video.pipeline.encoder")


def _import_deps() -> tuple[Any, Any]:
    try:
        import av
        import numpy as np

        return av, np
    except ImportError as exc:
        raise ImportError(
            "av and numpy are required for PyAVVideoEncoder. "
            "Install with: pip install roomkit[video]"
        ) from exc


class PyAVVideoEncoder(VideoEncoderProvider):
    """Encode raw RGB frames to H.264 NAL units using PyAV/libx264.

    Outputs individual NAL units (split from Annex B stream) suitable
    for RTP packetization.  Uses Constrained Baseline profile for
    WebRTC compatibility.

    Args:
        width: Frame width (must match input frames).
        height: Frame height.
        fps: Target frame rate (sets encoder time_base).
        codec: Encoder name (default ``libx264``).
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        fps: int = 30,
        codec: str = "libx264",
    ) -> None:
        self._av, self._np = _import_deps()
        self._width = width
        self._height = height
        self._fps = fps
        self._codec_name = codec
        self._ctx: Any = None
        self._pts = 0

    def _ensure_ctx(self, width: int, height: int) -> None:
        """Create encoder context on first frame (or if dimensions change)."""
        if self._ctx is not None and self._width == width and self._height == height:
            return
        if self._ctx is not None:
            self._ctx = None  # release old
        from fractions import Fraction

        self._width = width
        self._height = height
        ctx = self._av.CodecContext.create(self._codec_name, "w")
        ctx.width = width
        ctx.height = height
        ctx.pix_fmt = "yuv420p"
        ctx.time_base = Fraction(1, self._fps)
        ctx.options = {
            "tune": "zerolatency",
            "preset": "ultrafast",
            "profile": "baseline",
            "level": "3.1",
        }
        ctx.open()
        self._ctx = ctx
        self._pts = 0
        logger.info("H.264 encoder opened: %dx%d @ %dfps", width, height, self._fps)

    @property
    def name(self) -> str:
        return "pyav-h264"

    def encode(self, frame: VideoFrame) -> list[bytes]:
        """Encode a raw VideoFrame, return H.264 NAL units."""
        # Skip already-encoded frames
        if frame.codec in ENCODED_CODECS:
            return [frame.data]

        w = frame.width
        h = frame.height
        data = frame.data

        self._ensure_ctx(w, h)

        arr = self._np.frombuffer(data, dtype=self._np.uint8).reshape(h, w, 3)
        av_frame = self._av.VideoFrame.from_ndarray(arr, format="rgb24")
        av_frame.pts = self._pts
        self._pts += 1

        nals: list[bytes] = []
        for pkt in self._ctx.encode(av_frame):
            nals.extend(_split_annex_b(bytes(pkt)))
        return nals

    def flush(self) -> list[bytes]:
        if self._ctx is None:
            return []
        nals: list[bytes] = []
        for pkt in self._ctx.encode(None):
            nals.extend(_split_annex_b(bytes(pkt)))
        return nals

    def close(self) -> None:
        self._ctx = None


def _split_annex_b(data: bytes) -> list[bytes]:
    """Split Annex B byte stream into individual NAL units."""
    nals: list[bytes] = []
    i = 0
    start = -1
    while i < len(data):
        if i + 4 <= len(data) and data[i : i + 4] == b"\x00\x00\x00\x01":
            if start >= 0:
                nals.append(data[start:i])
            start = i + 4
            i += 4
        elif i + 3 <= len(data) and data[i : i + 3] == b"\x00\x00\x01":
            if start >= 0:
                nals.append(data[start:i])
            start = i + 3
            i += 3
        else:
            i += 1
    if start >= 0 and start < len(data):
        nals.append(data[start:])
    return nals
