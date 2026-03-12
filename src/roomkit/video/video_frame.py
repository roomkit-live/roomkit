"""VideoFrame data model for inbound video pipeline processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Codecs that carry encoded (compressed) data.
ENCODED_CODECS = frozenset({"h264", "vp8", "vp9", "av1"})

# Codecs that carry raw (uncompressed) pixel data.
RAW_CODECS = frozenset({"raw_rgb24", "raw_bgr24", "raw_yuv420p", "raw_nv12"})

VALID_CODECS = ENCODED_CODECS | RAW_CODECS


@dataclass
class VideoFrame:
    """A single frame of inbound video for pipeline processing.

    VideoFrame flows through video pipeline stages. Each stage may
    annotate the ``metadata`` dict with its results.

    For encoded codecs (h264, vp8, vp9, av1), ``data`` contains
    encoded NAL units or a compressed frame.  For raw codecs
    (raw_rgb24, raw_yuv420p, etc.), ``data`` contains pixel bytes.
    """

    data: bytes
    """Frame data — encoded NAL units or raw pixel bytes."""

    codec: str = "h264"
    """Codec identifier (e.g. 'h264', 'vp8', 'raw_rgb24')."""

    width: int = 640
    """Frame width in pixels."""

    height: int = 480
    """Frame height in pixels."""

    timestamp_ms: float | None = None
    """Timestamp in milliseconds (relative to session start)."""

    keyframe: bool = False
    """Whether this is a keyframe (IDR for H.264, key for VP8/VP9)."""

    sequence: int = 0
    """Frame sequence number (monotonically increasing per session)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Pipeline stages annotate results here."""

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise ValueError("VideoFrame.data must be bytes")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width and height must be positive, got {self.width}x{self.height}")
        if self.sequence < 0:
            raise ValueError(f"sequence must be >= 0, got {self.sequence}")
        if self.codec not in VALID_CODECS:
            raise ValueError(f"codec must be one of {sorted(VALID_CODECS)}, got {self.codec!r}")

    @property
    def is_encoded(self) -> bool:
        """Whether this frame carries encoded (compressed) data."""
        return self.codec in ENCODED_CODECS

    @property
    def is_raw(self) -> bool:
        """Whether this frame carries raw (uncompressed) pixel data."""
        return self.codec in RAW_CODECS
