"""Watermark video filter — overlay text on every frame."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from roomkit.video.pipeline.filter.base import FilterContext, VideoFilterProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.pipeline.filter")

# Position presets: (horizontal_align, vertical_align)
_POSITIONS = {
    "top-left": ("left", "top"),
    "top-right": ("right", "top"),
    "bottom-left": ("left", "bottom"),
    "bottom-right": ("right", "bottom"),
    "center": ("center", "center"),
}

# Padding from edges in pixels
_PADDING = 10


def _import_numpy():  # type: ignore[no-untyped-def]
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for WatermarkFilter. Install with: pip install roomkit[video]"
        ) from exc


class WatermarkFilter(VideoFilterProvider):
    """Overlay text on every video frame.

    Draws text at a configurable position using a simple pixel-based
    font renderer (no OpenCV required — uses numpy only).  Supports
    dynamic text via ``{timestamp}`` and ``{frame}`` placeholders.

    Args:
        text: Text to overlay.  Supports placeholders:
            ``{timestamp}`` — current UTC time (HH:MM:SS),
            ``{frame}`` — frame sequence number.
        position: One of ``top-left``, ``top-right``, ``bottom-left``,
            ``bottom-right``, ``center``.
        color: RGB tuple (0-255).
        bg_color: Background RGB tuple, or None for no background.
        font_scale: Text size multiplier (1.0 = ~16px height).
    """

    def __init__(
        self,
        text: str = "RECORDING",
        *,
        position: str = "top-right",
        color: tuple[int, int, int] = (255, 255, 255),
        bg_color: tuple[int, int, int] | None = (0, 0, 0),
        font_scale: float = 1.0,
    ) -> None:
        if position not in _POSITIONS:
            raise ValueError(f"position must be one of {sorted(_POSITIONS)}, got {position!r}")
        self._text = text
        self._position = position
        self._color = color
        self._bg_color = bg_color
        self._font_scale = font_scale
        self._np = _import_numpy()
        self._char_w = max(1, int(8 * font_scale))
        self._char_h = max(1, int(16 * font_scale))

    @property
    def name(self) -> str:
        return "watermark"

    def filter(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        if not frame.is_raw or frame.codec != "raw_rgb24":
            return frame

        text = self._resolve_text(frame)
        np = self._np

        w, h = frame.width, frame.height
        arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(h, w, 3).copy()

        text_w = len(text) * self._char_w
        text_h = self._char_h
        x, y = self._compute_position(w, h, text_w, text_h)

        # Draw background rectangle
        if self._bg_color is not None:
            pad = 4
            y1 = max(0, y - pad)
            y2 = min(h, y + text_h + pad)
            x1 = max(0, x - pad)
            x2 = min(w, x + text_w + pad)
            arr[y1:y2, x1:x2] = self._bg_color

        # Draw text character by character using simple block rendering
        self._draw_text(arr, text, x, y, w, h)

        from roomkit.video.video_frame import VideoFrame

        return VideoFrame(
            data=arr.tobytes(),
            codec="raw_rgb24",
            width=w,
            height=h,
            timestamp_ms=frame.timestamp_ms,
            keyframe=frame.keyframe,
            sequence=frame.sequence,
        )

    def _resolve_text(self, frame: VideoFrame) -> str:
        """Replace placeholders in the text template."""
        text = self._text
        if "{timestamp}" in text:
            text = text.replace("{timestamp}", datetime.now(UTC).strftime("%H:%M:%S"))
        if "{frame}" in text:
            text = text.replace("{frame}", str(frame.sequence))
        return text

    def _compute_position(
        self,
        img_w: int,
        img_h: int,
        text_w: int,
        text_h: int,
    ) -> tuple[int, int]:
        """Compute top-left corner for the text."""
        h_align, v_align = _POSITIONS[self._position]
        if h_align == "left":
            x = _PADDING
        elif h_align == "right":
            x = img_w - text_w - _PADDING
        else:
            x = (img_w - text_w) // 2
        if v_align == "top":
            y = _PADDING
        elif v_align == "bottom":
            y = img_h - text_h - _PADDING
        else:
            y = (img_h - text_h) // 2
        return max(0, x), max(0, y)

    def _draw_text(
        self,
        arr: object,
        text: str,
        x: int,
        y: int,
        img_w: int,
        img_h: int,
    ) -> None:
        """Draw text using simple block characters (no OpenCV needed)."""
        for ch in text:
            if x + self._char_w > img_w:
                break
            # Simple block font: fill character area with color
            # Skip spaces
            if ch != " ":
                y1 = max(0, y + 2)
                y2 = min(img_h, y + self._char_h - 2)
                x1 = max(0, x + 1)
                x2 = min(img_w, x + self._char_w - 1)
                if y2 > y1 and x2 > x1:
                    arr[y1:y2, x1:x2] = self._color  # type: ignore[index]
            x += self._char_w
