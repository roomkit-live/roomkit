"""Watermark video filter — overlay text on every frame."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from roomkit.video.pipeline.filter.base import FilterContext, VideoFilterProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.pipeline.filter")

_POSITIONS = {
    "top-left": ("left", "top"),
    "top-right": ("right", "top"),
    "bottom-left": ("left", "bottom"),
    "bottom-right": ("right", "bottom"),
    "center": ("center", "center"),
}

_PADDING = 10


def _import_cv2() -> Any:
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise ImportError(
            "opencv is required for WatermarkFilter. "
            "Install with: pip install roomkit[local-video]"
        ) from exc


def _import_numpy() -> Any:
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for WatermarkFilter. Install with: pip install roomkit[video]"
        ) from exc


class WatermarkFilter(VideoFilterProvider):
    """Overlay text on every video frame.

    Uses OpenCV ``putText`` for readable font rendering.  Supports
    dynamic text via ``{timestamp}`` and ``{frame}`` placeholders.

    Args:
        text: Text to overlay.  Supports placeholders:
            ``{timestamp}`` — current local time (HH:MM:SS),
            ``{frame}`` — frame sequence number.
        position: One of ``top-left``, ``top-right``, ``bottom-left``,
            ``bottom-right``, ``center``.
        color: RGB tuple (0-255).
        bg_color: Background RGB tuple, or None for no background.
        font_scale: OpenCV font scale (0.5 = small, 1.0 = normal, 2.0 = large).
        thickness: Font thickness in pixels.
    """

    def __init__(
        self,
        text: str = "RECORDING",
        *,
        position: str = "top-right",
        color: tuple[int, int, int] = (255, 255, 255),
        bg_color: tuple[int, int, int] | None = (0, 0, 0),
        font_scale: float = 0.6,
        thickness: int = 1,
    ) -> None:
        if position not in _POSITIONS:
            raise ValueError(f"position must be one of {sorted(_POSITIONS)}, got {position!r}")
        self._text = text
        self._position = position
        self._color = color
        self._bg_color = bg_color
        self._font_scale = font_scale
        self._thickness = thickness
        self._cv2 = _import_cv2()
        self._np = _import_numpy()
        self._font = self._cv2.FONT_HERSHEY_SIMPLEX

    @property
    def name(self) -> str:
        return "watermark"

    def filter(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        if not frame.is_raw or frame.codec != "raw_rgb24":
            return frame

        text = self._resolve_text(frame)
        cv2 = self._cv2
        np = self._np

        w, h = frame.width, frame.height
        arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(h, w, 3).copy()

        # Measure text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            self._font,
            self._font_scale,
            self._thickness,
        )
        x, y = self._compute_position(w, h, text_w, text_h + baseline)

        # Draw background rectangle
        if self._bg_color is not None:
            pad = 4
            cv2.rectangle(
                arr,
                (x - pad, y - pad),
                (x + text_w + pad, y + text_h + baseline + pad),
                self._color_rgb_to_bgr(self._bg_color),
                -1,  # filled
            )

        # Draw text (OpenCV putText origin is bottom-left of text)
        cv2.putText(
            arr,
            text,
            (x, y + text_h),
            self._font,
            self._font_scale,
            self._color_rgb_to_bgr(self._color),
            self._thickness,
            cv2.LINE_AA,
        )

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

    @staticmethod
    def _color_rgb_to_bgr(
        color: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """Our frames are RGB but OpenCV drawing functions expect BGR."""
        return (color[2], color[1], color[0])

    def _resolve_text(self, frame: VideoFrame) -> str:
        text = self._text
        if "{timestamp}" in text:
            text = text.replace(
                "{timestamp}",
                datetime.now().astimezone().strftime("%H:%M:%S"),
            )
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
