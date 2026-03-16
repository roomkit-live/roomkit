"""Video utility functions."""

from __future__ import annotations

from typing import Any

from roomkit.video.video_frame import VideoFrame

# Lazy-loaded optional deps
_cv2: Any = None
_np: Any = None


def _ensure_deps() -> None:
    global _cv2, _np  # noqa: PLW0603
    if _cv2 is None:
        try:
            import cv2

            _cv2 = cv2
        except ImportError as exc:
            msg = (
                "opencv is required for make_text_frame. "
                "Install with: pip install roomkit[local-video]"
            )
            raise ImportError(msg) from exc
    if _np is None:
        import numpy as np

        _np = np


def make_text_frame(
    text: str,
    *,
    width: int = 720,
    height: int = 480,
    bg_color: tuple[int, int, int] = (30, 30, 30),
    text_color: tuple[int, int, int] = (255, 255, 255),
    sub_color: tuple[int, int, int] = (150, 150, 150),
    font_scale: float = 0.8,
) -> VideoFrame:
    """Generate a VideoFrame with centered text on a solid background.

    Renders multi-line text (split on ``\\n``) centered on the frame.
    The first line uses ``text_color``, subsequent lines use ``sub_color``.

    Requires ``opencv-python-headless`` (``pip install roomkit[local-video]``).

    Args:
        text: Text to render. Use ``\\n`` for multiple lines.
        width: Frame width in pixels.
        height: Frame height in pixels.
        bg_color: Background RGB color.
        text_color: First line text RGB color.
        sub_color: Subsequent lines RGB color.
        font_scale: OpenCV font scale.

    Returns:
        A raw RGB24 :class:`VideoFrame`.

    Example::

        from roomkit.video.utils import make_text_frame

        frame = make_text_frame("Connecting to avatar...\\nPlease wait")
    """
    _ensure_deps()

    img = _np.zeros((height, width, 3), dtype=_np.uint8)
    img[:] = bg_color

    font = _cv2.FONT_HERSHEY_SIMPLEX
    lines = text.split("\n")

    # Measure total height
    line_metrics = []
    total_h = 0
    for line in lines:
        (tw, th), baseline = _cv2.getTextSize(line.strip(), font, font_scale, 2)
        line_metrics.append((tw, th, baseline))
        total_h += th + baseline + 10

    # Start y so the block is vertically centered
    y = (height - total_h) // 2

    for i, (line, (tw, th, baseline)) in enumerate(zip(lines, line_metrics, strict=True)):
        x = (width - tw) // 2
        y += th
        color = text_color if i == 0 else sub_color
        thickness = 2 if i == 0 else 1
        scale = font_scale if i == 0 else font_scale * 0.65
        # Recalculate for sub-lines with smaller scale
        if i > 0:
            (tw, th), baseline = _cv2.getTextSize(line.strip(), font, scale, thickness)
            x = (width - tw) // 2
        _cv2.putText(img, line.strip(), (x, y), font, scale, color, thickness, _cv2.LINE_AA)
        y += baseline + 10

    return VideoFrame(data=img.tobytes(), codec="raw_rgb24", width=width, height=height)
