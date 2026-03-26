"""Text overlay renderer using OpenCV."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.video.pipeline.overlay.base import Overlay, compute_position

if TYPE_CHECKING:
    import numpy as np

from roomkit.video.pipeline.overlay.base import OverlayRenderer

logger = logging.getLogger("roomkit.video.pipeline.overlay.text")

_DEFAULT_STYLE: dict[str, Any] = {
    "font_scale": 0.7,
    "color": (255, 255, 255),
    "bg_color": (0, 0, 0),
    "thickness": 1,
    "padding": 8,
    "line_spacing": 4,
    "max_width": None,
}


def _rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return (color[2], color[1], color[0])


def _import_cv2() -> Any:
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise ImportError(
            "opencv is required for TextOverlayRenderer. "
            "Install with: pip install roomkit[local-video]"
        ) from exc


def _import_numpy() -> Any:
    try:
        import numpy as np_mod

        return np_mod
    except ImportError as exc:
        raise ImportError(
            "numpy is required for TextOverlayRenderer. Install with: pip install roomkit[video]"
        ) from exc


class TextOverlayRenderer(OverlayRenderer):
    """Render text overlays using OpenCV ``putText``.

    Supports multi-line text (split on ``\\n``), configurable font,
    background rectangle, and all 9 named positions.

    Style keys:
        font_scale (float): OpenCV font scale. Default 0.7.
        color (tuple[int,int,int]): Text RGB color. Default white.
        bg_color (tuple[int,int,int] | None): Background RGB. Default black.
        thickness (int): Font thickness. Default 1.
        padding (int): Padding around text block. Default 8.
        line_spacing (int): Extra pixels between lines. Default 4.
    """

    def __init__(self) -> None:
        self._cv2 = _import_cv2()
        self._np = _import_numpy()
        self._font = self._cv2.FONT_HERSHEY_SIMPLEX
        # Cache: overlay_id → (version, rendered_patch, patch_w, patch_h)
        self._cache: dict[str, tuple[int, Any, int, int]] = {}

    @property
    def overlay_type(self) -> str:
        return "text"

    def render(
        self,
        canvas: np.ndarray,
        overlay: Overlay,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        content = overlay.content
        if not isinstance(content, str) or not content.strip():
            return canvas

        style = {**_DEFAULT_STYLE, **overlay.style}
        cached = self._cache.get(overlay.id)

        if cached is not None and cached[0] == overlay.version:
            patch, patch_w, patch_h = cached[1], cached[2], cached[3]
        else:
            patch, patch_w, patch_h = self._render_patch(content, style)
            self._cache[overlay.id] = (overlay.version, patch, patch_w, patch_h)

        padding = style["padding"]
        x, y = compute_position(
            overlay.position,
            frame_width,
            frame_height,
            patch_w,
            patch_h,
            custom_x=overlay.x,
            custom_y=overlay.y,
            padding=padding,
        )

        return self._blit(canvas, patch, x, y, overlay.opacity)

    def invalidate_cache(self, overlay_id: str) -> None:
        if overlay_id:
            self._cache.pop(overlay_id, None)
        else:
            self._cache.clear()

    def _render_patch(self, text: str, style: dict[str, Any]) -> tuple[Any, int, int]:
        """Render text to an RGBA patch image."""
        cv2 = self._cv2
        np = self._np

        font_scale = style["font_scale"]
        thickness = style["thickness"]
        padding = style["padding"]
        line_spacing = style["line_spacing"]

        lines = text.split("\n")
        line_metrics = []
        max_w = 0
        total_h = 0
        for line in lines:
            (tw, th), baseline = cv2.getTextSize(line, self._font, font_scale, thickness)
            line_metrics.append((tw, th, baseline))
            max_w = max(max_w, tw)
            total_h += th + baseline + line_spacing

        patch_w = max_w + padding * 2
        patch_h = total_h + padding * 2
        patch = np.zeros((patch_h, patch_w, 4), dtype=np.uint8)

        # Background
        bg_color = style.get("bg_color")
        if bg_color is not None:
            r, g, b = bg_color
            patch[:, :] = [r, g, b, 200]

        # Draw text lines
        color = style["color"]
        y_cursor = padding
        for line, (_tw, th, baseline) in zip(lines, line_metrics, strict=True):
            y_cursor += th
            cv2.putText(
                patch,
                line,
                (padding, y_cursor),
                self._font,
                font_scale,
                (*_rgb_to_bgr(color), 255),
                thickness,
                cv2.LINE_AA,
            )
            y_cursor += baseline + line_spacing

        return patch, patch_w, patch_h

    def _blit(
        self,
        canvas: np.ndarray,
        patch: Any,
        x: int,
        y: int,
        opacity: float,
    ) -> np.ndarray:
        """Blit an RGBA patch onto the RGB canvas with alpha blending."""
        np = self._np
        ch, cw = canvas.shape[:2]
        ph, pw = patch.shape[:2]

        # Clip to canvas bounds
        x1, y1 = max(0, x), max(0, y)
        x2 = min(cw, x + pw)
        y2 = min(ch, y + ph)
        if x1 >= x2 or y1 >= y2:
            return canvas

        px1 = x1 - x
        py1 = y1 - y
        px2 = px1 + (x2 - x1)
        py2 = py1 + (y2 - y1)

        region = patch[py1:py2, px1:px2]
        alpha = region[:, :, 3:4].astype(np.float32) / 255.0 * opacity
        rgb = region[:, :, :3].astype(np.float32)

        bg = canvas[y1:y2, x1:x2].astype(np.float32)
        # Alpha-convert RGB→BGR for the patch since canvas is RGB
        # but we drew with BGR putText — convert patch back to RGB
        rgb_corrected = rgb[:, :, ::-1]
        blended = (1.0 - alpha) * bg + alpha * rgb_corrected
        canvas[y1:y2, x1:x2] = blended.astype(np.uint8)
        return canvas
