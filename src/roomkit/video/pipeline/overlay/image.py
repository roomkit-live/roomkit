"""Image overlay renderer — blit images onto video frames."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.video.pipeline.overlay.base import Overlay, OverlayRenderer, compute_position

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("roomkit.video.pipeline.overlay.image")


def _import_cv2() -> Any:
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise ImportError(
            "opencv is required for ImageOverlayRenderer. "
            "Install with: pip install roomkit[local-video]"
        ) from exc


def _import_numpy() -> Any:
    try:
        import numpy as np_mod

        return np_mod
    except ImportError as exc:
        raise ImportError(
            "numpy is required for ImageOverlayRenderer. Install with: pip install roomkit[video]"
        ) from exc


class ImageOverlayRenderer(OverlayRenderer):
    """Render image overlays by blitting decoded images onto frames.

    ``overlay.content`` should be PNG/JPEG bytes.  Images are decoded
    once and cached until the overlay version changes.  Supports
    alpha blending for PNG images with transparency.

    Style keys:
        width (int | None): Target width (aspect-ratio preserved).
        height (int | None): Target height (aspect-ratio preserved).
        padding (int): Padding from edge. Default 10.
    """

    def __init__(self) -> None:
        self._cv2 = _import_cv2()
        self._np = _import_numpy()
        # Cache: overlay_id → (version, decoded_rgba, w, h)
        self._cache: dict[str, tuple[int, Any, int, int]] = {}

    @property
    def overlay_type(self) -> str:
        return "image"

    def render(
        self,
        canvas: np.ndarray,
        overlay: Overlay,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        content = overlay.content
        if not isinstance(content, bytes) or not content:
            return canvas

        cached = self._cache.get(overlay.id)
        if cached is not None and cached[0] == overlay.version:
            img_rgba, img_w, img_h = cached[1], cached[2], cached[3]
        else:
            img_rgba, img_w, img_h = self._decode(content, overlay.style)
            if img_rgba is None:
                return canvas
            self._cache[overlay.id] = (overlay.version, img_rgba, img_w, img_h)

        padding = overlay.style.get("padding", 10)
        x, y = compute_position(
            overlay.position,
            frame_width,
            frame_height,
            img_w,
            img_h,
            custom_x=overlay.x,
            custom_y=overlay.y,
            padding=padding,
        )

        return self._blit(canvas, img_rgba, x, y, overlay.opacity)

    def invalidate_cache(self, overlay_id: str) -> None:
        if overlay_id:
            self._cache.pop(overlay_id, None)
        else:
            self._cache.clear()

    def _decode(self, data: bytes, style: dict[str, Any]) -> tuple[Any | None, int, int]:
        """Decode image bytes to RGBA numpy array."""
        cv2 = self._cv2
        np = self._np

        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("Failed to decode image overlay")
            return None, 0, 0

        # Convert to RGBA
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

        # Resize if requested
        target_w = style.get("width")
        target_h = style.get("height")
        if target_w or target_h:
            h, w = img.shape[:2]
            if target_w and not target_h:
                target_h = int(h * target_w / w)
            elif target_h and not target_w:
                target_w = int(w * target_h / h)
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        return img, img.shape[1], img.shape[0]

    def _blit(
        self,
        canvas: np.ndarray,
        img_rgba: Any,
        x: int,
        y: int,
        opacity: float,
    ) -> np.ndarray:
        """Blit RGBA image onto RGB canvas with alpha blending."""
        np = self._np
        ch, cw = canvas.shape[:2]
        ih, iw = img_rgba.shape[:2]

        x1, y1 = max(0, x), max(0, y)
        x2 = min(cw, x + iw)
        y2 = min(ch, y + ih)
        if x1 >= x2 or y1 >= y2:
            return canvas

        px1, py1 = x1 - x, y1 - y
        region = img_rgba[py1 : py1 + (y2 - y1), px1 : px1 + (x2 - x1)]
        alpha = region[:, :, 3:4].astype(np.float32) / 255.0 * opacity
        rgb = region[:, :, :3].astype(np.float32)

        bg = canvas[y1:y2, x1:x2].astype(np.float32)
        canvas[y1:y2, x1:x2] = ((1.0 - alpha) * bg + alpha * rgb).astype(np.uint8)
        return canvas
