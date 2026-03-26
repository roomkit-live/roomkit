"""Image overlay renderer — blit images onto video frames."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.video.pipeline.overlay.base import (
    Overlay,
    OverlayRenderer,
    blit_rgba,
    compute_position,
    import_cv2,
    import_numpy,
)

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("roomkit.video.pipeline.overlay.image")


class ImageOverlayRenderer(OverlayRenderer):
    """Render image overlays by blitting decoded images onto frames.

    ``overlay.content`` should be PNG/JPEG bytes.  Images are decoded
    once and cached until the overlay version changes.  Supports
    alpha blending for PNG images with transparency.

    Style keys:
        width (int | None): Target width. If only width is set, height
            is computed to preserve aspect ratio.
        height (int | None): Target height. If only height is set, width
            is computed to preserve aspect ratio. If both are set, the
            image is stretched to fit.
        padding (int): Padding from edge. Default 10.
    """

    def __init__(self) -> None:
        self._cv2 = import_cv2()
        self._np = import_numpy()
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

        return blit_rgba(canvas, img_rgba, x, y, overlay.opacity, self._np)

    def invalidate_cache(self, overlay_id: str) -> None:
        self._cache.pop(overlay_id, None)

    def clear_cache(self) -> None:
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
