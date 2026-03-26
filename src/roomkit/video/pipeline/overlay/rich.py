"""Rich overlay renderer using Pillow for styled text and tables.

Requires ``Pillow>=10.0``::

    pip install roomkit[video-overlay]
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from roomkit.video.pipeline.overlay.base import (
    Overlay,
    OverlayRenderer,
    blit_rgba,
    compute_position,
    import_numpy,
)

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("roomkit.video.pipeline.overlay.rich")

_DEFAULT_STYLE: dict[str, Any] = {
    "width": 400,
    "font_size": 16,
    "color": (255, 255, 255),
    "bg_color": (0, 0, 0),
    "padding": 12,
    "line_spacing": 4,
    "table_border_color": (100, 100, 100),
    "table_header_bg": (50, 50, 80),
}


class RichOverlayRenderer(OverlayRenderer):
    """Render styled text and tables using Pillow.

    Supports multi-line styled text with TrueType fonts and simple
    table rendering with headers, borders, and cell alignment.

    ``overlay.content`` is a string.  For tables, pass a JSON-encoded
    dict with ``{"headers": [...], "rows": [[...], ...]}``.

    Style keys:
        width (int): Render width in pixels. Default 400.
        font_size (int): Font size. Default 16.
        color (tuple): Text RGB color. Default white.
        bg_color (tuple): Background RGB. Default black.
        padding (int): Padding. Default 12.
        table_border_color (tuple): Border RGB for tables.
        table_header_bg (tuple): Header background RGB for tables.
    """

    def __init__(self) -> None:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for RichOverlayRenderer. "
                "Install with: pip install roomkit[video-overlay]"
            ) from exc

        self._image_cls = Image
        self._draw_cls = ImageDraw
        self._font_cls = ImageFont
        self._np = import_numpy()
        # Cache: overlay_id → (version, rgba_array, w, h)
        self._cache: dict[str, tuple[int, Any, int, int]] = {}

    @property
    def overlay_type(self) -> str:
        return "rich"

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

        cached = self._cache.get(overlay.id)
        if cached is not None and cached[0] == overlay.version:
            patch, pw, ph = cached[1], cached[2], cached[3]
        else:
            style = {**_DEFAULT_STYLE, **overlay.style}
            patch, pw, ph = self._render_patch(content, style)
            self._cache[overlay.id] = (overlay.version, patch, pw, ph)

        padding = overlay.style.get("padding", _DEFAULT_STYLE["padding"])
        x, y = compute_position(
            overlay.position,
            frame_width,
            frame_height,
            pw,
            ph,
            custom_x=overlay.x,
            custom_y=overlay.y,
            padding=padding,
        )

        return blit_rgba(canvas, patch, x, y, overlay.opacity, self._np)

    def invalidate_cache(self, overlay_id: str) -> None:
        self._cache.pop(overlay_id, None)

    def clear_cache(self) -> None:
        self._cache.clear()

    def _render_patch(self, content: str, style: dict[str, Any]) -> tuple[Any, int, int]:
        """Render content to an RGBA numpy array."""
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "headers" in data and "rows" in data:
                return self._render_table(data, style)
        except (json.JSONDecodeError, TypeError):
            pass

        return self._render_text(content, style)

    def _render_text(self, text: str, style: dict[str, Any]) -> tuple[Any, int, int]:
        """Render styled multi-line text."""
        np = self._np
        width = style["width"]
        font = self._get_font(style["font_size"])
        padding = style["padding"]
        line_spacing = style["line_spacing"]

        lines = text.split("\n")
        line_h = style["font_size"] + line_spacing
        height = len(lines) * line_h + padding * 2

        img = self._image_cls.new("RGBA", (width, height), (*style["bg_color"], 200))
        draw = self._draw_cls.Draw(img)

        y_cursor = padding
        for line in lines:
            draw.text(
                (padding, y_cursor),
                line,
                fill=(*style["color"], 255),
                font=font,
            )
            y_cursor += line_h

        return np.array(img, dtype=np.uint8), width, height

    def _render_table(self, data: dict[str, Any], style: dict[str, Any]) -> tuple[Any, int, int]:
        """Render a simple table with headers and rows."""
        np = self._np
        headers = data["headers"]
        rows = data["rows"]
        font = self._get_font(style["font_size"])
        padding = style["padding"]
        row_h = style["font_size"] + padding
        col_w = style["width"] // max(len(headers), 1)
        width = col_w * len(headers)
        height = row_h * (len(rows) + 1) + padding

        img = self._image_cls.new("RGBA", (width, height), (*style["bg_color"], 200))
        draw = self._draw_cls.Draw(img)

        # Header row
        header_bg = (*style["table_header_bg"], 220)
        draw.rectangle([(0, 0), (width, row_h)], fill=header_bg)
        for i, header in enumerate(headers):
            draw.text(
                (i * col_w + padding // 2, padding // 2),
                str(header),
                fill=(*style["color"], 255),
                font=font,
            )

        # Data rows
        border = (*style["table_border_color"], 180)
        for r_idx, row in enumerate(rows):
            y_pos = (r_idx + 1) * row_h
            draw.line([(0, y_pos), (width, y_pos)], fill=border, width=1)
            for c_idx, cell in enumerate(row):
                draw.text(
                    (c_idx * col_w + padding // 2, y_pos + padding // 2),
                    str(cell),
                    fill=(*style["color"], 255),
                    font=font,
                )

        return np.array(img, dtype=np.uint8), width, height

    def _get_font(self, size: int) -> Any:
        """Get a Pillow font.  Falls back to default if unavailable."""
        try:
            return self._font_cls.truetype("DejaVuSans.ttf", size)
        except (OSError, AttributeError):
            return self._font_cls.load_default()
