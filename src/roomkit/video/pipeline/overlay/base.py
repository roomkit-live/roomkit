"""Overlay types, renderer ABC, and shared utilities for the video pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


# -- Shared import helpers ------------------------------------------------


def import_cv2() -> Any:
    """Import OpenCV, raising a helpful error if missing."""
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise ImportError(
            "opencv is required for overlay rendering. "
            "Install with: pip install roomkit[local-video]"
        ) from exc


def import_numpy() -> Any:
    """Import numpy, raising a helpful error if missing."""
    try:
        import numpy as np_mod

        return np_mod
    except ImportError as exc:
        raise ImportError(
            "numpy is required for overlay rendering. Install with: pip install roomkit[video]"
        ) from exc


# -- Position enum --------------------------------------------------------


class OverlayPosition(StrEnum):
    """Named positions for overlay placement."""

    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"
    TOP_RIGHT = "top-right"
    CENTER_LEFT = "center-left"
    CENTER = "center"
    CENTER_RIGHT = "center-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"
    BOTTOM_RIGHT = "bottom-right"
    CUSTOM = "custom"


# -- Overlay dataclass ----------------------------------------------------


@dataclass
class Overlay:
    """A single overlay to render on video frames.

    Args:
        id: Unique identifier (used for update/remove).
        content: Text string, image bytes (PNG/JPEG), or rich markup.
        overlay_type: Renderer type — ``"text"``, ``"image"``, or ``"rich"``.
        position: Named position or ``CUSTOM`` for manual x/y.
        x: Horizontal offset (only when ``position=CUSTOM``).
        y: Vertical offset (only when ``position=CUSTOM``).
        z_order: Stacking order — higher values render on top.
        opacity: Transparency (0.0 = invisible, 1.0 = opaque).
        style: Renderer-specific style keys (font_scale, color, etc.).
    """

    id: str
    content: str | bytes
    overlay_type: str = "text"
    position: OverlayPosition = OverlayPosition.BOTTOM_CENTER
    x: int = 0
    y: int = 0
    z_order: int = 0
    opacity: float = 1.0
    style: dict[str, Any] = field(default_factory=dict)
    version: int = field(default=0, repr=False)


# -- Renderer ABC ---------------------------------------------------------


class OverlayRenderer(ABC):
    """Renders a specific overlay type onto a numpy array.

    Each renderer handles one ``overlay_type`` (e.g. ``"text"``).
    Renderers should cache their output and only re-render when
    the overlay's ``version`` changes.
    """

    @property
    @abstractmethod
    def overlay_type(self) -> str:
        """The overlay type this renderer handles."""

    @abstractmethod
    def render(
        self,
        canvas: np.ndarray,
        overlay: Overlay,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """Render the overlay onto *canvas* (H, W, 3 RGB uint8).

        Returns the modified canvas (may be the same array).
        """

    def invalidate_cache(self, overlay_id: str) -> None:  # noqa: B027
        """Called when a specific overlay's content changes."""

    def clear_cache(self) -> None:  # noqa: B027
        """Clear all cached renders (called on filter reset)."""


# -- Shared geometry ------------------------------------------------------


def compute_position(
    position: OverlayPosition,
    frame_w: int,
    frame_h: int,
    content_w: int,
    content_h: int,
    custom_x: int = 0,
    custom_y: int = 0,
    padding: int = 10,
) -> tuple[int, int]:
    """Compute top-left (x, y) for an overlay given its position."""
    if position == OverlayPosition.CUSTOM:
        return custom_x, custom_y

    # Format: "vertical-horizontal" e.g. "top-left", "bottom-center"
    # Special case: "center" (no dash) means both axes centered
    h_align = position.value.split("-")[-1] if "-" in position.value else "center"
    v_align = position.value.split("-")[0] if "-" in position.value else "center"

    if h_align == "left":
        x = padding
    elif h_align == "right":
        x = frame_w - content_w - padding
    else:
        x = (frame_w - content_w) // 2

    if v_align == "top":
        y = padding
    elif v_align == "bottom":
        y = frame_h - content_h - padding
    else:
        y = (frame_h - content_h) // 2

    return max(0, x), max(0, y)


# -- Shared blitting ------------------------------------------------------


def blit_rgba(
    canvas: np.ndarray,
    patch: np.ndarray,
    x: int,
    y: int,
    opacity: float,
    np_mod: Any,
) -> np.ndarray:
    """Blit an RGBA patch onto an RGB canvas with alpha blending.

    Args:
        canvas: Target RGB array (H, W, 3) uint8.
        patch: Source RGBA array (H, W, 4) uint8.
        x: Top-left x on the canvas.
        y: Top-left y on the canvas.
        opacity: Global opacity multiplier (0.0–1.0).
        np_mod: The numpy module reference.
    """
    ch, cw = canvas.shape[:2]
    ph, pw = patch.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2 = min(cw, x + pw)
    y2 = min(ch, y + ph)
    if x1 >= x2 or y1 >= y2:
        return canvas

    px1, py1 = x1 - x, y1 - y
    region = patch[py1 : py1 + (y2 - y1), px1 : px1 + (x2 - x1)]
    alpha = region[:, :, 3:4].astype(np_mod.float32) / 255.0 * opacity
    rgb = region[:, :, :3].astype(np_mod.float32)

    bg = canvas[y1:y2, x1:x2].astype(np_mod.float32)
    canvas[y1:y2, x1:x2] = ((1.0 - alpha) * bg + alpha * rgb).astype(np_mod.uint8)
    return canvas
