"""Overlay types and renderer ABC for the video pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


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
        """Called when an overlay's content changes."""


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

    parts = position.value.rsplit("-", 1) if "-" in position.value else ("center", position.value)
    h_part, v_part = parts

    if v_part == "left" or h_part == "left":
        # Parse correctly: "top-left" → v="top", h="left"
        pass

    # Horizontal
    h_align = position.value.split("-")[-1] if "-" in position.value else "center"
    v_align = position.value.split("-")[0] if "-" in position.value else "center"

    if h_align == "left":
        x = padding
    elif h_align == "right":
        x = frame_w - content_w - padding
    else:  # center
        x = (frame_w - content_w) // 2

    if v_align == "top":
        y = padding
    elif v_align == "bottom":
        y = frame_h - content_h - padding
    else:  # center
        y = (frame_h - content_h) // 2

    return max(0, x), max(0, y)
