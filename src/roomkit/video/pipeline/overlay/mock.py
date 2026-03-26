"""Mock overlay renderer for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.video.pipeline.overlay.base import OverlayRenderer

if TYPE_CHECKING:
    import numpy as np

    from roomkit.video.pipeline.overlay.base import Overlay


class MockOverlayRenderer(OverlayRenderer):
    """Pass-through renderer that tracks calls for testing."""

    def __init__(self, overlay_type: str = "text") -> None:
        self._overlay_type = overlay_type
        self.render_count = 0
        self.last_overlay: Overlay | None = None
        self.invalidated: list[str] = []

    @property
    def overlay_type(self) -> str:
        return self._overlay_type

    def render(
        self,
        canvas: np.ndarray,
        overlay: Overlay,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        self.render_count += 1
        self.last_overlay = overlay
        return canvas

    def invalidate_cache(self, overlay_id: str) -> None:
        self.invalidated.append(overlay_id)
