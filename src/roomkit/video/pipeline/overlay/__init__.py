"""Rich video overlays — text, images, and tables on live video frames."""

from __future__ import annotations

from roomkit.video.pipeline.overlay.base import Overlay, OverlayPosition, OverlayRenderer
from roomkit.video.pipeline.overlay.filter import OverlayFilter
from roomkit.video.pipeline.overlay.image import ImageOverlayRenderer
from roomkit.video.pipeline.overlay.mock import MockOverlayRenderer
from roomkit.video.pipeline.overlay.subtitle import SubtitleManager, subtitle_overlay
from roomkit.video.pipeline.overlay.text import TextOverlayRenderer

__all__ = [
    "ImageOverlayRenderer",
    "MockOverlayRenderer",
    "Overlay",
    "OverlayFilter",
    "OverlayPosition",
    "OverlayRenderer",
    "SubtitleManager",
    "TextOverlayRenderer",
    "subtitle_overlay",
]

# RichOverlayRenderer requires Pillow (optional dep)
try:
    from roomkit.video.pipeline.overlay.rich import RichOverlayRenderer

    __all__ += ["RichOverlayRenderer"]
except ImportError:
    pass
