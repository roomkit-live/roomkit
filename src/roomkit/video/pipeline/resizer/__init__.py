"""Video resizer pipeline stage."""

from __future__ import annotations

from roomkit.video.pipeline.resizer.base import VideoResizerProvider
from roomkit.video.pipeline.resizer.mock import MockVideoResizerProvider

__all__ = [
    "MockVideoResizerProvider",
    "VideoResizerProvider",
]
