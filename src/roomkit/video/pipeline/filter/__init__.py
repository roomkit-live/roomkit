"""Video filter pipeline stage."""

from __future__ import annotations

from roomkit.video.pipeline.filter.base import FilterContext, VideoFilterProvider
from roomkit.video.pipeline.filter.mock import MockVideoFilterProvider
from roomkit.video.pipeline.filter.watermark import WatermarkFilter
from roomkit.video.pipeline.filter.yolo import YOLODetectorFilter

__all__ = [
    "FilterContext",
    "MockVideoFilterProvider",
    "VideoFilterProvider",
    "WatermarkFilter",
    "YOLODetectorFilter",
]
