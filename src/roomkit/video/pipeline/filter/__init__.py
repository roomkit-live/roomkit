"""Video filter pipeline stage."""

from __future__ import annotations

from roomkit.video.pipeline.filter.base import (
    FilterContext,
    FilterEvent,
    VideoFilterProvider,
)
from roomkit.video.pipeline.filter.mediapipe_face_touch import (
    FaceTouchConfig,
    FaceTouchFilter,
    FaceTouchSensitivity,
    FaceZone,
)
from roomkit.video.pipeline.filter.mock import MockVideoFilterProvider
from roomkit.video.pipeline.filter.mock_face_touch import MockFaceTouchFilter
from roomkit.video.pipeline.filter.watermark import WatermarkFilter
from roomkit.video.pipeline.filter.yolo import YOLODetectorFilter

__all__ = [
    "FaceTouchConfig",
    "FaceTouchFilter",
    "FaceTouchSensitivity",
    "FaceZone",
    "FilterContext",
    "FilterEvent",
    "MockFaceTouchFilter",
    "MockVideoFilterProvider",
    "VideoFilterProvider",
    "WatermarkFilter",
    "YOLODetectorFilter",
]
