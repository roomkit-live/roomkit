"""Video processing pipeline — pluggable stage orchestrator."""

from __future__ import annotations

from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.pipeline.decoder import MockVideoDecoderProvider, VideoDecoderProvider
from roomkit.video.pipeline.engine import VideoPipeline
from roomkit.video.pipeline.filter import (
    FaceTouchConfig,
    FaceTouchFilter,
    FaceTouchSensitivity,
    FaceZone,
    FilterContext,
    FilterEvent,
    MockFaceTouchFilter,
    MockVideoFilterProvider,
    VideoFilterProvider,
    WatermarkFilter,
    YOLODetectorFilter,
)
from roomkit.video.pipeline.filter.censor import CensorVideoFilter
from roomkit.video.pipeline.resizer import MockVideoResizerProvider, VideoResizerProvider
from roomkit.video.pipeline.transform import (
    MockVideoTransformProvider,
    VideoEffectTransform,
    VideoTransformProvider,
)

__all__ = [
    "CensorVideoFilter",
    "FaceTouchConfig",
    "FaceTouchFilter",
    "FaceTouchSensitivity",
    "FaceZone",
    "FilterContext",
    "FilterEvent",
    "MockFaceTouchFilter",
    "MockVideoDecoderProvider",
    "MockVideoFilterProvider",
    "MockVideoResizerProvider",
    "MockVideoTransformProvider",
    "VideoDecoderProvider",
    "VideoEffectTransform",
    "VideoFilterProvider",
    "VideoPipeline",
    "VideoPipelineConfig",
    "VideoResizerProvider",
    "VideoTransformProvider",
    "WatermarkFilter",
    "YOLODetectorFilter",
]
