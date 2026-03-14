"""Video processing pipeline — pluggable stage orchestrator."""

from __future__ import annotations

from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.pipeline.decoder import MockVideoDecoderProvider, VideoDecoderProvider
from roomkit.video.pipeline.engine import VideoPipeline
from roomkit.video.pipeline.resizer import MockVideoResizerProvider, VideoResizerProvider

__all__ = [
    "MockVideoDecoderProvider",
    "MockVideoResizerProvider",
    "VideoDecoderProvider",
    "VideoPipeline",
    "VideoPipelineConfig",
    "VideoResizerProvider",
]
