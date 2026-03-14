"""Video pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.video.pipeline.decoder.base import VideoDecoderProvider
    from roomkit.video.pipeline.filter.base import VideoFilterProvider
    from roomkit.video.pipeline.resizer.base import VideoResizerProvider
    from roomkit.video.pipeline.transform.base import VideoTransformProvider
    from roomkit.video.recorder.base import VideoRecorder, VideoRecordingConfig
    from roomkit.video.vision.base import VisionProvider


@dataclass
class VideoPipelineConfig:
    """Configuration for the video processing pipeline.

    All stages are optional — only configured stages run.

    Inbound order: [Decoder] -> [Resizer] -> [Transforms...] -> [Filters...] -> taps/vision

    Transforms modify pixel data (grayscale, blur, etc.) before
    filters inspect vision results.  Both are lists — multiple stages
    are chained in order.  Decoder, resizer, and vision are singular
    (one per pipeline).

    Mirrors :class:`AudioPipelineConfig` for the voice subsystem.
    """

    decoder: VideoDecoderProvider | None = None
    """Optional decoder for converting encoded frames to raw pixels."""

    resizer: VideoResizerProvider | None = None
    """Optional resizer for scaling raw frames to target dimensions."""

    transforms: list[VideoTransformProvider] = field(default_factory=list)
    """Transforms chained in order — each modifies pixel data."""

    filters: list[VideoFilterProvider] = field(default_factory=list)
    """Filters chained in order — each can inspect or replace frames."""

    vision: VisionProvider | None = None
    """Optional vision provider for periodic frame analysis."""

    recorder: VideoRecorder | None = None
    """Optional video recorder for capturing frames to file."""

    recording_config: VideoRecordingConfig | None = None
    """Optional recording configuration (storage path, codec, fps)."""
