"""Video pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.video.pipeline.decoder.base import VideoDecoderProvider
    from roomkit.video.pipeline.filter.base import VideoFilterProvider
    from roomkit.video.pipeline.resizer.base import VideoResizerProvider
    from roomkit.video.recorder.base import VideoRecorder, VideoRecordingConfig
    from roomkit.video.vision.base import VisionProvider


@dataclass
class VideoPipelineConfig:
    """Configuration for the video processing pipeline.

    All stages are optional — only configured stages run.

    Inbound order: [Decoder] -> [Resizer] -> [Filter] -> taps/vision

    Mirrors :class:`AudioPipelineConfig` for the voice subsystem.
    """

    decoder: VideoDecoderProvider | None = None
    """Optional decoder for converting encoded frames to raw pixels."""

    resizer: VideoResizerProvider | None = None
    """Optional resizer for scaling raw frames to target dimensions."""

    filter: VideoFilterProvider | None = None
    """Optional filter for inspecting/replacing frames (e.g., censor)."""

    vision: VisionProvider | None = None
    """Optional vision provider for periodic frame analysis."""

    recorder: VideoRecorder | None = None
    """Optional video recorder for capturing frames to file."""

    recording_config: VideoRecordingConfig | None = None
    """Optional recording configuration (storage path, codec, fps)."""
