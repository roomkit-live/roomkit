"""Video pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.video.recorder.base import VideoRecorder, VideoRecordingConfig


@dataclass
class VideoPipelineConfig:
    """Configuration for the video processing pipeline.

    All stages are optional. Additional stages (decoder, encoder,
    resizer, overlay, etc.) will be added in future phases.

    Mirrors :class:`AudioPipelineConfig` for the voice subsystem.
    """

    recorder: VideoRecorder | None = None
    """Optional video recorder for capturing frames to file."""

    recording_config: VideoRecordingConfig | None = None
    """Optional recording configuration (storage path, codec, fps)."""
