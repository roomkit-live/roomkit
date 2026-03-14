"""Video decoder pipeline stage."""

from __future__ import annotations

from roomkit.video.pipeline.decoder.base import VideoDecoderProvider
from roomkit.video.pipeline.decoder.mock import MockVideoDecoderProvider

__all__ = [
    "MockVideoDecoderProvider",
    "VideoDecoderProvider",
]
