"""Video resources owned by either a video channel or an audio/video channel."""

from __future__ import annotations

from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.pipeline.engine import VideoPipeline
from roomkit.video.vision.base import VisionProvider


class _VideoResources:
    def __init__(self, config: VideoPipelineConfig | None, vision: VisionProvider | None) -> None:
        self.pipeline = VideoPipeline(config) if config is not None else None
        self._vision = vision
        self._pipeline_vision = config.vision if config is not None else None
        self.closed = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.pipeline is not None:
                await self.pipeline.aclose()
        finally:
            if self._vision is not None and self._vision is not self._pipeline_vision:
                await self._vision.close()
