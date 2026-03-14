"""Video pipeline engine — frame processing orchestrator."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.video.pipeline.config import VideoPipelineConfig
    from roomkit.video.video_frame import VideoFrame
    from roomkit.video.vision.base import VisionResult

logger = logging.getLogger("roomkit.video.pipeline")


class VideoPipeline:
    """Orchestrates inbound video processing through pluggable stages.

    Inbound processing order:
        [Decoder] -> [Resizer] -> return processed frame

    Stages are optional — only configured stages run.  Vision analysis
    is async and invoked separately via :meth:`process_vision`.
    """

    def __init__(self, config: VideoPipelineConfig) -> None:
        self._config = config

    def process_inbound(self, session_id: str, frame: VideoFrame) -> VideoFrame | None:
        """Process an inbound video frame through the pipeline.

        Synchronous — must be fast.  Vision is async and runs via
        :meth:`process_vision` on a separate schedule.

        Args:
            session_id: Active session identifier.
            frame: Inbound video frame from the backend.

        Returns:
            Processed frame, or None if the frame should be dropped
            (e.g., decoder waiting for a keyframe).
        """
        current: VideoFrame | None = frame

        # Stage 1: Decode encoded frames to raw pixels.
        if self._config.decoder is not None and current is not None and current.is_encoded:
            try:
                current = self._config.decoder.decode(current)
            except Exception:
                logger.exception("Decoder error for frame seq=%d", frame.sequence)
                return None

        if current is None:
            return None

        # Stage 2: Resize if needed.
        if self._config.resizer is not None and current.is_raw:
            try:
                current = self._config.resizer.resize(current)
            except Exception:
                logger.exception("Resizer error for frame seq=%d", frame.sequence)
                # Continue with un-resized frame rather than dropping.

        return current

    async def process_vision(self, session_id: str, frame: VideoFrame) -> VisionResult | None:
        """Run vision analysis on a frame (async, periodic).

        Called by VideoChannel at ``vision_interval_ms``.  Returns
        None if no vision provider is configured.

        Args:
            session_id: Active session identifier.
            frame: Decoded video frame for analysis.

        Returns:
            VisionResult from the provider, or None.
        """
        if self._config.vision is None:
            return None
        try:
            return await self._config.vision.analyze_frame(frame)
        except Exception:
            logger.exception("Vision analysis error for session %s", session_id[:8])
            return None

    def reset(self, session_id: str) -> None:
        """Reset pipeline state for a session.

        Args:
            session_id: Session whose state should be cleared.
        """
        if self._config.decoder is not None:
            self._config.decoder.reset()
        if self._config.resizer is not None:
            # Resizer is stateless per-frame, but close() may release caches.
            pass

    def close(self) -> None:
        """Release all pipeline resources."""
        if self._config.decoder is not None:
            self._config.decoder.close()
        if self._config.resizer is not None:
            self._config.resizer.close()
