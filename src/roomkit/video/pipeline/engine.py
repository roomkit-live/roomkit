"""Video pipeline engine — frame processing orchestrator."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roomkit.video.pipeline.filter.base import FilterContext, FilterEvent

if TYPE_CHECKING:
    from roomkit.video.pipeline.config import VideoPipelineConfig
    from roomkit.video.video_frame import VideoFrame
    from roomkit.video.vision.base import VisionResult

logger = logging.getLogger("roomkit.video.pipeline")


class VideoPipeline:
    """Orchestrates inbound video processing through pluggable stages.

    Inbound processing order:
        [Decoder] -> [Resizer] -> [Transforms...] -> [Filters...] -> return processed frame

    Stages are optional — only configured stages run.  Vision analysis
    is async and invoked separately via :meth:`process_vision`.
    Transforms modify pixel data (grayscale, blur, etc.) before filters.
    Filters are chained — each receives a :class:`FilterContext` that
    is updated automatically when vision results arrive.
    """

    def __init__(self, config: VideoPipelineConfig) -> None:
        self._config = config
        self._filter_contexts: dict[str, FilterContext] = {}

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

        # Stage 3: Transforms (chained, pixel-level modifications).
        for txf in self._config.transforms:
            try:
                current = txf.transform(current)
            except Exception:
                logger.exception("Transform %s error for frame seq=%d", txf.name, frame.sequence)

        # Stage 4: Filters (chained, each can inspect/replace).
        if self._config.filters:
            ctx = self._filter_contexts.setdefault(session_id, FilterContext())
            ctx.session_id = session_id
            for flt in self._config.filters:
                try:
                    current = flt.filter(current, ctx)
                except Exception:
                    logger.exception("Filter %s error for frame seq=%d", flt.name, frame.sequence)

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
            result = await self._config.vision.analyze_frame(frame)
        except Exception:
            logger.exception("Vision analysis error for session %s", session_id[:8])
            return None

        # Update filter context with latest vision result
        if result is not None and self._config.filters:
            ctx = self._filter_contexts.setdefault(session_id, FilterContext())
            ctx.last_vision_result = result
            ctx.labels_detected = set(result.labels)

        return result

    def drain_events(self, session_id: str) -> list[FilterEvent]:
        """Drain and return pending filter events for a session.

        Called by the channel layer after :meth:`process_inbound` to
        collect events emitted by filters during frame processing.
        The events list is cleared atomically.

        Args:
            session_id: Active session identifier.

        Returns:
            List of filter events (empty if none pending).
        """
        ctx = self._filter_contexts.get(session_id)
        if ctx is None or not ctx.events:
            return []
        events = list(ctx.events)
        ctx.events.clear()
        return events

    def update_filter_context(self, session_id: str, result: VisionResult) -> None:
        """Update the filter context for a session with a new vision result.

        Called by VideoHooksMixin after vision analysis completes, so
        external code doesn't need to reach into private attributes.

        Args:
            session_id: Active session identifier.
            result: The latest vision analysis result.
        """
        if not self._config.filters:
            return
        ctx = self._filter_contexts.setdefault(session_id, FilterContext())
        ctx.last_vision_result = result
        ctx.labels_detected = set(result.labels)

    def reset(self, session_id: str) -> None:
        """Reset pipeline state for a session.

        Args:
            session_id: Session whose state should be cleared.
        """
        if self._config.decoder is not None:
            self._config.decoder.reset()
        for txf in self._config.transforms:
            txf.reset()
        for flt in self._config.filters:
            flt.reset()
        self._filter_contexts.pop(session_id, None)

    def close(self) -> None:
        """Release all pipeline resources."""
        if self._config.decoder is not None:
            self._config.decoder.close()
        if self._config.resizer is not None:
            self._config.resizer.close()
        for txf in self._config.transforms:
            txf.close()
        for flt in self._config.filters:
            flt.close()
        self._filter_contexts.clear()
