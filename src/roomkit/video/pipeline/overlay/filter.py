"""OverlayFilter — composite overlay filter for the video pipeline."""

from __future__ import annotations

import logging
import threading
from typing import Any

from roomkit.video.pipeline.filter.base import FilterContext, VideoFilterProvider
from roomkit.video.pipeline.overlay.base import Overlay, OverlayRenderer, import_numpy
from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.pipeline.overlay")


class OverlayFilter(VideoFilterProvider):
    """Composite overlay filter — renders multiple overlays on video frames.

    Implements :class:`VideoFilterProvider` so it plugs directly into
    ``VideoPipelineConfig.filters``.  Overlays are managed at runtime
    via :meth:`add_overlay`, :meth:`update_overlay`, :meth:`remove_overlay`.

    Thread-safe: ``filter()`` runs on the video callback thread while
    overlay updates may come from async hook handlers on the event loop.

    Args:
        renderers: List of :class:`OverlayRenderer` instances, one per
            overlay type (e.g. ``[TextOverlayRenderer()]``).
    """

    def __init__(self, renderers: list[OverlayRenderer] | None = None) -> None:
        self._lock = threading.Lock()
        self._overlays: dict[str, Overlay] = {}
        self._renderers: dict[str, OverlayRenderer] = {}
        self._np: Any = None
        for renderer in renderers or []:
            self._renderers[renderer.overlay_type] = renderer

    @property
    def name(self) -> str:
        return "overlay"

    # -- Overlay management (thread-safe) ---------------------------------

    def register_renderer(self, renderer: OverlayRenderer) -> None:
        """Register a renderer for an overlay type."""
        with self._lock:
            self._renderers[renderer.overlay_type] = renderer

    def add_overlay(self, overlay: Overlay) -> None:
        """Add an overlay.  Replaces any existing overlay with the same ID."""
        with self._lock:
            self._overlays[overlay.id] = overlay

    def update_overlay(self, overlay_id: str, **kwargs: Any) -> None:
        """Update overlay fields.  Bumps version for cache invalidation."""
        with self._lock:
            ov = self._overlays.get(overlay_id)
            if ov is None:
                return
            for key, val in kwargs.items():
                if hasattr(ov, key) and key != "id":
                    setattr(ov, key, val)
            ov.version += 1
            renderer = self._renderers.get(ov.overlay_type)
            if renderer is not None:
                renderer.invalidate_cache(overlay_id)

    def remove_overlay(self, overlay_id: str) -> None:
        """Remove an overlay by ID."""
        with self._lock:
            removed = self._overlays.pop(overlay_id, None)
            if removed is not None:
                renderer = self._renderers.get(removed.overlay_type)
                if renderer is not None:
                    renderer.invalidate_cache(overlay_id)

    def get_overlay(self, overlay_id: str) -> Overlay | None:
        """Get an overlay by ID (returns the live reference)."""
        with self._lock:
            return self._overlays.get(overlay_id)

    # -- VideoFilterProvider implementation --------------------------------

    def filter(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        if not frame.is_raw or frame.codec != "raw_rgb24":
            return frame

        with self._lock:
            if not self._overlays:
                return frame
            sorted_overlays = sorted(self._overlays.values(), key=lambda o: o.z_order)
            renderers = dict(self._renderers)

        if self._np is None:
            self._np = import_numpy()
        np = self._np

        w, h = frame.width, frame.height
        arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(h, w, 3).copy()

        for overlay in sorted_overlays:
            renderer = renderers.get(overlay.overlay_type)
            if renderer is None:
                continue
            arr = renderer.render(arr, overlay, w, h)

        context.metadata["overlay_ids"] = [o.id for o in sorted_overlays]

        return VideoFrame(
            data=arr.tobytes(),
            codec="raw_rgb24",
            width=w,
            height=h,
            timestamp_ms=frame.timestamp_ms,
            keyframe=frame.keyframe,
            sequence=frame.sequence,
        )

    def reset(self) -> None:
        with self._lock:
            self._overlays.clear()
            renderers = list(self._renderers.values())
        for renderer in renderers:
            renderer.clear_cache()

    def close(self) -> None:
        self.reset()
