"""Subtitle overlay — wire transcription events to video overlays."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.video.pipeline.overlay.base import Overlay, OverlayPosition
from roomkit.video.pipeline.overlay.filter import OverlayFilter
from roomkit.video.pipeline.overlay.text import TextOverlayRenderer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.video.pipeline.overlay.subtitle")

SUBTITLE_OVERLAY_ID = "_subtitle"

_DEFAULT_SUBTITLE_STYLE: dict[str, Any] = {
    "font_scale": 0.8,
    "color": (255, 255, 255),
    "bg_color": (0, 0, 0),
    "padding": 8,
}


class SubtitleManager:
    """Wire transcription events to a video overlay for live subtitles.

    Creates an :class:`OverlayFilter` with a text overlay at the
    bottom of the frame.  Registers an ``ON_TRANSCRIPTION`` hook that
    updates the overlay whenever new speech is transcribed.

    Args:
        kit: RoomKit instance (for hook registration).
        overlay_filter: OverlayFilter to manage.  If ``None``, creates
            one with a :class:`TextOverlayRenderer`.
        position: Subtitle position on the frame.
        translate_fn: Optional async ``(str) → str`` for live translation.
        max_lines: Maximum visible subtitle lines (rolling window).
        style: Style overrides for the text overlay.
    """

    def __init__(
        self,
        kit: RoomKit,
        overlay_filter: OverlayFilter | None = None,
        *,
        position: OverlayPosition = OverlayPosition.BOTTOM_CENTER,
        translate_fn: Callable[[str], Awaitable[str]] | None = None,
        max_lines: int = 2,
        style: dict[str, Any] | None = None,
    ) -> None:
        if overlay_filter is None:
            overlay_filter = OverlayFilter(renderers=[TextOverlayRenderer()])
        self._filter = overlay_filter
        self._translate_fn = translate_fn
        self._max_lines = max_lines
        self._lines: list[str] = []
        self._lines_lock = threading.Lock()

        merged_style = {**_DEFAULT_SUBTITLE_STYLE, **(style or {})}
        overlay = Overlay(
            id=SUBTITLE_OVERLAY_ID,
            content="",
            overlay_type="text",
            position=position,
            z_order=100,
            style=merged_style,
        )
        self._filter.add_overlay(overlay)

        # Register hook — captures self via closure
        @kit.hook(
            HookTrigger.ON_TRANSCRIPTION,
            execution=HookExecution.ASYNC,
            name="subtitle_overlay",
        )
        async def _on_transcription(event: object, ctx: object) -> None:
            text = getattr(event, "text", "")
            if not text:
                # Try content.body for RoomEvent-based transcription
                content = getattr(event, "content", None)
                if content is not None:
                    text = getattr(content, "body", "")
            if not text:
                return
            if self._translate_fn is not None:
                text = await self._translate_fn(text)
            with self._lines_lock:
                self._lines.append(text)
                if len(self._lines) > self._max_lines:
                    self._lines = self._lines[-self._max_lines :]
                joined = "\n".join(self._lines)
            self._filter.update_overlay(SUBTITLE_OVERLAY_ID, content=joined)

    @property
    def overlay_filter(self) -> OverlayFilter:
        """The OverlayFilter to add to ``VideoPipelineConfig.filters``."""
        return self._filter

    def clear(self) -> None:
        """Clear all subtitle text."""
        with self._lines_lock:
            self._lines.clear()
        self._filter.update_overlay(SUBTITLE_OVERLAY_ID, content="")

    def set_text(self, text: str) -> None:
        """Manually set subtitle text (bypasses transcription hook)."""
        self._filter.update_overlay(SUBTITLE_OVERLAY_ID, content=text)


def subtitle_overlay(
    kit: RoomKit,
    *,
    overlay_filter: OverlayFilter | None = None,
    translate_fn: Callable[[str], Awaitable[str]] | None = None,
    position: OverlayPosition = OverlayPosition.BOTTOM_CENTER,
    max_lines: int = 2,
    **style: Any,
) -> OverlayFilter:
    """One-liner: create an overlay filter with live subtitles.

    Returns the :class:`OverlayFilter` to add to
    ``VideoPipelineConfig.filters``.

    Example::

        overlay = subtitle_overlay(kit, font_scale=0.8)
        config = VideoPipelineConfig(filters=[overlay])
    """
    manager = SubtitleManager(
        kit,
        overlay_filter=overlay_filter,
        position=position,
        translate_fn=translate_fn,
        max_lines=max_lines,
        style=style or None,
    )
    return manager.overlay_filter
