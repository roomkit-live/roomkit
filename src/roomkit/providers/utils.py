"""Shared utilities for transport providers."""

from __future__ import annotations

from roomkit.models.event import RoomEvent, TextContent


def extract_event_text(event: RoomEvent) -> str:
    """Extract text from a RoomEvent, falling back to body attribute or empty string."""
    content = event.content
    if isinstance(content, TextContent):
        return content.body
    if hasattr(content, "body"):
        return str(content.body)
    return ""
