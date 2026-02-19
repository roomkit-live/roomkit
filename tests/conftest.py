"""Shared test fixtures and helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import (
    CompositeContent,
    EventSource,
    MediaContent,
    RoomEvent,
    TextContent,
)
from roomkit.models.event import EventContent as EventContentType
from roomkit.models.room import Room
from roomkit.store.memory import InMemoryStore


@pytest.fixture
def advance() -> Callable[[int], Coroutine[Any, Any, None]]:
    """Yield control to let pending tasks run without real delay.

    Replaces ``await asyncio.sleep(0.05)`` patterns with zero-delay
    event loop yields::

        await advance()       # 5 yields (default)
        await advance(10)     # 10 yields for heavier workloads
    """

    async def _advance(n: int = 5) -> None:
        for _ in range(n):
            await asyncio.sleep(0)

    return _advance


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def room() -> Room:
    return Room(id="test-room")


def make_event(
    room_id: str = "test-room",
    channel_id: str = "ch1",
    channel_type: ChannelType = ChannelType.SMS,
    body: str = "hello",
    **kwargs: object,
) -> RoomEvent:
    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id=channel_id, channel_type=channel_type),
        content=TextContent(body=body),
        **kwargs,  # type: ignore[arg-type]
    )


def make_media_event(
    room_id: str = "test-room",
    channel_id: str = "ch1",
    channel_type: ChannelType = ChannelType.SMS,
    url: str = "https://example.com/image.jpg",
    mime_type: str = "image/jpeg",
    caption: str | None = None,
    extra_urls: list[str] | None = None,
    body: str | None = None,
    **kwargs: object,
) -> RoomEvent:
    """Create a RoomEvent with media content.

    Args:
        url: Primary media URL.
        mime_type: MIME type for the primary media.
        caption: Optional caption for single-media messages.
        extra_urls: Additional media URLs for composite content.
        body: Text body (used with extra_urls for composite content).
    """
    content: EventContentType
    if extra_urls:
        parts: list[TextContent | MediaContent] = []
        if body:
            parts.append(TextContent(body=body))
        parts.append(MediaContent(url=url, mime_type=mime_type))
        for extra in extra_urls:
            parts.append(MediaContent(url=extra, mime_type=mime_type))
        content = CompositeContent(parts=parts)  # type: ignore[arg-type]
    else:
        content = MediaContent(url=url, mime_type=mime_type, caption=caption)

    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id=channel_id, channel_type=channel_type),
        content=content,
        **kwargs,  # type: ignore[arg-type]
    )


def make_binding(
    channel_id: str = "ch1",
    room_id: str = "test-room",
    channel_type: ChannelType = ChannelType.SMS,
    media_types: list[ChannelMediaType] | None = None,
) -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id=room_id,
        channel_type=channel_type,
        capabilities=ChannelCapabilities(media_types=media_types or [ChannelMediaType.TEXT]),
    )
