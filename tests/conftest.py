"""Shared test fixtures and helpers."""

from __future__ import annotations

import asyncio
import collections
import os
import traceback
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


class PoolCheckoutRecorder:
    """Stand-in for a store's connection pool that records every checkout.

    Substitute it for ``PostgresStore._ensure_pool()`` to count how many pooled
    connections a piece of work costs. A connection lent from an open
    ``store.connection()`` block never reaches the pool, so it is invisible
    here — which is exactly the quantity worth measuring: asyncpg resets a
    connection on release, so a checkout is a full extra round trip.

    Everything other than ``acquire`` is delegated to the real pool.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self.by_call_site: collections.Counter[str] = collections.Counter()

    @property
    def total(self) -> int:
        return sum(self.by_call_site.values())

    def reset(self) -> None:
        """Forget what was recorded — e.g. after a warm-up phase."""
        self.by_call_site.clear()

    def breakdown(self, per: int = 1) -> str:
        """The recorded checkouts per call site, divided by *per*."""
        return "\n".join(
            f"  {count / per:5.2f}  {site}" for site, count in self.by_call_site.most_common()
        )

    def acquire(self, *args: Any, **kwargs: Any) -> Any:
        self.by_call_site[_store_call_site()] += 1
        return self._pool.acquire(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)


def _store_call_site() -> str:
    """The innermost store method, plus the frame that asked it for a connection."""
    stack = traceback.extract_stack()[:-2]
    store_frame = next((f for f in reversed(stack) if "roomkit/store/" in f.filename), None)
    caller = next(
        (
            f
            for f in reversed(stack)
            if "roomkit/" in f.filename and "roomkit/store/" not in f.filename
        ),
        None,
    )
    return "{} <- {}:{} {}".format(
        store_frame.name if store_frame else "?",
        os.path.basename(caller.filename) if caller else "?",
        caller.lineno if caller else "?",
        caller.name if caller else "?",
    )


def make_event(
    room_id: str = "test-room",
    channel_id: str = "ch1",
    channel_type: ChannelType = ChannelType.SMS,
    body: str = "hello",
    participant_id: str | None = None,
    **kwargs: object,
) -> RoomEvent:
    return RoomEvent(
        room_id=room_id,
        source=EventSource(
            channel_id=channel_id, channel_type=channel_type, participant_id=participant_id
        ),
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
