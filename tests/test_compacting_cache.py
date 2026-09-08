"""Summary caches must respect the reader, visible content and lifecycle."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from roomkit.memory.compacting import CompactingMemory
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.context import RoomContext
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event


def context(text: str) -> RoomContext:
    return RoomContext(
        room=Room(id="r1"),
        recent_events=[make_event(body=text * 100) for _ in range(8)],
    )


async def test_cache_isolated_by_reader_and_visible_history() -> None:
    provider = MockAIProvider(responses=["private A", "public B", "updated B"])
    memory = CompactingMemory(SlidingWindowMemory(), provider, 100, min_events=1)
    event = make_event(body="now")
    private, public = context("private"), context("public")
    await memory.retrieve("r1", event, private, channel_id="a")
    result = await memory.retrieve("r1", event, public, channel_id="b")
    assert "public B" in str(result.messages)
    assert "private A" not in str(result.messages)
    await memory.retrieve("r1", event, public, channel_id="b")
    assert len(provider.calls) == 2
    # Keep event IDs unchanged: edits and permission-driven redactions count.
    public.recent_events[0] = public.recent_events[0].model_copy(
        update={"content": make_event(body="redacted" * 100).content}
    )
    result = await memory.retrieve("r1", event, public, channel_id="b")
    assert "updated B" in str(result.messages)
    assert len(provider.calls) == 3


async def test_reader_and_room_are_part_of_cache_identity() -> None:
    provider = MockAIProvider()
    memory = CompactingMemory(SlidingWindowMemory(), provider, 100, min_events=1)
    ctx, event = context("same"), make_event(body="now")
    for room, channel in (("r1", "a"), ("r1", "b"), ("r2", "b")):
        await memory.retrieve(room, event, ctx, channel_id=channel)
    assert len(provider.calls) == 3


async def test_advancing_window_regenerates_summary() -> None:
    provider = MockAIProvider()
    memory = CompactingMemory(SlidingWindowMemory(), provider, 100, min_events=1)
    ctx, event = context("old"), make_event(body="now")
    await memory.retrieve("r1", event, ctx)
    ctx.recent_events.extend([make_event(body="new" * 100) for _ in range(2)])
    await memory.retrieve("r1", event, ctx)
    assert len(provider.calls) == 2


async def test_clear_and_ttl_invalidate_summary() -> None:
    provider = MockAIProvider()
    memory = CompactingMemory(
        SlidingWindowMemory(), provider, 100, min_events=1, summary_cache_ttl_seconds=0
    )
    ctx, event = context("old"), make_event(body="now")
    await memory.retrieve("r1", event, ctx)
    await memory.retrieve("r1", event, ctx)
    assert len(provider.calls) == 2
    await memory.clear("r1")
    assert not memory._summary_cache


async def test_cache_is_bounded() -> None:
    memory = CompactingMemory(SlidingWindowMemory(), MockAIProvider(), 100, min_events=1)
    with patch("roomkit.memory.compacting._MAX_CACHE_ENTRIES", 2):
        for room in ("a", "b", "c"):
            await memory.retrieve(room, make_event(body="now"), context("old"))
    assert len(memory._summary_cache) == 2
    assert ("a", None) not in memory._summary_cache


async def test_clear_during_generation_does_not_restore_cache() -> None:
    started, finish = asyncio.Event(), asyncio.Event()
    provider = MockAIProvider()

    async def generate(ctx: AIContext) -> AIResponse:
        started.set()
        await finish.wait()
        return AIResponse(content="erased")

    provider.generate = generate  # type: ignore[assignment]
    memory = CompactingMemory(SlidingWindowMemory(), provider, 100, min_events=1)
    task = asyncio.create_task(memory.retrieve("r1", make_event(body="now"), context("old")))
    await started.wait()
    await memory.clear("r1")
    finish.set()
    await task
    assert not memory._summary_cache
