"""Around a streamed row's hooks, a stream reads the room's status once
(RMK-302, RFC §5.1); the row's commit reads the room under its lock (RMK-1286).

The status gate holds at every row a stream writes: a room closed mid-stream
takes no further row. Around its hooks, which run without the room lock, the
run reads the status at its first row, and again only once the kit closed or
archived a room since.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from roomkit import HookResult, HookTrigger, RoomTimers
from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.core.lanes import DeliveryCascade
from roomkit.core.mixins._streaming_segments import LaneSink
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType, RoomStatus
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel
from tests.test_streamed_chain_depth import _Voice

Closer = Callable[[RoomKit], Awaitable[Any]]


def _answers(rounds: int) -> list[AIResponse]:
    steps = [
        AIResponse(
            content=f"Step {i}.",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id=f"t{i}", name="lookup", arguments={})],
        )
        for i in range(rounds)
    ]
    return [*steps, AIResponse(content="Answer.", finish_reason="stop")]


async def _room(rounds: int, during_tool: Closer | None = None) -> RoomKit:
    kit = RoomKit()

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        if during_tool is not None:
            await during_tool(kit)
        return "ok"

    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(_Voice("voice1"))
    provider = MockAIProvider(streaming=True, ai_responses=_answers(rounds))
    tools = [AITool(name="lookup", description="d")]
    kit.register_channel(AIChannel("ai1", provider=provider, tool_handler=lookup, tools=tools))
    for room_id in ("r1", "r2"):
        await kit.create_room(room_id=room_id)
    for channel_id in ("sms1", "voice1"):
        await kit.attach_channel("r1", channel_id)
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    return kit


async def _say(kit: RoomKit) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go")),
        room_id="r1",
    )


def _count_status_reads(kit: RoomKit) -> list[str]:
    reads: list[str] = []
    read_status = kit._room_refuses_writes

    async def counting(room_id: str) -> bool:
        reads.append(room_id)
        return await read_status(room_id)

    kit._room_refuses_writes = counting  # type: ignore[method-assign]
    return reads


async def _ai_rows(kit: RoomKit) -> list[EventType]:
    events = await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))
    return [e.type for e in events if e.source.channel_id == "ai1"]


async def test_a_streamed_turn_reads_the_status_once_around_its_hooks() -> None:
    kit = await _room(rounds=4)
    reads = _count_status_reads(kit)

    await _say(kit)

    assert reads == ["r1"]
    assert (await _ai_rows(kit)).count(EventType.TOOL_CALL_END) == 4
    await kit.close()


async def test_a_row_reads_the_room_once() -> None:
    """Each tool round adds three rows, each reading the room at its commit
    (RMK-1286); a read that grew with the rows would not pass."""

    async def reads(rounds: int) -> int:
        kit = await _room(rounds=rounds)
        count = 0
        get_room = kit.store.get_room

        async def counting(room_id: str) -> Any:
            nonlocal count
            count += 1
            return await get_room(room_id)

        kit.store.get_room = counting  # type: ignore[method-assign]
        await _say(kit)
        await kit.close()
        return count

    assert await reads(4) - await reads(1) <= 3 * 3


async def test_another_room_closing_costs_one_read_and_blocks_nothing() -> None:
    async def close_the_other_room(kit: RoomKit) -> None:
        await kit.close_room("r2")

    kit = await _room(rounds=1, during_tool=close_the_other_room)
    reads = _count_status_reads(kit)

    await _say(kit)

    assert reads == ["r1", "r1"]
    assert (await _ai_rows(kit))[-1] == EventType.MESSAGE
    await kit.close()


async def _close(kit: RoomKit) -> None:
    await kit.close_room("r1")


async def _archive(kit: RoomKit) -> None:
    await kit.archive_room("r1")


async def _time_out(kit: RoomKit) -> None:
    past = datetime.now(UTC) - timedelta(seconds=10)
    await kit.set_room_timers(
        "r1",
        RoomTimers(inactive_after_seconds=0, closed_after_seconds=0, last_activity_at=past),
    )
    await kit.check_room_timers("r1")


@pytest.mark.parametrize("closer", [_close, _archive, _time_out], ids=lambda c: c.__name__)
async def test_every_close_refuses_the_rows_after_it(closer: Closer) -> None:
    """Refused, not recorded as blocked either, whatever a hook would say."""
    kit = await _room(rounds=1, during_tool=closer)

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="block_ends")
    async def block_ends(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.type == EventType.TOOL_CALL_END:
            return HookResult.block("no ends")
        return HookResult.allow()

    await _say(kit)

    rows = await _ai_rows(kit)
    assert EventType.TOOL_CALL_START in rows
    assert EventType.TOOL_CALL_END not in rows
    await kit.close()


async def test_a_close_during_the_hooks_refuses_the_row() -> None:
    kit = await _room(rounds=1)

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="closer")
    async def closer(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.type == EventType.TOOL_CALL_START:
            await kit.close_room("r1")
        return HookResult.allow()

    await _say(kit)

    assert EventType.TOOL_CALL_START not in await _ai_rows(kit)
    await kit.close()


async def test_a_close_before_the_first_row_is_read() -> None:
    """The context a stream was built from may predate a close the kit never
    counted (the store written directly): the first row reads the status."""
    kit = await _room(rounds=0)
    context = await kit._build_context("r1")
    room = await kit.store.get_room("r1")
    await kit.store.update_room(room.model_copy(update={"status": RoomStatus.CLOSED}))
    sink = LaneSink(
        kit,
        room_id="r1",
        context=context,
        cascade=DeliveryCascade("r1", reentry_budget=10),
    )
    row = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="ai1", channel_type="ai"),
        content=TextContent(body="late"),
    )

    assert await sink.commit(row, exclude=None) is None
    await kit.close()


async def _set_timers_due(kit: RoomKit) -> None:
    past = datetime.now(UTC) - timedelta(seconds=10)
    await kit.set_room_timers(
        "r1",
        RoomTimers(inactive_after_seconds=0, closed_after_seconds=0, last_activity_at=past),
    )


async def test_a_timer_due_at_a_row_s_commit_applies_there() -> None:
    """RFC §5.1: the row's commit is a section that reads the status, so the
    transitions due apply first; nothing else checked the timers here."""
    kit = await _room(rounds=1, during_tool=_set_timers_due)

    await _say(kit)

    room = await kit.store.get_room("r1")
    assert room is not None and room.status == RoomStatus.CLOSED
    assert EventType.TOOL_CALL_END not in await _ai_rows(kit)
    await kit.close()


async def test_a_refusal_the_commit_found_spares_the_next_row_s_hooks() -> None:
    """A status changed elsewhere moves no close count: the commit that reads
    it spares the rows after it their hooks."""

    async def close_elsewhere(kit: RoomKit) -> None:
        room = await kit.store.get_room("r1")
        await kit.store.update_room(room.model_copy(update={"status": RoomStatus.CLOSED}))

    kit = await _room(rounds=1, during_tool=close_elsewhere)
    seen: list[EventType] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="watch")
    async def watch(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if event.source.channel_id == "ai1":
            seen.append(event.type)
        return HookResult.allow()

    await _say(kit)

    assert seen == [EventType.MESSAGE, EventType.TOOL_CALL_START, EventType.TOOL_CALL_END]
    await kit.close()
