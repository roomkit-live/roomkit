"""The preparation and consumption of a real AI turn own separate scopes."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.memory.base import MemoryResult
from roomkit.memory.mock import MockMemoryProvider
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools import (
    current_response_metadata,
    current_tool_actor_id,
    current_tool_chain_depth,
    current_tool_room,
    current_tool_room_id,
    tool_turn_context,
)
from tests.conftest import make_event
from tests.test_framework import SimpleChannel
from tests.tool_loop_modes import read_reply


def _binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [{"name": "check", "description": "Check the turn"}]},
    )


def _event() -> RoomEvent:
    return make_event(room_id="r1", channel_id="sms1", participant_id="alice").model_copy(
        update={"chain_depth": 2}
    )


def _check_turn(room: Room) -> None:
    assert current_tool_room() is room
    assert current_tool_room_id() == room.id
    assert current_tool_actor_id() == "alice"
    assert current_tool_chain_depth() == 3
    assert current_response_metadata() is not None


async def test_preparation_installs_the_turn_after_ingestion(streaming: bool) -> None:
    room = Room(id="r1", metadata={"tenant": "acme"})
    seen: list[str] = []

    class Memory(MockMemoryProvider):
        async def ingest(
            self, room_id: str, event: RoomEvent, *, channel_id: str | None = None
        ) -> None:
            assert current_tool_room_id() == "outer"
            seen.append("ingest")

        async def retrieve(
            self,
            room_id: str,
            current_event: RoomEvent,
            context: RoomContext,
            *,
            channel_id: str | None = None,
        ) -> MemoryResult:
            _check_turn(room)
            assert current_response_metadata()["configured"] is True
            seen.append("memory")
            return MemoryResult()

    async def configure(binding: ChannelBinding, context: RoomContext) -> None:
        _check_turn(room)
        current_response_metadata()["configured"] = True
        seen.append("config")

    ch = AIChannel(
        "ai1",
        provider=MockAIProvider(responses=["done"], streaming=streaming),
        memory=Memory(),
        config_provider=configure,
    )
    with tool_turn_context(room_id="outer", actor_id="parent", chain_depth=1):
        output = await ch.on_event(_event(), _binding(), RoomContext(room=room))
        assert current_tool_room_id() == "outer"
        result = await read_reply(output)
        assert current_tool_room_id() == "outer"
        assert current_tool_actor_id() == "parent"
    assert seen == ["ingest", "config", "memory"]
    assert result.metadata["configured"] is True
    await ch.close()


async def test_a_failed_preparation_restores_its_caller() -> None:
    async def configure(binding: ChannelBinding, context: RoomContext) -> None:
        _check_turn(context.room)
        raise ValueError("configuration failed")

    ch = AIChannel("ai1", provider=MockAIProvider(), config_provider=configure)
    with tool_turn_context(room_id="outer", actor_id="parent", chain_depth=1):
        with pytest.raises(ValueError, match="configuration failed"):
            await ch.on_event(_event(), _binding(), RoomContext(room=Room(id="r1")))
        assert current_tool_room_id() == "outer"
        assert current_tool_actor_id() == "parent"
        assert current_tool_chain_depth() == 1
    await ch.close()


async def test_stream_consumption_does_not_revive_preparation_children(streaming: bool) -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("out"))
    await kit.create_room(room_id="probe")
    await kit.attach_channel("probe", "out")
    room = Room(id="r1")
    during_stream = asyncio.Event()
    after_stream = asyncio.Event()
    children: list[asyncio.Task[tuple[int, int]]] = []

    async def child(gate: asyncio.Event, label: str) -> tuple[int, int]:
        await gate.wait()
        _check_turn(room)
        await kit.deliver("probe", label)
        event = next(
            event
            for event in await kit.store.list_events("probe")
            if getattr(event.content, "body", None) == label
        )
        return current_tool_chain_depth(), event.chain_depth

    async def configure(binding: ChannelBinding, context: RoomContext) -> None:
        children.append(asyncio.create_task(child(during_stream, "preparation child")))

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        _check_turn(room)
        during_stream.set()
        # The preparation child runs while this independent stream scope is live.
        assert await children[0] == (3, 0)
        children.append(asyncio.create_task(child(after_stream, "stream child")))
        await kit.deliver("probe", "live stream")
        return "ok"

    ch = AIChannel(
        "ai1",
        provider=MockAIProvider(
            ai_responses=[
                AIResponse(content="", tool_calls=[AIToolCall(id="tc1", name="check")]),
                AIResponse(content="done"),
            ],
            streaming=streaming,
        ),
        config_provider=configure,
        tool_handler=handler,
    )
    try:
        output = await ch.on_event(_event(), _binding(), RoomContext(room=room))
        await asyncio.wait_for(read_reply(output), 5)
        after_stream.set()
        assert await asyncio.wait_for(children[1], 5) == (3, 0)
        events = await kit.store.list_events("probe")
        assert (
            next(
                event.chain_depth
                for event in events
                if getattr(event.content, "body", None) == "live stream"
            )
            == 3
        )
        assert current_tool_room() is None
    finally:
        during_stream.set()
        after_stream.set()
        await asyncio.gather(*children, return_exceptions=True)
        await ch.close()
        await kit.close()


async def test_turn_observers_keep_the_readable_turn(streaming: bool) -> None:
    kit = RoomKit()
    seen: list[tuple[str, str | None, str | None, int]] = []

    async def observe(event: Any, context: RoomContext) -> None:
        seen.append(
            (
                type(event).__name__,
                current_tool_room_id(),
                current_tool_actor_id(),
                current_tool_chain_depth(),
            )
        )
        assert current_tool_room() is not None

    for trigger in (
        HookTrigger.BEFORE_AI_GENERATION,
        HookTrigger.ON_TOOL_CALL,
        HookTrigger.ON_AI_RESPONSE,
    ):
        kit.hook(trigger, execution=HookExecution.ASYNC)(observe)
    kit.register_channel(SimpleChannel("sms1"))

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "ok"

    kit.register_channel(
        AIChannel(
            "ai1",
            tool_handler=handler,
            provider=MockAIProvider(
                ai_responses=[
                    AIResponse(content="", tool_calls=[AIToolCall(id="tc1", name="check")]),
                    AIResponse(content="done"),
                ],
                streaming=streaming,
            ),
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", metadata=_binding().metadata)
    await kit.send_event("r1", "sms1", TextContent(body="go"), chain_depth=2)
    await kit.close()
    assert len(seen) == 3  # Preparation, one call, one completed response.
    assert {value[1:] for value in seen} == {("r1", None, 3)}


async def test_cancelling_a_stream_expires_its_child_depth(streaming: bool) -> None:
    from roomkit.core._reentrance import chain_floor

    started = asyncio.Event()
    finish = asyncio.Event()
    children: list[asyncio.Task[tuple[int, int]]] = []

    async def child() -> tuple[int, int]:
        await finish.wait()
        return current_tool_chain_depth(), chain_floor()

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        children.append(asyncio.create_task(child()))
        started.set()
        await asyncio.Future()
        return "unreachable"

    ch = AIChannel(
        "ai1",
        tool_handler=handler,
        provider=MockAIProvider(
            ai_responses=[AIResponse(content="", tool_calls=[AIToolCall(id="tc1", name="check")])],
            streaming=streaming,
        ),
    )
    output = await ch.on_event(_event(), _binding(), RoomContext(room=Room(id="r1")))
    draining = asyncio.create_task(read_reply(output))
    try:
        await asyncio.wait_for(started.wait(), 5)
        draining.cancel()
        with pytest.raises(asyncio.CancelledError):
            await draining
        finish.set()
        assert await asyncio.wait_for(children[0], 5) == (3, 0)
        assert current_tool_room_id() is None
    finally:
        draining.cancel()
        finish.set()
        await asyncio.gather(draining, *children, return_exceptions=True)
        await ch.close()
