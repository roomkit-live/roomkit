"""The kit's independent local-executor lifecycle (RFC §10.6.7–§10.6.8)."""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import AIChannel, HookTrigger, InboundMessage, RoomKit, TextContent
from roomkit.core._turn_scope import turns_run_under
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.turns import InMemoryTurnBackend, TurnClient
from tests.test_framework import SimpleChannel
from tests.test_turns_channel_control import _Agent, _Recording, _turn


class ClosingChannel(SimpleChannel):
    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.closes = 0

    async def close(self) -> None:
        self.closes += 1


async def test_closing_the_local_executor_keeps_the_kits_borrowed_channels() -> None:
    """ExecutorChannels.close only closes channels its executor built."""
    async with RoomKit() as kit:
        channel = ClosingChannel("channel")
        kit.register_channel(channel)
        executor = kit.turn_executor
        assert not executor.closed
        assert executor.declares("channel")
        assert executor.takes("channel")
        await executor.close()
        await executor.close()
        assert executor.closed
        assert not executor.declares("channel")
        assert not executor.takes("channel")
        assert executor.hosts_channel("channel")
        assert channel.closes == 0
        await kit.create_room(room_id="still-open")
        await kit.attach_channel("still-open", "channel")
    assert channel.closes == 1


async def test_nested_inbound_has_a_separate_turn_scope() -> None:
    scopes: dict[str, tuple[str, ...]] = {}
    errors: dict[str, str | None] = {}
    async with RoomKit() as kit:
        kit.register_channel(SimpleChannel("sms"))
        provider = MockAIProvider(responses=["nested", "outer"])
        kit.register_channel(AIChannel("agent", provider=provider))
        for name in ("r1", "r2"):
            await kit.create_room(room_id=name)
            await kit.attach_channel(name, "sms")
            await kit.attach_channel(name, "agent")

        async def ask(room: str) -> Any:
            result = await kit.process_inbound(
                InboundMessage(
                    channel_id="sms", sender_id="alice", content=TextContent(body="hello")
                ),
                room_id=room,
            )
            errors[room] = None if result.error is None else str(result.error)
            return result

        @kit.turn_executor.hook(HookTrigger.BEFORE_AI_GENERATION)
        async def lifecycle(event: Any, context: Any) -> Any:
            room = context.room.id
            scopes[room] = turns_run_under()
            if room == "r1":
                await ask("r2")
            else:
                await kit.turn_executor.close()
            return event

        await asyncio.wait_for(ask("r1"), 5)
        await asyncio.sleep(0)
        assert len(scopes["r1"]) == len(scopes["r2"]) == 1
        assert scopes["r1"][0] not in scopes["r2"]
        assert errors["r1"] == "The turn ended before its output"
        assert await kit.store.list_events("r2")


async def test_nested_turn_client_preserves_both_ancestors_when_closing() -> None:
    scopes: dict[str, tuple[str, ...]] = {}
    completed: list[str] = []

    class NestedAgent(_Agent):
        async def on_event(self, event: Any, binding: Any, context: Any) -> Any:
            scopes[event.room_id] = turns_run_under()
            if event.room_id == "r1":
                await _turn(client, "r2")
            else:
                await executor.close()
            result = await super().on_event(event, binding, context)
            completed.append(event.room_id)
            return result

    backend = InMemoryTurnBackend()
    executor = _Recording(NestedAgent(), "local")
    client = TurnClient(backend)
    await backend.serve(executor)
    try:
        await asyncio.wait_for(_turn(client, "r1"), 5)
        assert scopes["r2"][:1] == scopes["r1"]
        assert len(scopes["r1"]) == 1 and len(scopes["r2"]) == 2
        assert completed == ["r2", "r1"]
    finally:
        await executor.close()
        await backend.close()


async def test_later_external_close_cuts_a_turn_previously_spared_by_its_own_close() -> None:
    async with RoomKit() as kit:
        kit.register_channel(SimpleChannel("sms"))
        kit.register_channel(AIChannel("agent", provider=MockAIProvider(responses=["unused"])))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms")
        await kit.attach_channel("r1", "agent")
        closed_inside = asyncio.Event()
        stayed = asyncio.Event()
        cleaned = asyncio.Event()

        @kit.turn_executor.hook(HookTrigger.BEFORE_AI_GENERATION)
        async def lifecycle(event: Any, context: Any) -> Any:
            await kit.turn_executor.close()
            closed_inside.set()
            try:
                await stayed.wait()
            finally:
                cleaned.set()
            return event

        running = asyncio.create_task(
            kit.process_inbound(
                InboundMessage(
                    channel_id="sms", sender_id="alice", content=TextContent(body="hello")
                ),
                room_id="r1",
            )
        )
        await asyncio.wait_for(closed_inside.wait(), 2)
        assert not running.done()
        await asyncio.wait_for(kit.turn_executor.close(), 2)
        result = await asyncio.wait_for(running, 2)
        assert cleaned.is_set()
        assert result.error is not None


async def test_a_close_from_an_owner_served_tool_spares_its_turn() -> None:
    """Owner calls retain turn ancestry even though they expose no TurnHost."""
    from roomkit import current_turn_host
    from roomkit.channels._tool_registry import orchestration_tool
    from tests.test_turns_owner_tools import PROBE, _agent, _call, _kit, _say

    kit = await _kit(_agent("agent", _call("probe", {}), "done"), attach=("agent",))
    seen: list[bool] = []

    async def serve(arguments: dict[str, Any]) -> str:
        assert current_turn_host() is None
        assert len(turns_run_under()) == 1
        await kit.turn_executor.close()
        seen.append(kit.turn_executor.closed)
        return "closed"

    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve), room_id="r1", owner=object()
    )
    try:
        result = await asyncio.wait_for(_say(kit, "go"), 2)
        assert result.error is None
        events = await kit.store.list_events("r1")
        assert any(getattr(event.content, "body", None) == "done" for event in events)
        assert seen == [True]
    finally:
        await kit.close()
