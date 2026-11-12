"""Local channel-call cancellation, inherited scope and detach cleanup."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from typing import Any

import pytest

from roomkit import ChannelOutput, TextContent, current_turn_host
from roomkit.core._turn_scope import turns_run_under
from roomkit.turns._executor_rooms import ExecutorRooms
from roomkit.turns._rooms import room_state_key
from roomkit.turns.wire import PreviousTurn
from tests.test_framework import SimpleChannel
from tests.test_turns_channel_control import _Agent, _kit


async def test_cancelled_detach_forgets_stored_state() -> None:
    kit, agent = await _kit()
    try:
        await kit.channel_call("agent", "room", "set", model="kept")
        assert room_state_key("agent") in (await kit.get_room("room")).metadata
        entered = asyncio.Event()

        async def interrupted(room_id: str) -> None:
            entered.set()
            await asyncio.Event().wait()

        agent.on_room_detached = interrupted
        task = asyncio.create_task(kit.detach_channel("room", "agent"))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
        assert room_state_key("agent") not in (await kit.get_room("room")).metadata
    finally:
        await kit.close()


@pytest.mark.in_process(
    "its turn calls the kit it holds, from the context an in-memory turn shares with "
    "the owner (RFC §10.6.5); across Redis a turn has none"
)
async def test_channel_call_can_close_executor_under_its_calling_turn() -> None:
    kit, agent = await _kit()
    kit.register_channel(SimpleChannel("sms"))
    await kit.attach_channel("room", "sms")
    finished = []
    seen = []

    async def verb(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        seen.append((len(turns_run_under()), current_turn_host() is not None))
        await kit.turn_executor.close()
        return "closed"

    async def on_event(event: Any, binding: Any, context: Any) -> ChannelOutput:
        assert await kit.channel_call("agent", "room", "close") == "closed"
        finished.append(True)
        return ChannelOutput.empty()

    agent.channel_call = verb
    agent.on_event = on_event
    try:
        await asyncio.wait_for(kit.send_event("room", "sms", TextContent(body="hello")), 2)
        assert seen == [(1, True)]
        assert finished == [True]
    finally:
        await kit.close()


async def test_detach_callback_can_reattach_its_channel() -> None:
    kit, agent = await _kit()
    completed = []

    async def detached(room_id: str) -> None:
        await kit.attach_channel(room_id, "agent")
        completed.append(True)

    agent.on_room_detached = detached
    try:
        await asyncio.wait_for(kit.detach_channel("room", "agent"), 2)
        assert completed == [True]
        assert await kit.store.get_binding("room", "agent") is not None
    finally:
        await kit.close()


async def test_cancelled_stale_drop_is_retried() -> None:
    channel = _Agent()
    started = asyncio.Event()
    attempts = 0

    async def get_channel(channel_id: str) -> _Agent:
        return channel

    async def interrupted(room_id: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            started.set()
            await asyncio.Event().wait()

    channel.on_room_detached = interrupted
    rooms = ExecutorRooms("local", get_channel, lambda _: nullcontext())
    rooms.enter("agent", "room", "t1", None)
    stale = rooms.enter("agent", "room", "t2", PreviousTurn(turn_id="elsewhere"))
    assert stale
    dropping = asyncio.create_task(rooms._ready(channel, "agent", "room", {}, stale=stale))
    await asyncio.wait_for(started.wait(), 2)
    dropping.cancel()
    await asyncio.gather(dropping, return_exceptions=True)
    stale = rooms.enter("agent", "room", "t3", PreviousTurn(turn_id="t2"))
    assert stale
    await rooms._ready(channel, "agent", "room", {}, stale=stale)
    assert attempts == 2
