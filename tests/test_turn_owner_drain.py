"""Owner cleanup belongs to its turn and shares its cancellation bound."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit.channels._tool_registry import _TURN_OWNER_TOOLS
from roomkit.models.channel import ChannelOutput
from roomkit.providers.ai.base import AITool
from roomkit.turns import TurnClient, TurnExecutor, TurnTimeouts
from roomkit.turns.base import OwnerTools, TurnBackend
from roomkit.turns.wire import OwnerTool, ToolCall, TurnRequest
from tests.bounds import CUT, expire, opened_with_cut
from tests.test_turns_backend import ScriptedChannel, _binding, _context, _event


class _HeldTools(OwnerTools):
    def __init__(self) -> None:
        self.started: dict[str, asyncio.Event] = {}
        self.cleaning: dict[str, asyncio.Event] = {}
        self.release: dict[str, asyncio.Event] = {}

    def prepare(self, room_id: str) -> None:
        self.started[room_id] = asyncio.Event()
        self.cleaning[room_id] = asyncio.Event()
        self.release[room_id] = asyncio.Event()

    def declared(self, channel_id: str, room_id: str) -> list[OwnerTool]:
        return [OwnerTool(tool=AITool(name="hold", description="Hold."), waits_by_design=True)]

    async def serve(self, request: TurnRequest, call: ToolCall) -> str:
        room_id = request.room_id
        self.started[room_id].set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cleaning[room_id].set()
            await self.release[room_id].wait()
        return "never"


async def _script(channel: ScriptedChannel) -> AsyncIterator[Any]:
    yield "ready"
    tools = _TURN_OWNER_TOOLS.get()
    assert tools is not None
    serve = tools.entries["hold"].serve
    assert serve is not None
    await serve({})
    yield "never"


def _client(backend: TurnBackend, tools: _HeldTools) -> TurnClient:
    channel = ScriptedChannel(_script)
    timeouts = TurnTimeouts(cancel=CUT)
    return TurnClient(
        backend,
        executor=TurnExecutor({"agent": channel}.get, hosts={"agent"}, timeouts=timeouts),
        tools=tools,
        timeouts=timeouts,
    )


async def _reading(client: TurnClient, tools: _HeldTools, room_id: str) -> tuple[Any, Any]:
    tools.prepare(room_id)
    event = _event().model_copy(update={"room_id": room_id})
    binding = _binding().model_copy(update={"room_id": room_id})
    context = _context()
    context = context.model_copy(update={"room": context.room.model_copy(update={"id": room_id})})
    output: ChannelOutput = await client.run(event, binding, context)
    stream = output.response_stream
    assert await anext(stream) == "ready"
    reading = asyncio.create_task(anext(stream))
    await tools.started[room_id].wait()
    return stream, reading


async def test_end_received_with_owner_cleanup_held_keeps_its_original_cancel_bound(
    turn_backend: TurnBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = _HeldTools()
    client = _client(turn_backend, tools)
    async with asyncio.timeout(5):
        bounds = opened_with_cut(monkeypatch)
        stream, reading = await _reading(client, tools, "room")
        try:
            reading.cancel("first cut")
            await tools.cleaning["room"].wait()
            while not stream._turn._ended:
                await asyncio.sleep(0)
            assert not reading.done()
            reading.cancel("second cut")
            await asyncio.sleep(0)
            assert len(bounds) == 1
            expire(bounds)
            with pytest.raises(asyncio.CancelledError):
                await reading
            assert stream.end_owed
            assert not tools.release["room"].is_set()
        finally:
            tools.release["room"].set()
            await client.close()


async def test_cancelling_one_turn_neither_cuts_nor_waits_for_another_owner_call(
    turn_backend: TurnBackend,
) -> None:
    tools = _HeldTools()
    client = _client(turn_backend, tools)
    async with asyncio.timeout(5):
        first, reading = await _reading(client, tools, "first")
        other, other_reading = await _reading(client, tools, "other")
        try:
            reading.cancel()
            await tools.cleaning["first"].wait()
            tools.release["first"].set()
            with pytest.raises(asyncio.CancelledError):
                await reading
            assert not first.end_owed
            assert not other_reading.done()
            assert not tools.cleaning["other"].is_set()
            other_reading.cancel()
            await tools.cleaning["other"].wait()
            tools.release["other"].set()
            with pytest.raises(asyncio.CancelledError):
                await other_reading
            assert not other.end_owed
        finally:
            for release in tools.release.values():
                release.set()
            await client.close()
