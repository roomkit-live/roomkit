"""Per-turn tool bounds preserve defaults and remain private to each turn."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import AIChannelTurnConfig
from roomkit.channels.ai import AIChannel
from roomkit.models import ChannelBinding, Room, RoomContext
from roomkit.providers.ai.base import AIResponse, AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.human_input import HumanInputToolHandler
from tests.conftest import make_event
from tests.test_channels.test_ai_router import _Picks
from tests.test_tool_policy_exemptions import _binding, _calls, _tool_payload, _turn
from tests.test_tool_timeout import BOUND, LONGER, SLOW, _calling, _Hung
from tests.tool_loop_modes import LoopRun, respond


@pytest.mark.parametrize("given", [None, {}, {"other": 0.2}])
async def test_absent_names_keep_the_channel_per_tool_bound(
    streaming: bool, given: dict[str, float | None] | None
) -> None:
    provider, hung = _calling("slow", streaming), _Hung(delay=LONGER)

    async def resolve(_binding: ChannelBinding, _context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(
            tools=[AITool(**SLOW)], tool_handler=hung.handler, tool_timeouts=given
        )

    channel = AIChannel(
        "ai1",
        provider=provider,
        config_provider=resolve,
        tool_timeout_seconds=BOUND,
        tool_timeouts={"slow": None},
    )
    await _turn(channel)
    assert not hung.cancelled
    assert _tool_payload(provider.calls[1], "slow") == {"ok": True}


async def test_a_turn_bound_applies_to_a_tool_that_waits_by_design(streaming: bool) -> None:
    ask = AITool(name="ask_user", description="Ask", parameters={"type": "object"})
    human = HumanInputToolHandler({"ask_user"}, timeout=LONGER, tool_definitions=[ask])
    provider = _calling("ask_user", streaming)

    async def resolve(_binding: ChannelBinding, _context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(tool_timeouts={"ask_user": BOUND})

    channel = AIChannel(
        "ai1", provider=provider, human_input_handler=human, config_provider=resolve
    )
    await _turn(channel)
    assert "ToolTimeoutError" in _tool_payload(provider.calls[1], "ask_user")["error"]


async def test_simultaneous_turns_keep_independent_bounds(streaming: bool) -> None:
    call = _calls(("slow", {}))[0]
    provider = MockAIProvider(
        ai_responses=[call, call, AIResponse(content="done"), AIResponse(content="done")],
        streaming=streaming,
    )
    started = {room: asyncio.Event() for room in ("unbounded", "bounded")}
    release = asyncio.Event()
    cancelled: set[str] = set()

    async def resolve(_binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        room = context.room.id

        async def handler(_name: str, _arguments: dict[str, Any]) -> str:
            started[room].set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.add(room)
                raise
            return "finished"

        return AIChannelTurnConfig(
            tools=[AITool(**SLOW)],
            tool_handler=handler,
            tool_timeouts={"slow": None if room == "unbounded" else BOUND},
        )

    channel = AIChannel("ai1", provider=provider, config_provider=resolve)

    async def run(room: str) -> LoopRun:
        binding = _binding().model_copy(update={"room_id": room})
        return await respond(
            channel, make_event(room_id=room), binding, RoomContext(room=Room(id=room))
        )

    tasks: list[asyncio.Task[LoopRun]] = []
    try:
        tasks.append(asyncio.create_task(run("unbounded")))
        await asyncio.wait_for(started["unbounded"].wait(), 1.0)
        tasks.append(asyncio.create_task(run("bounded")))
        bounded = await asyncio.wait_for(tasks[1], 1.0)
        assert bounded.calls[0].failed
        assert "ToolTimeoutError" in str(bounded.calls[0].result)
        assert cancelled == {"bounded"}
        assert not tasks[0].done()
        release.set()
        unbounded = await asyncio.wait_for(tasks[0], 1.0)
        assert not unbounded.calls[0].failed
        assert unbounded.calls[0].result == "finished"
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_the_turn_budget_is_validated_before_its_tool_bounds(streaming: bool) -> None:
    provider = _calling("slow", streaming)

    async def resolve(_binding: ChannelBinding, _context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(tool_timeouts={"slow": 0.0})

    channel = AIChannel("ai1", provider=provider, config_provider=resolve)
    binding = _binding().model_copy(update={"metadata": {"turn_budget_tokens": 0}})
    with pytest.raises(ValueError, match="turn_budget_tokens must be a positive"):
        await respond(channel, make_event(), binding, RoomContext(room=Room(id="r1")))
    assert provider.calls == []


async def test_the_router_is_asked_before_a_turn_bound_is_refused(streaming: bool) -> None:
    provider = _calling("slow", streaming)
    router = _Picks(provider)

    async def resolve(_binding: ChannelBinding, _context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(tool_timeouts={"slow": 0.0})

    channel = AIChannel("ai1", router=router, config_provider=resolve)
    with pytest.raises(ValueError, match="tool timeout for 'slow' must be positive or None"):
        await _turn(channel)
    assert len(router.requests) == 1
    assert provider.calls == []
