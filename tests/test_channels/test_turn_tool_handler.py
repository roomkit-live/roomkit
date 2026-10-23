"""A turn's host tools run through the handler that resolved their surface."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import AIChannel, AIChannelTurnConfig
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond

LOOKUP = AITool(name="lookup", description="Read this member's records", parameters={})


def _call(name: str = "lookup", **arguments: Any) -> AIResponse:
    return AIResponse(
        content="",
        tool_calls=[AIToolCall(id=f"call-{name}", name=name, arguments=arguments)],
        finish_reason="tool_calls",
    )


async def _turn(channel: AIChannel, room_id: str) -> LoopRun:
    binding = ChannelBinding(
        room_id=room_id,
        channel_id=channel.channel_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    context = RoomContext(room=Room(id=room_id), bindings=[binding])
    return await respond(channel, make_event(room_id=room_id), binding, context)


async def test_concurrent_turns_keep_their_resolved_handler(streaming: bool) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def resolve(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        async def serve(name: str, arguments: dict[str, Any]) -> str:
            calls.append(binding.room_id)
            if binding.room_id == "first":
                entered.set()
                await asyncio.wait_for(release.wait(), 1)
            else:
                release.set()
            return binding.room_id

        return AIChannelTurnConfig(tools=[LOOKUP], tool_handler=serve)

    fallback = AsyncMock(return_value="wrong attach")
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[_call(), _call(), AIResponse(content="done"), AIResponse(content="done")],
    )
    channel = AIChannel("agent", provider=provider, config_provider=resolve)
    try:
        async with asyncio.TaskGroup() as group:
            first = group.create_task(_turn(channel, "first"))
            await asyncio.wait_for(entered.wait(), 1)
            channel.tool_handler = fallback
            second = group.create_task(_turn(channel, "second"))
        assert first.result().calls[0].result == "first"
        assert second.result().calls[0].result == "second"
        assert calls == ["first", "second"]
        fallback.assert_not_awaited()
    finally:
        await channel.close()


@pytest.mark.parametrize("config", [None, AIChannelTurnConfig(tools=[LOOKUP])])
async def test_no_override_keeps_the_default_without_leaking_a_previous_turn(
    streaming: bool, config: AIChannelTurnConfig | None
) -> None:
    selected = AsyncMock(return_value="per-turn")
    fallback = AsyncMock(return_value="default")
    configs = iter([AIChannelTurnConfig(tools=[LOOKUP], tool_handler=selected), config])

    async def resolve(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig | None:
        return next(configs)

    provider = MockAIProvider(
        streaming=streaming, ai_responses=[_call(), AIResponse(content="done")]
    )
    channel = AIChannel(
        "agent", provider=provider, tools=[LOOKUP], tool_handler=fallback, config_provider=resolve
    )
    try:
        assert (await _turn(channel, "room")).calls[0].result == "per-turn"
        assert (await _turn(channel, "room")).calls[0].result == "default"
        selected.assert_awaited_once()
        fallback.assert_awaited_once()
    finally:
        await channel.close()


async def test_channel_tools_stay_local_and_unoffered_tools_stay_refused(streaming: bool) -> None:
    selected = AsyncMock(return_value="per-turn")
    fallback = AsyncMock(return_value="wrong attach")

    async def resolve(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(tools=[LOOKUP], tool_handler=selected)

    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            _call("find_tools", query="lookup"),
            _call(),
            _call("other_member_tool"),
            AIResponse(content="done"),
        ],
    )
    channel = AIChannel(
        "agent",
        provider=provider,
        tool_handler=fallback,
        config_provider=resolve,
        tool_search=True,
    )
    try:
        run = await _turn(channel, "room")
        assert [call.name for call in run.calls] == ["find_tools", "lookup", "other_member_tool"]
        assert not run.calls[0].failed
        assert run.calls[1].result == "per-turn"
        assert run.calls[2].failed
        selected.assert_awaited_once_with("lookup", {})
        fallback.assert_not_awaited()
    finally:
        await channel.close()
