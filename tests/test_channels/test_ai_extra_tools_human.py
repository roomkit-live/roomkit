"""Renaming a host definition follows the person's intercepted names."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import Agent, AIChannel, ChannelBinding, Room, RoomContext
from roomkit.models import ChannelCategory, ChannelType
from roomkit.providers.ai import AIResponse, AITool, AIToolCall, MockAIProvider
from roomkit.tools import HumanInputToolHandler
from tests.conftest import make_event
from tests.tool_loop_modes import respond


@pytest.mark.parametrize("channel_type", [AIChannel, Agent], ids=["channel", "agent"])
@pytest.mark.parametrize("to_human", [False, True], ids=["leave-human", "enter-human"])
async def test_renaming_a_host_definition_follows_human_interception(
    channel_type: type[AIChannel], streaming: bool, to_human: bool
) -> None:
    called: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        called.append(name)
        return "host answer"

    human = HumanInputToolHandler({"ask"}, timeout=3)
    before, after = ("read", "ask") if to_human else ("ask", "read")
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(content="", tool_calls=[AIToolCall(id="c1", name=after)]),
            AIResponse(content="done"),
        ],
    )
    channel = channel_type(
        "ai",
        provider=provider,
        tools=[AITool(name=before, description="host definition")],
        tool_handler=handler,
        human_input_handler=human,
        tool_search=False,
    )
    channel.extra_tools[0].name = after
    binding = ChannelBinding(
        room_id="r1",
        channel_id="ai",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )

    async def answer() -> None:
        if not to_human:
            return
        while not human.handler.pending:
            await asyncio.sleep(0)
        pending = next(iter(human.handler.pending.values()))
        assert pending.tool_call_id == "c1"
        assert human.handler.resolve(pending.pending_id, "human answer")

    work = asyncio.gather(
        respond(channel, make_event(body="hello"), binding, RoomContext(room=Room(id="r1"))),
        answer(),
    )
    try:
        await asyncio.wait_for(work, 3)
    finally:
        if not work.done():
            work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        await channel.close()
    results = [
        part
        for message in provider.calls[1].messages
        if isinstance(message.content, list)
        for part in message.content
        if getattr(part, "type", None) == "tool_result"
    ]
    assert len(results) == 1
    assert results[0].result == ("human answer" if to_human else "host answer")
    assert called == ([] if to_human else ["read"])
