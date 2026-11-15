"""A resolved catalogue follows its preparation and consumption contexts."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.channels import AIChannel
from roomkit.memory import MemoryResult, MockMemoryProvider
from roomkit.models import ChannelBinding, Room, RoomContext, RoomEvent
from roomkit.providers.ai import AIResponse, AITool, AIToolCall, MockAIProvider
from roomkit.tools import current_tool_allowed_names, tool_turn_context
from tests.test_ai_turn_context_lifetime import _binding, _event
from tests.tool_loop_modes import respond


@pytest.mark.parametrize("names", [[], ["check"]], ids=["empty", "resolved"])
async def test_resolution_distinguishes_unresolved_from_empty(
    names: list[str], streaming: bool
) -> None:
    seen: list[tuple[str, set[str] | None]] = []

    class Memory(MockMemoryProvider):
        async def retrieve(
            self,
            room_id: str,
            current_event: RoomEvent,
            context: RoomContext,
            *,
            channel_id: str | None = None,
        ) -> MemoryResult:
            seen.append(("memory", current_tool_allowed_names()))
            return MemoryResult()

    async def configure(binding: ChannelBinding, context: RoomContext) -> None:
        seen.append(("config", current_tool_allowed_names()))

    channel = AIChannel(
        "ai1",
        provider=MockAIProvider(responses=["done"], streaming=streaming),
        config_provider=configure,
        memory=Memory(),
        tools=[AITool(name=name, description=name) for name in names],
        tool_search=False,
    )
    try:
        with tool_turn_context(tools=[AITool(name="outer", description="outer")]):
            binding = _binding().model_copy(update={"metadata": {}})
            await respond(channel, _event(), binding, RoomContext(room=Room(id="r1")))
            assert current_tool_allowed_names() == {"outer"}
        assert current_tool_allowed_names() is None
        assert seen == [("config", None), ("memory", set(names))]
    finally:
        await channel.close()


async def test_preparation_keeps_its_base_but_shares_round_declarations(streaming: bool) -> None:
    resume = asyncio.Event()
    children: list[asyncio.Task[set[str] | None]] = []
    calls: list[set[str] | None] = []

    async def read_later() -> set[str] | None:
        await resume.wait()
        return current_tool_allowed_names()

    async def configure(binding: ChannelBinding, context: RoomContext) -> None:
        children.append(asyncio.create_task(read_later()))

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        calls.append(current_tool_allowed_names())
        if len(calls) == 1:
            children.append(asyncio.create_task(read_later()))
        else:
            resume.set()
        return "done"

    async def after(event: Any) -> None:
        event.withdraw("book", "read_stored_result")

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", tool_calls=[AIToolCall(id=cid, name="check", arguments={})])
            for cid in ("c1", "c2")
        ]
        + [AIResponse(content="done")],
        streaming=streaming,
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        config_provider=configure,
        tools=[AITool(name=name, description=name) for name in ("check", "book")],
        tool_handler=handler,
        tool_search=False,
    )
    channel._after_tool_round_hook = after
    try:
        await respond(channel, _event(), _binding(), RoomContext(room=Room(id="r1")))
        prepared, consumed = await asyncio.wait_for(asyncio.gather(*children), 2)
        assert calls == [{"check", "book", "read_stored_result"}, {"check"}]
        assert prepared == {"check", "book", "read_stored_result"}
        assert consumed == {"check"}
        assert current_tool_allowed_names() is None
    finally:
        for child in children:
            child.cancel()
        await asyncio.gather(*children, return_exceptions=True)
        await channel.close()
