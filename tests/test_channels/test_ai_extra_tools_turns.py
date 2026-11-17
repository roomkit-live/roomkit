"""Live host definitions keep their dispatch and tool-source boundaries."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from roomkit import (
    Agent,
    AIChannel,
    AIChannelTurnConfig,
    ChannelBinding,
    ChannelCategory,
    ChannelType,
    InboundMessage,
    Room,
    RoomContext,
    RoomKit,
    TextContent,
)
from roomkit.providers.ai import (
    AIContext,
    AIResponse,
    AITool,
    AIToolCall,
    AIToolResultPart,
    MockAIProvider,
)
from roomkit.skills import Skill, SkillMetadata, SkillRegistry
from roomkit.tools import HumanInputToolHandler
from tests.conftest import make_event
from tests.test_framework import SimpleChannel
from tests.test_tool_protocol import FakeTool
from tests.tool_loop_modes import respond

pytestmark = pytest.mark.parametrize("channel_type", [AIChannel, Agent], ids=["channel", "agent"])


def _binding(metadata: dict[str, Any] | None = None) -> ChannelBinding:
    return ChannelBinding(
        room_id="r1",
        channel_id="ai",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata=metadata or {},
    )


async def _respond(channel: AIChannel, metadata: dict[str, Any] | None = None) -> None:
    reply = await respond(
        channel, make_event(body="hello"), _binding(metadata), RoomContext(room=Room(id="r1"))
    )
    assert reply.text == "done"


@pytest.mark.parametrize("self_served", [False, True], ids=["host-handler", "tool-object"])
async def test_renaming_a_definition_does_not_rename_its_handler(
    channel_type: type[AIChannel],
    streaming: bool,
    self_served: bool,
) -> None:
    called: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        called.append(name)
        return f"served:{name}"

    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(content="", tool_calls=[AIToolCall(id="c1", name="renamed")]),
            AIResponse(content="done"),
        ],
    )
    definition = (
        FakeTool("original", "object answer")
        if self_served
        else AITool(name="original", description="host tool")
    )
    channel = channel_type(
        "ai",
        provider=provider,
        tools=[definition],
        tool_handler=None if self_served else handler,
        tool_search=False,
    )
    channel.extra_tools[0].name = "renamed"
    try:
        await _respond(channel)
    finally:
        await channel.close()
    assert [tool.name for tool in provider.calls[0].tools] == ["renamed", "read_stored_result"]
    results = [
        part
        for message in provider.calls[1].messages
        if isinstance(message.content, list)
        for part in message.content
        if isinstance(part, AIToolResultPart)
    ]
    assert len(results) == 1
    assert results[0].is_error is self_served
    if self_served:
        assert called == []
        assert results[0].outcome == "unserved"
        assert results[0].result == '{"error": "No handler for tool renamed"}'
    else:
        assert called == ["renamed"]
        assert results[0].result == "served:renamed"


@pytest.mark.parametrize("collision", ["binding", "host", "builtin"])
async def test_a_renamed_host_tool_obeys_the_turns_collision_rules(
    channel_type: type[AIChannel],
    streaming: bool,
    collision: str,
) -> None:
    provider = MockAIProvider(responses=["done"], streaming=streaming)
    first = AITool(name="first", description="first host")
    second = AITool(name="second", description="second host")
    channel = channel_type("ai", provider=provider, tools=[first, second], tool_search=False)
    target = {"binding": "from_binding", "host": "second", "builtin": "read_stored_result"}
    channel.extra_tools[0].name = target[collision]
    try:
        await _respond(channel, {"tools": [{"name": "from_binding", "description": "binding"}]})
    finally:
        await channel.close()
    tools = provider.calls[0].tools
    expected = ["from_binding", "second", "read_stored_result"]
    assert [tool.name for tool in tools] == expected
    if collision == "binding":
        assert tools[0].description == "first host"
    else:
        assert tools[0].description == "binding"
    assert tools[1].description == "second host"
    assert tools[2].description != "first host"
    assert [tool.name for tool in channel.extra_tools] == [target[collision], "second"]


async def test_retained_nested_values_write_until_their_field_is_replaced(
    channel_type: type[AIChannel],
    streaming: bool,
) -> None:
    provider = MockAIProvider(responses=["done"], streaming=streaming)
    channel = channel_type(
        "ai",
        provider=provider,
        tools=[
            AITool(
                name="read",
                description="read",
                parameters={"nested": {"version": 1}},
                tags=["old"],
            )
        ],
        tool_search=False,
    )
    definition = channel.extra_tools[0]
    old_parameters, old_tags = definition.parameters, definition.tags
    old_parameters["nested"]["version"] = 2
    old_tags.append("live")
    try:
        await _respond(channel)
        first = next(tool for tool in provider.calls[0].tools if tool.name == "read")
        assert first.parameters == {"nested": {"version": 2}}
        assert first.tags == ["old", "live"]
        definition.parameters = {"replacement": 3}
        definition.tags = ["replacement"]
        old_parameters["nested"]["version"] = 4
        old_tags.append("detached")
        await _respond(channel)
        second = next(tool for tool in provider.calls[1].tools if tool.name == "read")
        assert second.parameters == {"replacement": 3}
        assert second.tags == ["replacement"]
    finally:
        await channel.close()


async def test_a_provider_already_generating_reads_a_live_host_definition(
    channel_type: type[AIChannel],
    streaming: bool,
) -> None:
    ready, release = asyncio.Event(), asyncio.Event()
    seen: list[str] = []

    class PausedProvider(MockAIProvider):
        async def generate(self, context: AIContext) -> AIResponse:
            tool = next(tool for tool in context.tools if tool.name == "read")
            seen.append(tool.description)
            ready.set()
            await release.wait()
            seen.append(tool.description)
            return await super().generate(context)

    provider = PausedProvider(responses=["done"], streaming=streaming)
    channel = channel_type(
        "ai",
        provider=provider,
        tools=[
            AITool(name="read", description="before"),
        ],
        tool_search=False,
    )
    turn = asyncio.create_task(_respond(channel))
    try:
        await asyncio.wait_for(ready.wait(), 3)
        channel.extra_tools[0].description = "while generating"
        release.set()
        await asyncio.wait_for(turn, 3)
    finally:
        release.set()
        if not turn.done():
            turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await channel.close()
    assert seen == ["before", "while generating"]


async def test_other_tool_sources_never_enter_the_host_tool_list(
    channel_type: type[AIChannel],
    streaming: bool,
    tmp_path: Path,
) -> None:
    async def configured(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(tools=[AITool(name="configured", description="per turn")])

    registry = SkillRegistry()
    registry.add(
        Skill(
            metadata=SkillMetadata(name="guide", description="Guide"),
            instructions="Be helpful",
            path=tmp_path,
        )
    )
    human = HumanInputToolHandler(
        {"ask"}, tool_definitions=[AITool(name="ask", description="Ask")]
    )
    provider = MockAIProvider(responses=["done"], streaming=streaming)
    host = AITool(name="host", description="constructor")
    channel = channel_type(
        "ai",
        provider=provider,
        tools=[host],
        config_provider=configured,
        skills=registry,
        human_input_handler=human,
        tool_search=False,
    )
    async with RoomKit() as kit:
        kit.register_channel(SimpleChannel("sms"))
        kit.register_channel(channel)
        kit.setup_task_tools(channel)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms")
        await kit.attach_channel("r1", "ai", metadata={"tools": [{"name": "binding"}]})
        assert channel.extra_tools == [host]
        await kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id="user", content=TextContent(body="hello")),
            room_id="r1",
        )
        assert channel.extra_tools == [host]
    assert len(provider.calls) == 1
    names = [tool.name for tool in provider.calls[0].tools]
    assert names[:4] == ["configured", "host", "task_status", "cancel_task"]
    assert "ask" in names and "activate_skill" in names and "read_stored_result" in names
    assert "binding" not in names
