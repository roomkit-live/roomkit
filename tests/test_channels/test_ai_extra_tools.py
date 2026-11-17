"""The public host-tool list is independent of each turn's offered tools."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import Agent, AIChannel, ChannelBinding, Room, RoomContext
from roomkit.models import ChannelCategory, ChannelType
from roomkit.providers.ai import AITool, MockAIProvider
from tests.conftest import make_event
from tests.test_tool_protocol import FakeTool
from tests.tool_loop_modes import respond


@pytest.fixture(params=[AIChannel, Agent], ids=["channel", "agent"])
def channel_type(request: pytest.FixtureRequest) -> type[AIChannel]:
    return request.param


@pytest.mark.parametrize("tools", [None, []], ids=["default", "empty"])
def test_no_host_tools_gives_an_empty_list(
    channel_type: type[AIChannel], tools: list[AITool] | None
) -> None:
    channel = channel_type("ai", provider=MockAIProvider(), tools=tools)
    assert channel.extra_tools == []


def test_host_definitions_keep_constructor_order_and_fields(
    channel_type: type[AIChannel],
) -> None:
    typed = AITool(
        name="typed",
        description="typed tool",
        tags=["one"],
        changes_state=True,
        defer_loading=True,
    )
    tools: list[Any] = [
        FakeTool("local", "ok"),
        typed,
        {"name": "raw", "description": "raw tool", "tags": ["two"]},
    ]
    channel = channel_type("ai", provider=MockAIProvider(), tools=tools)
    definitions = channel.extra_tools
    assert [definition.name for definition in definitions] == ["local", "typed", "raw"]
    assert definitions[0].model_dump() == {
        "name": "local",
        "description": "Fake local",
        "parameters": {"type": "object", "properties": {}},
        "tags": [],
        "changes_state": False,
        "defer_loading": False,
    }
    assert definitions[1] is typed
    assert definitions[1].model_dump() == {
        "name": "typed",
        "description": "typed tool",
        "parameters": {},
        "tags": ["one"],
        "changes_state": True,
        "defer_loading": True,
    }
    assert definitions[2].model_dump() == {
        "name": "raw",
        "description": "raw tool",
        "parameters": {},
        "tags": ["two"],
        "changes_state": False,
        "defer_loading": False,
    }


def test_the_input_and_returned_list_containers_do_not_change_host_tools(
    channel_type: type[AIChannel],
) -> None:
    tool = AITool(name="read", description="read tool")
    given = [tool]
    channel = channel_type("ai", provider=MockAIProvider(), tools=given)
    first = channel.extra_tools
    second = channel.extra_tools
    assert first is not second
    assert first is not given
    given.clear()
    first.clear()
    second.append(AITool(name="added", description="not installed"))
    assert channel.extra_tools == [tool]


async def test_a_returned_definition_remains_live_for_the_next_turn(
    channel_type: type[AIChannel],
    streaming: bool,
) -> None:
    provider = MockAIProvider(responses=["ok"], streaming=streaming)
    channel = channel_type(
        "ai",
        provider=provider,
        tools=[AITool(name="read", description="original", defer_loading=True)],
        tool_search=False,
    )
    binding = ChannelBinding(
        room_id="r1",
        channel_id="ai",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    context = RoomContext(room=Room(id="r1"))
    try:
        await respond(channel, make_event(body="first"), binding, context)
        assert [tool.name for tool in provider.calls[0].tools] == ["read_stored_result"]
        returned = channel.extra_tools[0]
        returned.description = "updated through the public getter"
        returned.parameters["type"] = "object"
        returned.tags.append("updated")
        returned.changes_state = True
        returned.defer_loading = False
        await respond(channel, make_event(body="second"), binding, context)
    finally:
        await channel.close()
    assert len(provider.calls) == 2
    offered = next(tool for tool in provider.calls[1].tools if tool.name == "read")
    assert offered.description == "updated through the public getter"
    assert offered.parameters == {"type": "object"}
    assert offered.tags == ["updated"]
    assert offered.changes_state is True
    assert offered.defer_loading is False
    assert channel.extra_tools[0] is returned


async def test_binding_tools_are_offered_without_becoming_host_tools(
    channel_type: type[AIChannel],
    streaming: bool,
) -> None:
    provider = MockAIProvider(responses=["ok"], streaming=streaming)
    host = AITool(name="host", description="constructor tool")
    channel = channel_type("ai", provider=provider, tools=[host], tool_search=False)
    binding = ChannelBinding(
        room_id="r1",
        channel_id="ai",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [{"name": "room_tool", "description": "binding tool"}]},
    )
    try:
        await respond(channel, make_event(body="hello"), binding, RoomContext(room=Room(id="r1")))
    finally:
        await channel.close()
    assert len(provider.calls) == 1
    assert [tool.name for tool in provider.calls[0].tools] == [
        "room_tool",
        "host",
        "read_stored_result",
    ]
    assert channel.extra_tools == [host]
