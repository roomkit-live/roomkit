"""The channel uses the same mixed definitions and handler precedence."""

from __future__ import annotations

from typing import Any

from tests.conftest import make_event
from tests.test_tool_protocol import FakeTool
from tests.tool_loop_modes import respond

from roomkit import AIChannel, ChannelBinding, ChannelCategory, ChannelType, Room, RoomContext
from roomkit.providers.ai import AIResponse, AIToolCall, AIToolResultPart, MockAIProvider


async def test_raw_schemas_and_tool_objects_reach_their_own_handlers() -> None:
    asked: list[str] = []

    async def explicit(name: str, arguments: dict[str, Any]) -> str:
        asked.append(name)
        if name == "lookup":
            return str(arguments["q"])
        return '{"error": "Unknown tool: echo"}'

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                tool_calls=[
                    AIToolCall(id="c1", name="lookup", arguments={"q": "raw result"}),
                    AIToolCall(id="c2", name="echo"),
                ],
            ),
            AIResponse(content="done"),
        ]
    )
    channel = AIChannel(
        "ai",
        provider=provider,
        tool_handler=explicit,
        tools=[
            {
                "name": "lookup",
                "description": "Raw schema",
                "tags": ["raw"],
                "changes_state": True,
                "parameters": {"type": "object"},
            },
            FakeTool("echo", "object result"),
        ],
    )
    binding = ChannelBinding(
        channel_id="ai",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    try:
        await respond(channel, make_event(body="go"), binding, RoomContext(room=Room(id="r1")))
    finally:
        await channel.close()

    assert asked == ["lookup", "echo"]
    assert len(provider.calls) == 2
    assert [(tool.name, tool.tags, tool.changes_state) for tool in provider.calls[0].tools] == [
        ("lookup", ["raw"], True),
        ("echo", [], False),
        ("read_stored_result", [], False),
    ]
    results = [
        part
        for message in provider.calls[1].messages
        if isinstance(message.content, list)
        for part in message.content
        if isinstance(part, AIToolResultPart)
    ]
    assert [(part.tool_call_id, part.result, part.is_error) for part in results] == [
        ("c1", "raw result", False),
        ("c2", "object result", False),
    ]
