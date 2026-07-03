"""Tool-call ephemeral events fire on the realtime bus alongside thinking.

Regression tests for the bug where ``_publish_tool_event`` existed on
``AIEventsMixin`` but had no call sites: subscribers saw the model's
reasoning stream live (THINKING_*) while tool calls were invisible until
a page reload. All three generation paths must publish TOOL_CALL_START /
TOOL_CALL_END: the streaming internal tool loop, the streaming external
handler path, and the non-streaming tool loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.tools.external import PolicyExternalToolHandler
from tests.test_framework import SimpleChannel

_TOOLS = [{"name": "search", "description": "Search"}]


async def _run_turn(kit: RoomKit, ai: AIChannel) -> list[EphemeralEvent]:
    """Wire a room, subscribe to its realtime bus, run one inbound turn."""
    sms = SimpleChannel("sms1")
    kit.register_channel(sms)
    kit.register_channel(ai)

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel(
        "r1", "ai1", category=ChannelCategory.INTELLIGENCE, metadata={"tools": _TOOLS}
    )

    received: list[EphemeralEvent] = []

    async def on_event(ev: EphemeralEvent) -> None:
        received.append(ev)

    await kit.realtime.subscribe_to_room("r1", on_event)

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )

    # InMemoryRealtime dispatches subscriber callbacks via background tasks;
    # yield once so they run before we inspect the list.
    await asyncio.sleep(0.05)
    return received


def _tool_events(
    received: list[EphemeralEvent],
) -> tuple[list[EphemeralEvent], list[EphemeralEvent]]:
    starts = [e for e in received if e.type == EphemeralEventType.TOOL_CALL_START]
    ends = [e for e in received if e.type == EphemeralEventType.TOOL_CALL_END]
    return starts, ends


async def test_streaming_tool_loop_publishes_tool_events() -> None:
    """Internal-handler streaming loop: one START + one END per round."""

    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        return f"result of {name}"

    provider = MockAIProvider(
        streaming=True,
        ai_responses=[
            AIResponse(
                content="",
                thinking="round one reasoning",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="tc1", name="search", arguments={"q": "x"})],
            ),
            AIResponse(content="Done.", thinking="round two reasoning", finish_reason="stop"),
        ],
    )
    kit = RoomKit()
    ai = AIChannel("ai1", provider=provider, tool_handler=tool_handler, thinking_budget=4096)

    received = await _run_turn(kit, ai)
    starts, ends = _tool_events(received)

    assert len(starts) == 1
    assert starts[0].data["tool_calls"] == [
        {"id": "tc1", "name": "search", "arguments": {"q": "x"}}
    ]
    assert starts[0].data["round"] == 0
    assert starts[0].channel_id == "ai1"

    assert len(ends) == 1
    assert ends[0].data["tool_calls"] == [
        {"id": "tc1", "name": "search", "result": "result of search"}
    ]
    assert ends[0].data["round"] == 0
    assert isinstance(ends[0].data["duration_ms"], int)

    # The bug scenario: reasoning AND tool events both reach subscribers.
    assert any(e.type == EphemeralEventType.THINKING_START for e in received)

    await kit.close()


async def test_streaming_tool_end_result_preview_capped() -> None:
    """END payloads carry a bounded result preview (500 chars)."""

    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        return "x" * 600

    provider = MockAIProvider(
        streaming=True,
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="tc1", name="search", arguments={})],
            ),
            AIResponse(content="Done.", finish_reason="stop"),
        ],
    )
    kit = RoomKit()
    ai = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

    received = await _run_turn(kit, ai)
    _, ends = _tool_events(received)

    assert len(ends) == 1
    assert len(ends[0].data["tool_calls"][0]["result"]) == 500

    await kit.close()


async def test_non_streaming_tool_loop_publishes_tool_events() -> None:
    """Non-streaming tool loop publishes the same START/END pairs."""

    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        return "42"

    provider = MockAIProvider(
        streaming=False,
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="tc1", name="calc", arguments={"n": 1})],
            ),
            AIResponse(content="The answer is 42.", finish_reason="stop"),
        ],
    )
    kit = RoomKit()
    ai = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

    received = await _run_turn(kit, ai)
    starts, ends = _tool_events(received)

    assert len(starts) == 1
    assert starts[0].data["tool_calls"][0]["name"] == "calc"
    assert starts[0].data["round"] == 0
    assert len(ends) == 1
    assert ends[0].data["tool_calls"][0]["result"] == "42"

    await kit.close()


async def test_external_handler_streaming_publishes_tool_events() -> None:
    """External-handler path (provider executed the tool) publishes START/END."""
    provider = MockAIProvider(
        streaming=True,
        ai_responses=[
            AIResponse(
                content="I ran the tool.",
                finish_reason="stop",
                tool_calls=[
                    AIToolCall(
                        id="tc1",
                        name="Bash",
                        arguments={"cmd": "ls", "_result": "file.txt"},
                    )
                ],
            ),
        ],
    )
    kit = RoomKit()
    ai = AIChannel("ai1", provider=provider, external_tool_handler=PolicyExternalToolHandler())

    received = await _run_turn(kit, ai)
    starts, ends = _tool_events(received)

    assert len(starts) == 1
    # `_result` is stripped from the arguments before publishing.
    assert starts[0].data["tool_calls"] == [
        {"id": "tc1", "name": "Bash", "arguments": {"cmd": "ls"}}
    ]
    assert len(ends) == 1
    assert ends[0].data["tool_calls"] == [{"id": "tc1", "name": "Bash", "result": "file.txt"}]

    await kit.close()
