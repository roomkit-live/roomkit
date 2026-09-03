"""``ON_AI_RESPONSE`` carries the turn's text as a readable transcript.

A tool call cuts the model's text into segments, one MESSAGE each. Joined with
nothing between them, the hook's ``response_content`` read as one run-on
sentence (``first.Working``), and the non-streaming loop reported the last
segment alone. The paths now report the same thing, and the streaming one
reports what the room saw, not the raw text the dedup filter withheld.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import TextContent
from roomkit.models.room import Room
from roomkit.models.tool_call import (
    RESPONSE_SEGMENT_SEPARATOR,
    AIResponseEvent,
    response_transcript,
)
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.test_framework import SimpleChannel

_TOOLS = [{"name": "search", "description": "Search"}]

_ROUNDS = [
    AIResponse(
        content="Let me look.",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="tc1", name="search", arguments={})],
    ),
    AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="tc2", name="search", arguments={})],
    ),
    AIResponse(content="Done.", finish_reason="stop"),
]


async def _tool_handler(name: str, args: dict[str, Any]) -> str:
    return "result"


async def _report(provider: MockAIProvider) -> AIResponseEvent:
    """One turn straight through ``on_event``; returns what the hook received."""
    captured: list[AIResponseEvent] = []

    async def after_response(event: AIResponseEvent) -> None:
        captured.append(event)

    ch = AIChannel("ai1", provider=provider, tool_handler=_tool_handler)
    ch._after_response_hook = after_response
    output = await ch.on_event(
        make_event(body="go", channel_id="sms1"),
        ChannelBinding(
            channel_id="ai1",
            room_id="room-1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
            metadata={"tools": _TOOLS},
        ),
        RoomContext(room=Room(id="room-1")),
    )
    if output.response_stream is not None:
        async for _ in output.response_stream:
            pass
    assert len(captured) == 1
    return captured[0]


def test_the_transcript_keeps_the_separator_between_segments_only() -> None:
    segments, content = response_transcript(["", "Let me look.", "", "Done.", ""])
    assert segments == ["Let me look.", "Done."]
    assert content == "Let me look." + RESPONSE_SEGMENT_SEPARATOR + "Done."
    assert response_transcript([]) == ([], "")


@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "non-streaming"])
async def test_segments_are_separated_at_tool_calls(streaming: bool) -> None:
    # The silent round (a call with no text before it) adds no separator.
    event = await _report(MockAIProvider(ai_responses=list(_ROUNDS), streaming=streaming))
    assert event.segments == ["Let me look.", "Done."]
    assert event.response_content == "Let me look.\n\nDone."


@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "non-streaming"])
async def test_a_turn_without_a_tool_call_is_unchanged(streaming: bool) -> None:
    event = await _report(
        MockAIProvider(
            ai_responses=[AIResponse(content="Hello there.", finish_reason="stop")],
            streaming=streaming,
        )
    )
    assert event.response_content == "Hello there."
    assert event.segments == ["Hello there."]


@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "non-streaming"])
async def test_a_turn_without_text_reports_nothing(streaming: bool) -> None:
    event = await _report(
        MockAIProvider(
            ai_responses=[AIResponse(content="", finish_reason="stop")], streaming=streaming
        )
    )
    assert event.segments == []
    assert event.response_content == ""


async def test_the_streaming_transcript_reports_what_the_room_saw() -> None:
    """A model that replays its previous round's text after a tool call.

    The stream withholds the replayed prefix from the room (the dedup
    filter), so the room persists ``Done.`` for the second round while the
    model's raw text was ``Let me look.Done.``. The transcript follows the
    room, not the raw text — that is the one the hook's consumers read.
    """
    kit = RoomKit()
    seen: list[AIResponseEvent] = []

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def observe(event: AIResponseEvent, ctx: Any) -> None:
        seen.append(event)

    responses = [
        AIResponse(
            content="Let me look.",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id="tc1", name="search", arguments={})],
        ),
        AIResponse(content="Let me look.Done.", finish_reason="stop"),
    ]
    sms = SimpleChannel("sms1")
    ai = AIChannel(
        "ai1",
        provider=MockAIProvider(ai_responses=responses, streaming=True),
        tool_handler=_tool_handler,
    )
    kit.register_channel(sms)
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel(
        "r1", "ai1", category=ChannelCategory.INTELLIGENCE, metadata={"tools": _TOOLS}
    )
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )
    await asyncio.sleep(0.05)

    persisted = [
        e.content.body
        for e in await kit.store.list_events("r1")
        if e.source.channel_id == "ai1" and getattr(e.content, "body", None)
    ]
    assert persisted == ["Let me look.", "Done."]
    assert len(seen) == 1
    assert seen[0].segments == persisted
    assert seen[0].response_content == "Let me look.\n\nDone."
    await kit.close()
