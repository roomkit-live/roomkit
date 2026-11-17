"""Public kit and provider boundaries for structured task plans."""

from __future__ import annotations

from typing import Any

from roomkit import RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


def plan(tasks: Any, call_id: str = "p1") -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name="plan_tasks", arguments={"tasks": tasks})],
    )


def answer(text: str = "Done.") -> AIResponse:
    return AIResponse(content=text, finish_reason="stop")


def message(body: str = "Plan it.") -> InboundMessage:
    return InboundMessage(channel_id="ws", sender_id="alice", content=TextContent(body=body))


async def room(
    rounds: list[AIResponse], streaming: bool, *, rooms: tuple[str, ...] = ("r",), **kwargs: Any
) -> tuple[RoomKit, AIChannel, MockAIProvider]:
    model = MockAIProvider(ai_responses=rounds, streaming=streaming)
    channel = AIChannel("agent", provider=model, enable_planning=True, system_prompt="Fixed.")
    kit = RoomKit(**kwargs)
    kit.register_channel(SimpleChannel("ws"))
    kit.register_channel(channel)
    for name in rooms:
        await kit.create_room(room_id=name)
        await kit.attach_channel(name, "ws")
        await kit.attach_channel(name, "agent")
    return kit, channel, model
