"""An async human-input observer keeps the local AI turn it observes."""

from __future__ import annotations

from typing import Any

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools import (
    current_response_metadata,
    current_tool_actor_id,
    current_tool_chain_depth,
    current_tool_room,
    current_tool_room_id,
)
from roomkit.tools.human_input import HumanInputToolHandler
from tests.test_framework import SimpleChannel


async def test_a_human_input_observer_keeps_the_turn(streaming: bool) -> None:
    kit = RoomKit()
    human = HumanInputToolHandler(
        {"ask"}, timeout=5, tool_definitions=[AITool(name="ask", description="Ask the person")]
    )
    seen: list[tuple[Any, ...]] = []

    async def configure(binding: Any, context: Any) -> None:
        current_response_metadata()["configured"] = True

    @kit.hook(HookTrigger.ON_USER_INPUT_REQUIRED, execution=HookExecution.ASYNC)
    async def observe(event: Any, context: Any) -> None:
        room, record = current_tool_room(), current_response_metadata()
        seen.append(
            (
                current_tool_room_id(),
                room.id if room else None,
                current_tool_actor_id(),
                current_tool_chain_depth(),
                record.get("configured") if record is not None else None,
            )
        )
        human.handler.resolve(event.pending_id, "yes")

    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(
        AIChannel(
            "ai1",
            human_input_handler=human,
            config_provider=configure,
            provider=MockAIProvider(
                ai_responses=[
                    AIResponse(content="", tool_calls=[AIToolCall(id="tc1", name="ask")]),
                    AIResponse(content="done"),
                ],
                streaming=streaming,
            ),
        )
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1")
    try:
        await kit.send_event("r1", "sms1", TextContent(body="go"), chain_depth=2)
    finally:
        await kit.close()
    assert seen == [("r1", "r1", None, 3, True)]
