"""Cancel a suspended tool, drain it, and continue using the same RoomKit.

Run with: uv run python examples/cancel_delivery.py
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from roomkit import Agent, ChannelCategory, InboundMessage, RoomKit, TextContent, WebSocketChannel
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider

logger = logging.getLogger(__name__)


async def main() -> None:
    started, finished = asyncio.Event(), asyncio.Event()

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        started.set()
        try:
            await asyncio.Future()
        finally:
            finished.set()
        return "unreachable"

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="", tool_calls=[AIToolCall(id="call", name="lookup", arguments={})]
            ),
            AIResponse(content="Ready for the next turn"),
        ]
    )
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("input"))
    kit.register_channel(
        Agent(
            "agent",
            provider=provider,
            tool_handler=lookup,
            tools=[
                AITool(name="lookup", description="Example lookup", parameters={"type": "object"}),
            ],
        )
    )
    await kit.create_room(room_id="example")
    await kit.attach_channel("example", "input")
    await kit.attach_channel("example", "agent", category=ChannelCategory.INTELLIGENCE)
    message = InboundMessage(channel_id="input", sender_id="user", content=TextContent(body="Go"))
    try:
        result = await kit.process_inbound(message, room_id="example", defer_delivery=True)
        await started.wait()
        if result.delivery is not None:
            await result.delivery.cancel(reason="hangup")
            logger.info(
                "Cancelled: %s; tool finished: %s", result.cancellation_reason, finished.is_set()
            )
        next_turn = await kit.process_inbound(message, room_id="example")
        logger.info("Next turn: %s", next_turn.response_events[-1].content)
    finally:
        await kit.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
