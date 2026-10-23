"""Two rooms share a channel; each turn's host tools use its own backend.

Run: uv run python examples/turn_tool_handler.py
No credentials or external services are needed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from roomkit import AIChannel, AIChannelTurnConfig, InboundMessage, RoomKit, TextContent
from roomkit.channels import WebSocketChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider

logger = logging.getLogger(__name__)
LOOKUP = AITool(name="lookup", description="Read the current room's records", parameters={})


async def resolve(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
    async def serve(name: str, arguments: dict[str, Any]) -> str:
        result = f"Records belonging to {context.room.id}"
        logger.info("%s: %s", name, result)
        return result

    return AIChannelTurnConfig(tools=[LOOKUP], tool_handler=serve)


async def main() -> None:
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                tool_calls=[AIToolCall(id="lookup", name="lookup", arguments={})],
                finish_reason="tool_calls",
            ),
            AIResponse(content="Read complete."),
        ]
    )
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("chat"))
    kit.register_channel(AIChannel("agent", provider=provider, config_provider=resolve))
    try:
        for room_id in ("room-a", "room-b"):
            await kit.create_room(room_id=room_id)
            await kit.attach_channel(room_id, "chat")
            await kit.attach_channel(room_id, "agent")
        for room_id in ("room-b", "room-a"):
            await kit.process_inbound(
                InboundMessage(
                    channel_id="chat", sender_id="member", content=TextContent(body="Read")
                ),
                room_id=room_id,
            )
    finally:
        await kit.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
