"""MiniMax with a tool: a round's reasoning goes back to the model, on MiniMax's
API or on a server of your own.

MiniMax asks for its model's reasoning back in a tool loop, as "essential for
achieving the model's best performance". RoomKit sends none to a generic Chat
Completions server, so the class is the choice: :class:`MiniMaxAIProvider` sends
it in ``reasoning_content``, for as long as a turn lasts. A round that did not
think sends nothing.

Requires:
    pip install roomkit[openai]

Environment:
    MINIMAX_API_KEY   your key (https://platform.minimax.io), "unused" on your own server
    MINIMAX_BASE_URL  a server of your own, http://gpu:8000/v1 for one (default: MiniMax's API)
    MINIMAX_MODEL     model id (default: MiniMax-M3.1-Flash); MiniMaxAI/MiniMax-M3 on vLLM for one

Run with:
    MINIMAX_API_KEY=... uv run python examples/minimax_ai.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import require_env, setup_logging

from roomkit import (
    AIChannel,
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai import AITool
from roomkit.providers.minimax import MiniMaxAIProvider, MiniMaxConfig

WEATHER = AITool(
    name="get_weather",
    description="The weather in a city today.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


async def get_weather(name: str, arguments: dict[str, Any]) -> str:
    return f"Sunny in {arguments.get('city', 'that city')}, 12 degrees."


async def main() -> None:
    env = require_env("MINIMAX_API_KEY")
    settings: dict[str, Any] = {
        "api_key": env["MINIMAX_API_KEY"],
        "model": os.environ.get("MINIMAX_MODEL", "MiniMax-M3.1-Flash"),
        "reasoning_effort": "low",
    }
    if base_url := os.environ.get("MINIMAX_BASE_URL"):
        settings["base_url"] = base_url
    provider = MiniMaxAIProvider(MiniMaxConfig(**settings))

    kit = RoomKit()
    user = WebSocketChannel("user")
    kit.register_channel(user)
    kit.register_channel(
        AIChannel(
            "assistant",
            provider=provider,
            system_prompt="Use the weather tool, then answer in a few words.",
            tools=[WEATHER],
            tool_handler=get_weather,
        )
    )

    async def on_event(_connection: str, event: RoomEvent) -> None:
        if event.source.channel_id == "assistant" and isinstance(event.content, TextContent):
            print(f"Assistant: {event.content.body}")

    user.register_connection("me", on_event, room_id="weather")
    await kit.create_room(room_id="weather")
    await kit.attach_channel("weather", "user")
    await kit.attach_channel("weather", "assistant", category=ChannelCategory.INTELLIGENCE)

    for text in ("What is the weather in Montreal today?", "And in Quebec?"):
        print(f"\nYou: {text}")
        await kit.process_inbound(
            InboundMessage(channel_id="user", sender_id="me", content=TextContent(body=text))
        )
    await kit.close()


if __name__ == "__main__":
    setup_logging("minimax_ai")
    asyncio.run(main())
