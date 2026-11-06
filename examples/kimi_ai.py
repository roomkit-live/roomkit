"""Kimi with a tool: a round's reasoning goes back to the model, on Kimi's API or
on a server of your own.

Kimi answers 400 to a tool round without its ``reasoning_content``, and says the
model needs it to follow its own reasoning. RoomKit sends none to a generic Chat
Completions server, so the class is the choice: :class:`KimiAIProvider` sends it,
for as long as a turn lasts.

The example runs two turns: the first calls a tool, and the second reads that
round again, with no reasoning stored.

Requires:
    pip install roomkit[kimi]

Environment:
    KIMI_API_KEY   your key (https://platform.kimi.ai), "unused" on your own server
    KIMI_BASE_URL  a server of your own, http://gpu:8000/v1 for one (default: Kimi's API)
    KIMI_MODEL     model id (default: kimi-k3.1-pro); moonshotai/Kimi-K3 on vLLM for one

Run with:
    KIMI_API_KEY=... uv run python examples/kimi_ai.py
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
from roomkit.providers.kimi import KimiAIProvider, KimiConfig

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
    env = require_env("KIMI_API_KEY")
    settings: dict[str, Any] = {
        "api_key": env["KIMI_API_KEY"],
        "model": os.environ.get("KIMI_MODEL", "kimi-k3.1-pro"),
        "reasoning_effort": "low",
    }
    if base_url := os.environ.get("KIMI_BASE_URL"):
        settings["base_url"] = base_url
    provider = KimiAIProvider(KimiConfig(**settings))

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
    setup_logging("kimi_ai")
    asyncio.run(main())
