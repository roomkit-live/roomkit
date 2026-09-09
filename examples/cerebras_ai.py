"""Cerebras assistant using RoomKit's shared OpenAI-compatible transport.

Run with:
    CEREBRAS_API_KEY=... uv run --extra cerebras python examples/cerebras_ai.py

Optionally set CEREBRAS_MODEL and CEREBRAS_REASONING_EFFORT. GPT OSS accepts
low/medium/high; qwen-3.8-27b also accepts none for shorter response latency.
"""

from __future__ import annotations

import asyncio
import logging
import os

from roomkit import (
    AIChannel,
    CerebrasAIProvider,
    CerebrasConfig,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)

logger = logging.getLogger("roomkit.examples.cerebras")


async def main() -> None:
    provider = CerebrasAIProvider(
        CerebrasConfig(
            api_key=os.environ["CEREBRAS_API_KEY"],
            model=os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b"),
            reasoning_effort=os.environ.get("CEREBRAS_REASONING_EFFORT", "low"),
        )
    )
    async with RoomKit() as kit:
        ws = WebSocketChannel("user")
        kit.register_channel(ws)
        kit.register_channel(
            AIChannel("assistant", provider=provider, system_prompt="Keep answers concise.")
        )
        await kit.create_room(room_id="demo")
        await kit.attach_channel("demo", "user")
        await kit.attach_channel("demo", "assistant")

        async def receive(_connection: str, event: RoomEvent) -> None:
            if isinstance(event.content, TextContent):
                logger.info("Assistant: %s", event.content.body)

        ws.register_connection("browser", receive, room_id="demo")
        await kit.process_inbound(
            InboundMessage(
                channel_id="user",
                sender_id="alice",
                content=TextContent(
                    body="Suggest one way to make voice assistants more responsive."
                ),
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
