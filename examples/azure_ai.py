"""Azure AI Studio provider example.

Demonstrates using Azure AI Foundry deployments (DeepSeek, GPT-4o, Mistral,
etc.) with RoomKit's AIChannel via the OpenAI-compatible Chat Completions API.

Requires:
    pip install roomkit[azure]

Environment variables:
    AZURE_API_KEY       — Azure API key
    AZURE_ENDPOINT      — Azure AI Foundry project endpoint
    AZURE_DEPLOYMENT    — Model deployment name (e.g. "DeepSeek-R1", "gpt-4o")

Run with:
    AZURE_API_KEY=... AZURE_ENDPOINT=... uv run python examples/azure_ai.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import os

from shared import require_env

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.azure import AzureAIConfig, AzureAIProvider


async def main() -> None:
    env = require_env("AZURE_API_KEY", "AZURE_ENDPOINT")

    provider = AzureAIProvider(
        AzureAIConfig(
            api_key=env["AZURE_API_KEY"],
            azure_endpoint=env["AZURE_ENDPOINT"],
            model=os.environ.get("AZURE_DEPLOYMENT", "DeepSeek-R1"),
        )
    )

    kit = RoomKit()

    ws = WebSocketChannel("ws-user")
    ai = AIChannel(
        "ai-assistant",
        provider=provider,
        system_prompt="You are a helpful assistant powered by Azure AI.",
    )

    kit.register_channel(ws)
    kit.register_channel(ai)

    # Capture what the user receives back.
    inbox: list[RoomEvent] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_receive)

    await kit.create_room(room_id="azure-demo")
    await kit.attach_channel("azure-demo", "ws-user")
    await kit.attach_channel("azure-demo", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="In one sentence, what is Azure AI Foundry?"),
        )
    )
    print(f"Sent message -> blocked={result.blocked}")

    for ev in inbox:
        print(f"  AI replied: {ev.content.body}")  # type: ignore[union-attr]


if __name__ == "__main__":
    asyncio.run(main())
