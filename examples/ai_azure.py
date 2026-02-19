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
    uv run python examples/ai_azure.py
"""

from __future__ import annotations

import asyncio
import os

from roomkit import AIChannel, RoomKit, TextContent, WebSocketChannel
from roomkit.providers.azure import AzureAIConfig, AzureAIProvider


async def main() -> None:
    kit = RoomKit()

    ws = WebSocketChannel("ws-user")
    ai = AIChannel(
        "ai-assistant",
        provider=AzureAIProvider(
            AzureAIConfig(
                api_key=os.environ["AZURE_API_KEY"],
                azure_endpoint=os.environ["AZURE_ENDPOINT"],
                model=os.environ.get("AZURE_DEPLOYMENT", "DeepSeek-R1"),
            )
        ),
        system_prompt="You are a helpful assistant powered by Azure AI.",
    )

    kit.register_channel(ws)
    kit.register_channel(ai)

    room = await kit.create_room("azure-demo")
    await kit.join(room.room_id, ws, "user-1")
    await kit.join(room.room_id, ai, "ai-1")

    # Simulate a user message
    await kit.send(
        room.room_id,
        ws.channel_id,
        TextContent(text="What models are available on Azure AI Studio?"),
        sender_id="user-1",
    )

    # Retrieve the AI response
    events = await kit.get_events(room.room_id)
    for event in events:
        print(f"[{event.source.channel_id}] {event.content}")


if __name__ == "__main__":
    asyncio.run(main())
