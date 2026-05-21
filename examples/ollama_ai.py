"""Ollama AI example — AI-powered assistant via Ollama's native API.

Uses ``roomkit.providers.ollama.OllamaAIProvider`` rather than the
OpenAI-compatible shim, so the model's ``think`` parameter and the
streamed ``thinking`` field actually work. See ``ollama_cli.py`` for
an interactive version that demonstrates ``think`` on/off, streaming
on/off, and MCP tool calls.

Run with:
    OLLAMA_HOST=http://localhost:11434 OLLAMA_MODEL=qwen3:8b \\
        uv run python examples/ollama_ai.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import setup_logging  # noqa: E402

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ollama import OllamaAIProvider, OllamaConfig

setup_logging("ollama_ai")


async def main() -> None:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

    provider = OllamaAIProvider(OllamaConfig(host=host, model=model))

    kit = RoomKit()

    ws = WebSocketChannel("ws-user")
    ai = AIChannel(
        "ai-assistant",
        provider=provider,
        system_prompt="You are a helpful assistant. Keep answers concise.",
    )
    kit.register_channel(ws)
    kit.register_channel(ai)

    inbox: list[RoomEvent] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_receive)

    await kit.create_room(room_id="demo-room")
    await kit.attach_channel("demo-room", "ws-user")
    await kit.attach_channel("demo-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="What is RoomKit, in one sentence?"),
        )
    )
    print(f"Sent message -> blocked={result.blocked}")
    for ev in inbox:
        if ev.source.channel_id == "ai-assistant":
            print(f"  AI replied: {ev.content.body}")  # type: ignore[union-attr]

    await provider.close()
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
