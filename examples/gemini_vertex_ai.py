"""Gemini on Vertex AI example — regional endpoint, no data retention.

Runs Google's Gemini models through Vertex AI in a pinned Google Cloud region,
so prompts and responses stay in-region (data residency) and are not retained
to train Google's models — the backend to reach for under Québec Law 25 /
PIPEDA. It is the same Gemini, just authenticated via Application Default
Credentials (ADC) instead of an API key.

Requires:
    pip install roomkit[gemini]
    gcloud auth application-default login   # provides ADC

Environment variables:
    GOOGLE_CLOUD_PROJECT     — Google Cloud project id (required)
    GEMINI_VERTEX_LOCATION   — Vertex region (default: northamerica-northeast1 / Montréal)
    GEMINI_VERTEX_MODEL      — model id (default: gemini-3.1-flash-lite)

Run with:
    GOOGLE_CLOUD_PROJECT=my-proj uv run python examples/gemini_vertex_ai.py
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
from roomkit.providers.gemini import GeminiVertexConfig, GeminiVertexProvider


async def main() -> None:
    env = require_env("GOOGLE_CLOUD_PROJECT")

    provider = GeminiVertexProvider(
        GeminiVertexConfig(
            project=env["GOOGLE_CLOUD_PROJECT"],
            # Pin the region for data residency — defaults to Montréal.
            location=os.environ.get("GEMINI_VERTEX_LOCATION", "northamerica-northeast1"),
            model=os.environ.get("GEMINI_VERTEX_MODEL", "gemini-3.1-flash-lite"),
        )
    )

    kit = RoomKit()

    ws = WebSocketChannel("ws-user")
    ai = AIChannel(
        "ai-assistant",
        provider=provider,
        system_prompt="You are a helpful assistant running on Vertex AI.",
    )

    kit.register_channel(ws)
    kit.register_channel(ai)

    inbox: list[RoomEvent] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_receive)

    await kit.create_room(room_id="vertex-demo")
    await kit.attach_channel("vertex-demo", "ws-user")
    await kit.attach_channel("vertex-demo", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(
                body="In one sentence, what is Vertex AI's data residency benefit?"
            ),
        )
    )
    print(f"Sent message -> blocked={result.blocked}")

    for ev in inbox:
        print(f"  AI replied: {ev.content.body}")  # type: ignore[union-attr]


if __name__ == "__main__":
    asyncio.run(main())
