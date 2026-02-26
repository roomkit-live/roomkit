"""AI thinking / reasoning with per-room thinking budget.

Demonstrates how to enable AI thinking (chain-of-thought reasoning)
in AIChannel. The thinking content is captured, published as ephemeral
events, and preserved in conversation history across tool-loop rounds.

Supported providers:
- Anthropic: native extended thinking (requires Claude 3.5+ models)
- Ollama / vLLM: <think>...</think> tag parsing (DeepSeek-R1, QwQ, etc.)
- OpenAI: <think> tag parsing via the OpenAI-compatible API

Run with:
    uv run python examples/ai_thinking.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType


async def main() -> None:
    kit = RoomKit()

    ws = WebSocketChannel("ws-user")

    # MockAIProvider with thinking content simulates a reasoning model.
    # In production, use AnthropicAIProvider or create_vllm_provider().
    ai = AIChannel(
        "ai-thinker",
        provider=MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="The answer is 42.",
                    thinking="Let me reason step by step. First, I consider "
                    "the question from multiple angles. The phrase 'meaning "
                    "of life' is often associated with Douglas Adams...",
                    finish_reason="stop",
                    usage={"prompt_tokens": 20, "completion_tokens": 15},
                ),
            ]
        ),
        system_prompt="You are a thoughtful assistant. Think carefully before answering.",
        thinking_budget=8192,
    )
    kit.register_channel(ws)
    kit.register_channel(ai)

    inbox: list[RoomEvent] = []
    thinking_log: list[dict[str, Any]] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_recv)

    # Subscribe to ephemeral events to observe thinking in real time.
    # AIChannel publishes THINKING_START / THINKING_END events while
    # the model reasons, before broadcasting the final answer.
    async def on_ephemeral(event: EphemeralEvent) -> None:
        if event.type in (EphemeralEventType.THINKING_START, EphemeralEventType.THINKING_END):
            thinking_log.append({"type": event.type.value, "data": event.data})

    # --- Default thinking budget from AIChannel constructor ---
    print("=== Default Thinking Budget ===")
    await kit.create_room(room_id="think-room")

    # Subscribe to the room's ephemeral events
    sub_id = await kit.realtime.subscribe_to_room("think-room", on_ephemeral)

    await kit.attach_channel("think-room", "ws-user")
    await kit.attach_channel(
        "think-room",
        "ai-thinker",
        category=ChannelCategory.INTELLIGENCE,
    )

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="What is the meaning of life?"),
        )
    )

    mock: MockAIProvider = ai._provider  # type: ignore[assignment]
    if mock.calls:
        ctx = mock.calls[-1]
        print(f"  Thinking budget: {ctx.thinking_budget}")

    # Show thinking events (emitted before the answer)
    for entry in thinking_log:
        if entry["type"] == "thinking_end":
            print(f"  AI thinking: {entry['data']['thinking']}")

    for ev in inbox:
        if ev.source.channel_id == "ai-thinker":
            print(f"  AI answer: {ev.content.body}")  # type: ignore[union-attr]

    await kit.realtime.unsubscribe(sub_id)

    # --- Per-room thinking budget via binding metadata ---
    print("\n=== Per-Room Thinking Budget ===")
    inbox.clear()
    thinking_log.clear()

    await kit.detach_channel("think-room", "ws-user")
    await kit.detach_channel("think-room", "ai-thinker")

    await kit.create_room(room_id="deep-think-room")
    sub_id = await kit.realtime.subscribe_to_room("deep-think-room", on_ephemeral)

    await kit.attach_channel("deep-think-room", "ws-user")
    await kit.attach_channel(
        "deep-think-room",
        "ai-thinker",
        category=ChannelCategory.INTELLIGENCE,
        metadata={
            "system_prompt": "You are a math tutor. Show your reasoning.",
            "thinking_budget": 16384,  # Override per room
        },
    )

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Prove that sqrt(2) is irrational."),
        )
    )

    if len(mock.calls) > 1:
        ctx = mock.calls[-1]
        print(f"  Thinking budget: {ctx.thinking_budget}")
        print(f"  System prompt: {ctx.system_prompt[:50]}...")

    for entry in thinking_log:
        if entry["type"] == "thinking_end":
            print(f"  AI thinking: {entry['data']['thinking']}")

    for ev in inbox:
        if ev.source.channel_id == "ai-thinker":
            print(f"  AI answer: {ev.content.body}")  # type: ignore[union-attr]

    await kit.realtime.unsubscribe(sub_id)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
