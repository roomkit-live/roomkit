"""Per-turn AI configuration with ``config_provider``.

Channel config (system prompt, tools, sampling, reasoning) is often dynamic
in real deployments — admin edits, per-user gating, feature flags. Binding
metadata is a snapshot taken at attach time, so it becomes a second source of
truth that goes stale. ``AIChannel(config_provider=...)`` resolves the config
fresh at the start of every generation instead.

This example demonstrates the resolution chain, from the most specific source
that has an opinion:

    1. binding.metadata        — per-room operator intent, always wins
    2. config_provider result  — resolved by your callback, every turn
    3. AIChannel constructor   — the channel default

``None`` at a tier means "not set here" and defers outward, so an unset knob
never overrides with a default.

The knobs shown here are the reasoning pair. A thinking model costs two to
three times the tokens and the latency of a direct answer, and that trade is
not the same in an agent's tool loop — where the model mostly shapes results
it already has — as in a chat turn where the reasoning is the value.

Run with:
    uv run python examples/ai_turn_config.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    AIChannelTurnConfig,
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider

# ---------------------------------------------------------------------------
# 1. The per-turn callback
# ---------------------------------------------------------------------------


async def resolve_turn_config(
    binding: ChannelBinding,
    context: RoomContext,
) -> AIChannelTurnConfig | None:
    """Decide this turn's config from the application's own source of truth.

    Here the room's own metadata stands in for whatever you would really
    consult — a settings table, a feature flag service, a per-user plan.
    Returning ``None`` leaves every tier below untouched.
    """
    if context.room.metadata.get("mode") == "agent":
        # An agent working through its tools does not need to think out loud;
        # it is shaping results it already has. Buy the latency back.
        return AIChannelTurnConfig(enable_thinking=False)

    # A chat turn is where the reasoning is the value.
    return AIChannelTurnConfig(enable_thinking=True, reasoning_effort="high")


# ---------------------------------------------------------------------------
# 2. Main demo
# ---------------------------------------------------------------------------


async def main() -> None:
    kit = RoomKit()

    ws = WebSocketChannel("ws-user")
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="Done.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 5},
            )
        ]
        * 3
    )

    # The channel defaults are the last tier: they apply only when neither
    # the binding nor the callback has an opinion.
    ai = AIChannel(
        "ai-agent",
        provider=provider,
        system_prompt="You are a helpful assistant.",
        enable_thinking=True,
        reasoning_effort="low",
        config_provider=resolve_turn_config,
    )

    kit.register_channel(ws)
    kit.register_channel(ai)

    inbox: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_recv, room_id="chat-room")

    def show(label: str) -> None:
        """Print what actually reached the provider for the last turn."""
        ctx = provider.calls[-1]
        print(f"  {label}")
        print(f"    enable_thinking : {ctx.enable_thinking}")
        print(f"    reasoning_effort: {ctx.reasoning_effort}")

    async def ask(room_id: str, text: str) -> None:
        # The socket ends up bound to three rooms, so the target is explicit
        # rather than inferred — the default router refuses to guess.
        await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id="user",
                content=TextContent(body=text),
            ),
            room_id=room_id,
        )

    # --- Tier 2: the callback decides (chat mode) ---
    print("=== config_provider decides — chat room ===")
    await kit.create_room(room_id="chat-room")
    await kit.attach_channel("chat-room", "ws-user")
    await kit.attach_channel(
        "chat-room",
        "ai-agent",
        category=ChannelCategory.INTELLIGENCE,
    )

    await ask("chat-room", "Why is the sky blue?")
    show("callback returned enable_thinking=True, reasoning_effort='high'")

    # --- Tier 2: same callback, different room, opposite answer ---
    print("\n=== config_provider decides — agent room ===")
    await kit.create_room(room_id="agent-room", metadata={"mode": "agent"})
    await kit.attach_channel("agent-room", "ws-user")
    ws.subscribe("user-conn", "agent-room")  # same socket, second conversation
    await kit.attach_channel(
        "agent-room",
        "ai-agent",
        category=ChannelCategory.INTELLIGENCE,
    )

    await ask("agent-room", "Run the migration and report what changed.")
    show(
        "same channel, same provider — the turn switched reasoning off; "
        "reasoning_effort was left None by the callback, so it fell through "
        "to the channel default"
    )

    # --- Tier 1: binding metadata outranks the callback ---
    print("\n=== binding metadata wins — audited room ===")
    await kit.create_room(room_id="audited-room", metadata={"mode": "agent"})
    await kit.attach_channel("audited-room", "ws-user")
    ws.subscribe("user-conn", "audited-room")
    await kit.attach_channel(
        "audited-room",
        "ai-agent",
        category=ChannelCategory.INTELLIGENCE,
        metadata={
            # The room is in agent mode, so the callback would switch thinking
            # off — but an operator asked for a reasoning trace on this room,
            # and per-room operator intent always wins.
            "enable_thinking": True,
            "reasoning_effort": "high",
        },
    )

    await ask("audited-room", "Run the migration and report what changed.")
    show("callback said False; the binding overrode it")

    print(f"\nTurns generated: {len(provider.calls)}, events delivered: {len(inbox)}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
