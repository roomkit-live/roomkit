"""AI Memory — SummarizingMemory for long conversations.

SummarizingMemory uses a two-tier strategy to keep conversations within
the context budget:

- **Tier 1** — truncates old message bodies (cheap, no LLM call)
- **Tier 2** — summarizes old messages with a lightweight model (Haiku)

Thresholds are set low in this demo so you can see compression after
just a few exchanges.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/ai_memory.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.env import require_env

from roomkit import (
    ChannelCategory,
    CLIChannel,
    RoomKit,
)
from roomkit.channels.ai import AIChannel
from roomkit.memory import SlidingWindowMemory, SummarizingMemory
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")
    api_key = env["ANTHROPIC_API_KEY"]

    # Main provider for generation
    main_provider = AnthropicAIProvider(AnthropicConfig(api_key=api_key))

    # Lightweight provider for summarization (Haiku — fast and cheap)
    summary_provider = AnthropicAIProvider(
        AnthropicConfig(api_key=api_key, model="claude-haiku-4-5-20251001")
    )

    # Low thresholds so tier 1/2 trigger after a few turns
    memory = SummarizingMemory(
        inner=SlidingWindowMemory(max_events=100),
        provider=summary_provider,
        max_context_tokens=8000,
        tier1_ratio=0.4,
        tier2_ratio=0.7,
    )

    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=main_provider,
        system_prompt=(
            "You are a helpful assistant. Give detailed, thorough answers.\n"
            "If you notice a conversation summary at the start of the history, "
            "mention it briefly so the user can see memory compression in action."
        ),
        memory=memory,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    await kit.create_room(room_id="memory-room")
    await kit.attach_channel("memory-room", "cli")
    await kit.attach_channel("memory-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    await cli.run(
        kit,
        room_id="memory-room",
        welcome=(
            "\nMemory demo — SummarizingMemory compresses old messages.\n"
            "Thresholds are low (8k tokens) so compression triggers after a\n"
            "few exchanges. The AI will mention when it sees a summary.\n"
        ),
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
