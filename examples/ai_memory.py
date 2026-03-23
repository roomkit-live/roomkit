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
from roomkit.memory.base import MemoryResult
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig

_CYAN = "\033[36m"
_RESET = "\033[0m"


class VerboseSummarizingMemory(SummarizingMemory):
    """SummarizingMemory that prints context usage and compression events."""

    async def retrieve(
        self,
        room_id: str,
        current_event: RoomEvent,
        context: RoomContext,
        *,
        channel_id: str | None = None,
    ) -> MemoryResult:
        # Tokens before compression
        inner_result = await self._inner.retrieve(
            room_id, current_event, context, channel_id=channel_id
        )
        before_tokens = self._estimate_events_tokens(inner_result.events)
        budget = self._max_context_tokens
        pct_before = int(before_tokens / budget * 100) if budget else 0

        # Run normal retrieve (which may apply tier 1/2)
        result = await super().retrieve(room_id, current_event, context, channel_id=channel_id)

        after_tokens = self._estimate_events_tokens(result.events)
        pct_after = int(after_tokens / budget * 100) if budget else 0

        tier1_threshold = int(budget * self._tier1_ratio)
        tier2_threshold = int(budget * self._tier2_ratio)

        if before_tokens > tier2_threshold and after_tokens < before_tokens:
            print(
                f"\n{_CYAN}  [memory] tier 2 — summarized: "
                f"{before_tokens} → {after_tokens} tokens "
                f"({pct_before}% → {pct_after}% of {budget}){_RESET}\n"
            )
        elif before_tokens > tier1_threshold and after_tokens < before_tokens:
            print(
                f"\n{_CYAN}  [memory] tier 1 — truncated: "
                f"{before_tokens} → {after_tokens} tokens "
                f"({pct_before}% → {pct_after}% of {budget}){_RESET}\n"
            )
        else:
            print(f"\n{_CYAN}  [memory] {after_tokens}/{budget} tokens ({pct_after}%){_RESET}\n")

        return result


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")
    api_key = env["ANTHROPIC_API_KEY"]

    # Main provider for generation
    main_provider = AnthropicAIProvider(AnthropicConfig(api_key=api_key))

    # Lightweight provider for summarization (Haiku — fast and cheap)
    summary_provider = AnthropicAIProvider(
        AnthropicConfig(api_key=api_key, model="claude-haiku-4-5-20251001")
    )

    # Tight budget so compression is visible after a few exchanges
    memory = VerboseSummarizingMemory(
        inner=SlidingWindowMemory(max_events=100),
        provider=summary_provider,
        max_context_tokens=4000,
        tier1_ratio=0.3,
        tier2_ratio=0.5,
        summary_max_tokens=100,
        min_events=2,
    )

    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=main_provider,
        system_prompt="You are a helpful assistant. Keep answers to 2-3 sentences max.",
        max_tokens=150,
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
            "Budget is 4k tokens with short responses, so compression\n"
            "triggers after a few exchanges.\n"
            "Context usage is shown after each turn.\n"
        ),
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
