"""AI Memory — SummarizingMemory for long conversations.

SummarizingMemory uses a two-tier strategy to keep conversations within
what the turn leaves of the model's window (its system prompt, its tools,
the runtime's notes and its reply are paid for first):

- **Tier 1** — truncates old message bodies (cheap, no LLM call)
- **Tier 2** — summarizes old messages with a lightweight model (Haiku)

Thresholds are set low in this demo so you can see compression after
just a few exchanges. Ask about a few cities, then which one came first:
the printed summary is what still knows.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/ai_memory.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import require_env

from roomkit import (
    AIChannel,
    ChannelCategory,
    CLIChannel,
    MemoryResult,
    RoomContext,
    RoomEvent,
    RoomKit,
)
from roomkit.memory import (
    MemoryProvider,
    SlidingWindowMemory,
    SummarizingMemory,
    current_turn_footprint,
    estimate_event_tokens,
    history_budget,
)
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig

_CYAN = "\033[36m"
_RESET = "\033[0m"

# The summarizing memory's settings, which the report below reads too. The
# window is tight so compression is visible after a few exchanges: the system
# prompt and the runtime's notes take about 300 tokens of it, which leaves the
# history about 1,000.
WINDOW = 1300
MAX_EVENTS = 100
TIER1_RATIO = 0.3
TIER2_RATIO = 0.5


def _tokens(events: list[RoomEvent]) -> int:
    return sum(estimate_event_tokens(event) for event in events)


def _history_budget(current_event: RoomEvent) -> int:
    """What the turn leaves the history of the window, as SummarizingMemory
    reckons it (RFC §20): the window less the input the AI channel measured,
    the turn's reply budget and the current event. Its tiers are fractions
    of it, and it keeps the history within it."""
    footprint = current_turn_footprint()
    return history_budget(
        max_context_tokens=WINDOW,
        safety_margin_ratio=0.0,
        reserved_tokens=footprint.input_tokens if footprint else 0,
        current_event=current_event,
        reply_tokens=footprint.reply_tokens if footprint else 0,
    )


class UsageReport(MemoryProvider):
    """Wraps a memory and prints what it kept of the room's history, and
    each summary, after each turn."""

    def __init__(self, memory: MemoryProvider) -> None:
        self._memory = memory

    @property
    def recent_events_window(self) -> int:
        return self._memory.recent_events_window

    async def retrieve(
        self,
        room_id: str,
        current_event: RoomEvent,
        context: RoomContext,
        *,
        channel_id: str | None = None,
    ) -> MemoryResult:
        # The room's latest events as the sliding window reads them, before compression.
        earlier = [e for e in context.recent_events[-MAX_EVENTS:] if e.id != current_event.id]
        before_tokens = _tokens(earlier)
        budget = _history_budget(current_event)

        result = await self._memory.retrieve(
            room_id, current_event, context, channel_id=channel_id
        )

        after_tokens = _tokens(result.events)
        pct_before = int(before_tokens / budget * 100) if budget else 0
        pct_after = int(after_tokens / budget * 100) if budget else 0
        if before_tokens > int(budget * TIER2_RATIO) and after_tokens < before_tokens:
            print(
                f"\n{_CYAN}  [memory] tier 2 — summarized: "
                f"{before_tokens} → {after_tokens} tokens "
                f"({pct_before}% → {pct_after}% of {budget}){_RESET}"
            )
            # The summary rides result.messages, ahead of the events kept verbatim.
            for message in result.messages:
                if isinstance(message.content, str) and "[Conversation summary" in message.content:
                    print(f"{_CYAN}  {message.content}{_RESET}")
            print()
        elif before_tokens > int(budget * TIER1_RATIO) and after_tokens < before_tokens:
            print(
                f"\n{_CYAN}  [memory] tier 1 — truncated: "
                f"{before_tokens} → {after_tokens} tokens "
                f"({pct_before}% → {pct_after}% of {budget}){_RESET}\n"
            )
        else:
            print(f"\n{_CYAN}  [memory] {after_tokens}/{budget} tokens ({pct_after}%){_RESET}\n")
        return result

    async def ingest(
        self, room_id: str, event: RoomEvent, *, channel_id: str | None = None
    ) -> None:
        await self._memory.ingest(room_id, event, channel_id=channel_id)

    async def clear(self, room_id: str) -> None:
        await self._memory.clear(room_id)

    async def close(self) -> None:
        await self._memory.close()


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")
    api_key = env["ANTHROPIC_API_KEY"]

    # Main provider for generation
    main_provider = AnthropicAIProvider(AnthropicConfig(api_key=api_key, model="claude-opus-5"))

    # Lightweight provider for summarization (Haiku — fast and cheap)
    summary_provider = AnthropicAIProvider(
        AnthropicConfig(api_key=api_key, model="claude-haiku-5-5")
    )

    # Tier 2 keeps recent events within (half of what the turn leaves -
    # summary_max_tokens) tokens but never fewer than min_events. With about
    # 100 tokens and min_events=2, exactly the last question and answer stay
    # verbatim, so the summarized part ends on an answer: one ending on a
    # question gets it answered, not summarized. summary_max_tokens also
    # leaves room for every city asked about so far.
    memory = UsageReport(
        SummarizingMemory(
            inner=SlidingWindowMemory(max_events=MAX_EVENTS),
            provider=summary_provider,
            max_context_tokens=WINDOW,
            tier1_ratio=TIER1_RATIO,
            tier2_ratio=TIER2_RATIO,
            summary_max_tokens=400,
            min_events=2,
        )
    )

    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=main_provider,
        system_prompt="You are a helpful assistant.",
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
            "The window is 1,300 tokens, about 1,000 of them left to the history:\n"
            "past half of that, older exchanges are summarized by Haiku, so\n"
            "compression triggers after a few exchanges.\n"
            "Context usage (and each summary) is shown after each turn.\n"
        ),
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
