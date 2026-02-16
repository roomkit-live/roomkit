"""Memory provider example — custom AI context construction.

Demonstrates how to use a MemoryProvider to control what conversation
history is included in AI context.  Shows three approaches:

1. Default SlidingWindowMemory (last N events)
2. Custom provider that injects a conversation summary
3. Custom provider that filters events by channel

Run with:
    uv run python examples/memory_provider.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    AIMessage,
    ChannelCategory,
    InboundMessage,
    MemoryProvider,
    MemoryResult,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider

# --- Custom memory provider --------------------------------------------------


class SummaryMemory(MemoryProvider):
    """Prepends a summary message and includes only the last few events.

    A real implementation might call an LLM to generate the summary,
    query a vector store, or load cross-room context.
    """

    def __init__(self, summary: str, recent_count: int = 5) -> None:
        self._summary = summary
        self._recent_count = recent_count

    @property
    def name(self) -> str:
        return "SummaryMemory"

    async def retrieve(
        self,
        room_id: str,
        current_event: RoomEvent,
        context: RoomContext,
        *,
        channel_id: str | None = None,
    ) -> MemoryResult:
        # Pre-built summary message (e.g. from a prior summarization pass)
        summary_msg = AIMessage(role="system", content=self._summary)

        # Plus the most recent events for immediate context
        recent = context.recent_events[-self._recent_count :]

        return MemoryResult(messages=[summary_msg], events=recent)


# --- Main --------------------------------------------------------------------


async def main() -> None:
    # --- 1. Default behavior (SlidingWindowMemory) ---------------------------
    print("=== Default SlidingWindowMemory ===")

    provider = MockAIProvider(responses=["Got it!", "Sure thing!"])
    ai = AIChannel(
        "ai-default",
        provider=provider,
        system_prompt="You are helpful.",
        max_context_events=10,  # configures the default SlidingWindowMemory
    )

    kit = RoomKit()
    ws = WebSocketChannel("ws")

    inbox: list[RoomEvent] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user", on_receive)

    kit.register_channel(ws)
    kit.register_channel(ai)

    await kit.create_room(room_id="room1")
    await kit.attach_channel("room1", "ws")
    await kit.attach_channel("room1", "ai-default", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(channel_id="ws", sender_id="user", content=TextContent(body="Hello!"))
    )

    print(f"  AI replied: {inbox[-1].content.body}")  # type: ignore[union-attr]
    print(f"  Context had {len(provider.calls[0].messages)} message(s)")

    # --- 2. Custom SummaryMemory ---------------------------------------------
    print("\n=== Custom SummaryMemory ===")

    provider2 = MockAIProvider(responses=["Thanks for the context!"])
    memory = SummaryMemory(
        summary="The user previously discussed billing issues and was offered a 10% discount.",
        recent_count=3,
    )
    ai2 = AIChannel(
        "ai-summary",
        provider=provider2,
        system_prompt="You are a support agent.",
        memory=memory,
    )

    kit2 = RoomKit()
    ws2 = WebSocketChannel("ws2")

    inbox2: list[RoomEvent] = []

    async def on_receive2(_conn: str, event: RoomEvent) -> None:
        inbox2.append(event)

    ws2.register_connection("user", on_receive2)

    kit2.register_channel(ws2)
    kit2.register_channel(ai2)

    await kit2.create_room(room_id="room2")
    await kit2.attach_channel("room2", "ws2")
    await kit2.attach_channel("room2", "ai-summary", category=ChannelCategory.INTELLIGENCE)

    await kit2.process_inbound(
        InboundMessage(
            channel_id="ws2",
            sender_id="user",
            content=TextContent(body="Can you check my account?"),
        )
    )

    ctx = provider2.calls[0]
    print(f"  AI replied: {inbox2[-1].content.body}")  # type: ignore[union-attr]
    print(f"  Context had {len(ctx.messages)} message(s)")
    print(f"  First message (summary): {ctx.messages[0].content}")
    print(f"  Last message (current):  {ctx.messages[-1].content}")

    # --- Cleanup -------------------------------------------------------------
    await ai.close()
    await ai2.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
