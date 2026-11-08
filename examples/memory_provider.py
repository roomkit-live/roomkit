"""Memory provider example — custom AI context construction.

Demonstrates how to use a MemoryProvider to control what conversation
history is included in AI context.  Shows three approaches:

1. Default SlidingWindowMemory (last N events)
2. Custom provider that injects a conversation summary
3. A provider that compacts the history between two tool rounds

Run with:
    uv run python examples/memory_provider.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import (
    AIChannel,
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
from roomkit.memory import SlidingWindowMemory
from roomkit.providers.ai import AIMessage, AIResponse, AIToolCall, MockAIProvider

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
        # Pre-built summary message (e.g. from a prior summarization pass).
        # A "user" message, like the built-in memories inject: Anthropic and
        # others reject a "system" message inside the conversation.
        summary_msg = AIMessage(role="user", content=f"[Conversation summary]\n{self._summary}")

        # Plus the most recent events for immediate context. The current
        # message is left out: the channel adds it itself, as the last turn.
        history = [e for e in context.recent_events if e.id != current_event.id]
        recent = history[-self._recent_count :]

        return MemoryResult(messages=[summary_msg], events=recent)


class CompactingBetweenRounds(SlidingWindowMemory):
    """Reads the room's latest events, and keeps a long tool turn within its
    window: past *ceiling* tokens measured by a round, the history before
    the turn's input is replaced by a one-line summary.

    A real implementation would have a model write the summary, or use
    :class:`~roomkit.memory.SummarizingMemory`, which does so past its
    ``tier2_ratio``.
    """

    def __init__(self, ceiling: int) -> None:
        super().__init__(max_events=50)
        self._ceiling = ceiling

    async def compact(
        self,
        room_id: str,
        history: list[AIMessage],
        *,
        usage: dict[str, int],
        channel_id: str | None = None,
    ) -> list[AIMessage] | None:
        if sum(usage.values()) <= self._ceiling or not history:
            return None  # the turn goes on as it is
        return [AIMessage(role="user", content=f"[Summary of {len(history)} messages]")]


async def run_compaction() -> None:
    print("\n=== Compaction between two tool rounds ===")

    def lookup(call_id: str) -> AIResponse:
        return AIResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id=call_id, name="lookup", arguments={"q": call_id})],
            usage={"input_tokens": 7000, "output_tokens": 10},
        )

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="Hi!"),
            lookup("c1"),
            AIResponse(content="Found it."),
        ]
    )

    async def tools(name: str, arguments: dict[str, Any]) -> str:
        return "order 42: shipped"

    ai = AIChannel(
        "ai-compact",
        provider=provider,
        tool_handler=tools,
        memory=CompactingBetweenRounds(ceiling=6000),
    )
    kit = RoomKit()
    ws = WebSocketChannel("ws3")
    ws.register_connection("user", lambda _conn, _event: asyncio.sleep(0), room_id="room3")
    kit.register_channel(ws)
    kit.register_channel(ai)
    await kit.create_room(room_id="room3")
    await kit.attach_channel("room3", "ws3")
    await kit.attach_channel(
        "room3",
        "ai-compact",
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [{"name": "lookup", "description": "Find an order."}]},
    )

    for body in ("Hello!", "Where is order 42?"):
        await kit.process_inbound(
            InboundMessage(channel_id="ws3", sender_id="user", content=TextContent(body=body))
        )

    # The second turn's first round measured 7,010 tokens, past the ceiling:
    # its next request opens with the summary, joined to the turn's input.
    last = provider.calls[-1].messages
    print(f"  The last request opens with: {last[0].content!r}")
    await kit.close()


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

    ws.register_connection("user", on_receive, room_id="room1")

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

    ws2.register_connection("user", on_receive2, room_id="room2")

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
    # Consecutive user turns are merged, so the summary leads the current message.
    print(f"  Context had {len(ctx.messages)} message(s):")
    for message in ctx.messages:
        print(f"    {message.role}: {message.content!r}")

    # --- 3. Compaction between two tool rounds -------------------------------
    await run_compaction()

    # --- Cleanup -------------------------------------------------------------
    await ai.close()
    await ai2.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
