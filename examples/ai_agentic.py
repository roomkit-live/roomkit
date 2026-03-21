"""Agentic AI features — planning, eviction, and summarizing memory.

Demonstrates the agentic capabilities of AIChannel:

1. **Planning tools** — AI creates structured task plans to track progress
2. **Large output eviction** — oversized tool results are stored externally
   and replaced with previews; the AI can paginate back
3. **SummarizingMemory** — two-tier context budget management for long
   conversations (tier 1: truncation, tier 2: LLM summarization)
4. **Dangling tool call recovery** — automatic (no setup needed)

Run with:
    uv run python examples/ai_agentic.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.memory import SlidingWindowMemory, SummarizingMemory
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider

# ---------------------------------------------------------------------------
# Example tool: simulates a database query returning a large result
# ---------------------------------------------------------------------------


class QueryDatabase:
    """Simulate a database query that returns many rows.

    Implements the Tool protocol: .definition property + .handler() method.
    """

    @property
    def definition(self) -> dict:
        return {
            "name": "query_database",
            "description": "Query the customer database. Returns CSV rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL query to execute"},
                },
                "required": ["sql"],
            },
        }

    async def handler(self, name: str, arguments: dict) -> str:
        # Simulate a large result (~300 lines)
        header = "id,name,email,plan,revenue,signup_date"
        rows = [
            f"{i},Customer {i},customer{i}@example.com,"
            f"{'enterprise' if i % 3 == 0 else 'pro'},{i * 100 + 50},2025-{(i % 12) + 1:02d}-15"
            for i in range(1, 301)
        ]
        return header + "\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# Mock provider that demonstrates planning + tool calling
# ---------------------------------------------------------------------------


def make_planning_provider() -> MockAIProvider:
    """Create a mock provider that plans work, then calls tools."""
    return MockAIProvider(
        ai_responses=[
            # Turn 1: AI creates a plan
            AIResponse(
                content="I'll analyze the customer database. Let me create a plan first.",
                tool_calls=[
                    AIToolCall(
                        id="tc_plan",
                        name="_plan_tasks",
                        arguments={
                            "tasks": [
                                {"title": "Query customer database", "status": "in_progress"},
                                {"title": "Analyze revenue by plan type", "status": "pending"},
                                {"title": "Summarize findings", "status": "pending"},
                            ]
                        },
                    )
                ],
            ),
            # Turn 2: After plan is stored, query database
            AIResponse(
                content="Plan created. Now querying the database...",
                tool_calls=[
                    AIToolCall(
                        id="tc_query",
                        name="query_database",
                        arguments={"sql": "SELECT * FROM customers"},
                    )
                ],
            ),
            # Turn 3: After tool results (evicted), update plan and summarize
            AIResponse(
                content="I've retrieved 300 customer records. The data has been evicted "
                "due to size. Based on the preview:\n\n"
                "- Enterprise customers: ~100 (every 3rd)\n"
                "- Pro customers: ~200\n"
                "- Revenue range: $150 to $30,050\n\n"
                "Analysis complete.",
                tool_calls=[
                    AIToolCall(
                        id="tc_plan2",
                        name="_plan_tasks",
                        arguments={
                            "tasks": [
                                {"title": "Query customer database", "status": "completed"},
                                {"title": "Analyze revenue by plan type", "status": "completed"},
                                {"title": "Summarize findings", "status": "completed"},
                            ]
                        },
                    )
                ],
            ),
            # Turn 4: Final response after plan update
            AIResponse(
                content="All tasks complete! The customer database has 300 records "
                "across enterprise and pro plans.",
                tool_calls=[],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    kit = RoomKit()

    ws = WebSocketChannel("ws-user")
    kit.register_channel(ws)

    inbox: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_recv)

    # --- Create an agentic AI channel ---
    provider = make_planning_provider()

    # Use SummarizingMemory for long conversation support
    summary_provider = MockAIProvider(
        responses=["Summary: user asked about customer data, AI queried 300 records."]
    )
    memory = SummarizingMemory(
        inner=SlidingWindowMemory(max_events=100),
        provider=summary_provider,
        max_context_tokens=128_000,
    )

    ai = AIChannel(
        "ai-analyst",
        provider=provider,
        system_prompt="You are a data analyst. Use planning to organize complex tasks.",
        enable_planning=True,  # enables _plan_tasks tool
        evict_threshold_tokens=500,  # low threshold for demo (evicts the 300-row CSV)
        memory=memory,
        tools=[QueryDatabase()],
    )
    kit.register_channel(ai)

    # --- Subscribe to ephemeral events for plan updates ---
    plan_updates: list[dict] = []

    async def on_ephemeral(event) -> None:
        data = event.data if hasattr(event, "data") else {}
        if data.get("type") == "plan_updated":
            plan_updates.append(data)
            tasks = data["tasks"]
            done = sum(1 for t in tasks if t["status"] == "completed")
            total = len(tasks)
            print(f"  [Plan Update] {done}/{total} tasks completed")
            for t in tasks:
                icon = {"completed": "[x]", "in_progress": "[-]", "pending": "[ ]"}.get(
                    t["status"], "[ ]"
                )
                print(f"    {icon} {t['title']}")

    await kit.subscribe_room("analysis-room", on_ephemeral)

    # --- Create room and run ---
    await kit.create_room(room_id="analysis-room")
    await kit.attach_channel("analysis-room", "ws-user")
    await kit.attach_channel("analysis-room", "ai-analyst", category=ChannelCategory.INTELLIGENCE)

    # User sends a request
    print("=== Agentic AI: Planning + Eviction + Memory ===\n")
    print("User: Analyze our customer database and summarize findings.\n")

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user-1",
            content=TextContent(body="Analyze our customer database and summarize findings."),
        )
    )

    # Give the tool loop time to complete
    await asyncio.sleep(0.5)

    # Show results
    print("\n--- AI Responses ---")
    for event in inbox:
        if isinstance(event.content, TextContent) and event.content.body:
            print(f"\nAI: {event.content.body[:200]}")

    print("\n--- Eviction Stats ---")
    evicted = ai._eviction._store
    print(f"  Evicted results: {len(evicted)}")
    for rid in evicted:
        lines = evicted[rid].count("\n") + 1
        print(f"  {rid}: {lines} lines stored")

    print("\n--- Plan Updates Received ---")
    print(f"  Total plan events: {len(plan_updates)}")

    print("\n--- Memory Provider ---")
    print(f"  Type: {memory.name}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
