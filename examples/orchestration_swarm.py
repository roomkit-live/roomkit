"""Swarm orchestration strategy example.

Demonstrates a swarm where every agent can hand off to every other
agent — no linear ordering. The entry agent handles initial messages,
and agents hand off freely as the conversation evolves.

Uses the ``Swarm`` orchestration strategy for automatic bidirectional
handoff wiring.

Run with:
    uv run python examples/orchestration_swarm.py
"""

from __future__ import annotations

import asyncio
import logging

# Suppress chain-depth warnings from AI-to-AI reentry (expected in multi-agent setups)
logging.getLogger("roomkit").setLevel(logging.ERROR)

from roomkit import Agent, InboundMessage, RoomKit, Swarm, TextContent, WebSocketChannel
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.event import RoomEvent
from roomkit.orchestration.handoff import HandoffMemoryProvider, _room_id_var
from roomkit.orchestration.state import get_conversation_state
from roomkit.providers.ai.mock import MockAIProvider

# --- Helpers -----------------------------------------------------------------


def find_reply(events: list[RoomEvent], agent_id: str, start: int = 0) -> RoomEvent | None:
    """Find the first event from a specific agent after `start` index."""
    for event in events[start:]:
        if event.source.channel_id == agent_id:
            return event
    return None


# --- Main --------------------------------------------------------------------


async def main() -> None:
    # Three specialist agents — any can hand off to any other
    ai_sales = Agent(
        "agent-sales",
        provider=MockAIProvider(responses=["Great choice! Let me help with pricing."]),
        role="Sales agent",
        description="Handles product inquiries and pricing",
        system_prompt="You are a sales agent.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    ai_support = Agent(
        "agent-support",
        provider=MockAIProvider(responses=["I'll troubleshoot that for you."]),
        role="Support agent",
        description="Handles technical issues and troubleshooting",
        system_prompt="You handle support requests.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    ai_billing = Agent(
        "agent-billing",
        provider=MockAIProvider(responses=["Let me check your invoice."]),
        role="Billing agent",
        description="Handles billing, invoices, and payment issues",
        system_prompt="You handle billing questions.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )

    # Swarm strategy: all agents can hand off to each other.
    # Sales is the entry point — handles initial messages.
    kit = RoomKit(
        orchestration=Swarm(
            agents=[ai_sales, ai_support, ai_billing],
            entry="agent-sales",
        ),
    )

    # Transport channel
    ws = WebSocketChannel("ws-user")
    inbox: list[RoomEvent] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user", on_receive)
    kit.register_channel(ws)

    # Create room — Swarm wires bidirectional handoff automatically
    await kit.create_room(room_id="swarm-room")
    await kit.attach_channel("swarm-room", "ws-user")

    # --- Simulate conversation ------------------------------------------------

    # 1. Initial message → sales (entry agent)
    print("=== Sales handles initial message ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Hi, I want to buy the Pro plan."),
        )
    )
    reply = find_reply(inbox, "agent-sales", mark)
    print(f"  Sales: {reply.content.body}")  # type: ignore[union-attr]

    # 2. Sales → Support (bidirectional handoff)
    print("\n=== Handoff: sales -> support ===")
    _room_id_var.set("swarm-room")
    await ai_sales.tool_handler(
        "handoff_conversation",
        {
            "target": "agent-support",
            "reason": "User also has a technical issue",
            "summary": "User wants Pro plan but has a setup problem.",
        },
    )

    room = await kit.get_room("swarm-room")
    state = get_conversation_state(room)
    print(f"  Active agent: {state.active_agent_id}")

    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="My API key doesn't work."),
        )
    )
    reply = find_reply(inbox, "agent-support", mark)
    print(f"  Support: {reply.content.body}")  # type: ignore[union-attr]

    # 3. Support → Billing (support can reach billing directly)
    print("\n=== Handoff: support -> billing ===")
    _room_id_var.set("swarm-room")
    await ai_support.tool_handler(
        "handoff_conversation",
        {
            "target": "agent-billing",
            "reason": "API key issue was billing-related (expired trial)",
            "summary": "User's trial expired. Needs Pro plan activation.",
        },
    )

    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Can you activate my Pro plan?"),
        )
    )
    reply = find_reply(inbox, "agent-billing", mark)
    print(f"  Billing: {reply.content.body}")  # type: ignore[union-attr]

    # 4. Billing → Sales (back to sales — bidirectional!)
    print("\n=== Handoff: billing -> sales (back!) ===")
    _room_id_var.set("swarm-room")
    await ai_billing.tool_handler(
        "handoff_conversation",
        {
            "target": "agent-sales",
            "reason": "Plan activated, back to sales for upsell",
            "summary": "Pro plan active. User may want add-ons.",
        },
    )

    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="What add-ons do you have?"),
        )
    )
    reply = find_reply(inbox, "agent-sales", mark)
    print(f"  Sales: {reply.content.body}")  # type: ignore[union-attr]

    # --- Results -------------------------------------------------------------

    print("\n=== Conversation State ===")
    room = await kit.get_room("swarm-room")
    state = get_conversation_state(room)
    print(f"  Active agent: {state.active_agent_id}")
    print(f"  Handoff count: {state.handoff_count}")

    print("\n=== Handoff History ===")
    for t in state.phase_history:
        print(f"  {t.from_agent} -> {t.to_agent} ({t.reason})")

    # Show each agent's handoff tool targets
    print("\n=== Handoff Tool Targets ===")
    for agent in [ai_sales, ai_support, ai_billing]:
        tool = next(
            (t for t in agent._injected_tools if t.name == "handoff_conversation"),
            None,
        )
        if tool:
            targets = tool.parameters["properties"]["target"].get("enum", [])
            print(f"  {agent.channel_id} can reach: {targets}")

    # Cleanup
    for ch in [ai_sales, ai_support, ai_billing]:
        await ch.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
