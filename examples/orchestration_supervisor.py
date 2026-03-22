"""Supervisor orchestration — manual delegation mode.

Demonstrates the default Supervisor mode (no ``strategy``) where
per-worker ``delegate_to_<id>`` tools are injected and the AI
decides when to call them.

Compare with:
- ``orchestration_content_workflow.py`` — ``strategy="sequential"``
- ``orchestration_parallel_tasks.py`` — ``strategy="parallel"``

Run with:
    uv run python examples/orchestration_supervisor.py
"""

from __future__ import annotations

import asyncio
import json
import logging

logging.getLogger("roomkit").setLevel(logging.ERROR)

from roomkit import Agent, InboundMessage, RoomKit, Supervisor, TextContent, WebSocketChannel
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.event import RoomEvent
from roomkit.orchestration.handoff import _room_id_var
from roomkit.orchestration.state import get_conversation_state
from roomkit.providers.ai.mock import MockAIProvider

# --- Helpers -----------------------------------------------------------------


def find_reply(events: list[RoomEvent], agent_id: str, start: int = 0) -> RoomEvent | None:
    """Find the first event from a specific agent after ``start`` index."""
    for event in events[start:]:
        if event.source.channel_id == agent_id:
            return event
    return None


# --- Main --------------------------------------------------------------------


async def main() -> None:
    # Supervisor — talks to the user
    manager = Agent(
        "agent-manager",
        provider=MockAIProvider(
            responses=[
                "I'll have our researcher look into that.",
                "The research is complete. Here's what we found.",
            ]
        ),
        role="Project manager",
        system_prompt="You coordinate work across your team.",
        memory=SlidingWindowMemory(max_events=50),
    )

    # Workers — run in isolated child rooms
    researcher = Agent(
        "agent-researcher",
        provider=MockAIProvider(responses=["Research complete: found 3 key findings."]),
        role="Researcher",
        system_prompt="You research topics thoroughly.",
        memory=SlidingWindowMemory(max_events=50),
    )
    coder = Agent(
        "agent-coder",
        provider=MockAIProvider(responses=["Implementation complete: added the feature."]),
        role="Software engineer",
        system_prompt="You write clean, tested code.",
        memory=SlidingWindowMemory(max_events=50),
    )

    # No strategy — per-worker tools are injected, AI decides when to call them.
    kit = RoomKit(
        orchestration=Supervisor(
            supervisor=manager,
            workers=[researcher, coder],
        ),
    )

    # Transport channel
    ws = WebSocketChannel("ws-user")
    inbox: list[RoomEvent] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user", on_receive)
    kit.register_channel(ws)

    await kit.create_room(room_id="project-room")
    await kit.attach_channel("project-room", "ws-user")

    # --- Simulate conversation ------------------------------------------------

    # 1. User asks manager for help
    print("=== User asks for research ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Can you research the latest trends in AI?"),
        )
    )
    reply = find_reply(inbox, "agent-manager", mark)
    print(f"  Manager: {reply.content.body}")  # type: ignore[union-attr]

    # 2. Manager delegates to researcher (inline by default)
    print("\n=== Manager delegates to researcher ===")
    _room_id_var.set("project-room")
    result = await manager.tool_handler(
        "delegate_to_agent-researcher",
        {"task": "Research the latest trends in AI for 2025."},
    )
    parsed = json.loads(result)
    print(f"  Status: {parsed['status']}")
    print(f"  Worker: {parsed['worker']}")
    print(f"  Result: {parsed['result'][:80]}...")

    # 3. Show that workers are NOT in the main room
    print("\n=== Room bindings ===")
    bindings = await kit.store.list_bindings("project-room")
    print(f"  Channels in room: {[b.channel_id for b in bindings]}")
    print("  (Workers run in child rooms, not the main room)")

    # 4. Show injected per-worker tools
    print("\n=== Supervisor's tools ===")
    for tool in manager._injected_tools:
        print(f"  {tool.name}")

    # 5. Show conversation state
    print("\n=== Conversation State ===")
    room = await kit.get_room("project-room")
    state = get_conversation_state(room)
    print(f"  Phase: {state.phase}")
    print(f"  Active agent: {state.active_agent_id}")

    await kit.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
