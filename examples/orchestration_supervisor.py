"""Supervisor orchestration strategy example.

Demonstrates a supervisor agent that delegates tasks to worker agents
running in child rooms. The supervisor handles all user interaction,
while workers execute tasks in the background and report results.

Uses the ``Supervisor`` orchestration strategy which automatically
injects ``delegate_to_<worker>`` tools and manages child rooms.

Run with:
    uv run python examples/orchestration_supervisor.py
"""

from __future__ import annotations

import asyncio
import json
import logging

# Suppress chain-depth warnings from AI-to-AI reentry (expected in multi-agent setups)
logging.getLogger("roomkit").setLevel(logging.ERROR)

from roomkit import Agent, InboundMessage, RoomKit, Supervisor, TextContent, WebSocketChannel
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.event import RoomEvent
from roomkit.orchestration.handoff import _room_id_var
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
    # Supervisor — handles all user interaction
    manager = Agent(
        "agent-manager",
        provider=MockAIProvider(
            responses=[
                "I'll have our researcher look into that.",
                "The research is complete. Here's what we found.",
            ]
        ),
        role="Project manager",
        description="Coordinates work by delegating to specialist workers",
        system_prompt="You manage tasks by delegating to workers.",
        memory=SlidingWindowMemory(max_events=50),
    )

    # Workers — run tasks in isolated child rooms (not attached to user room)
    researcher = Agent(
        "agent-researcher",
        provider=MockAIProvider(responses=["Research complete: found 3 key findings."]),
        role="Researcher",
        description="Researches topics and provides detailed analysis",
        system_prompt="You research topics thoroughly.",
        memory=SlidingWindowMemory(max_events=50),
    )
    coder = Agent(
        "agent-coder",
        provider=MockAIProvider(responses=["Implementation complete: added the feature."]),
        role="Software engineer",
        description="Writes and reviews code",
        system_prompt="You write clean, tested code.",
        memory=SlidingWindowMemory(max_events=50),
    )

    # Supervisor strategy: manager talks to user, delegates to workers.
    # Workers are registered on the kit but NOT attached to the user's room.
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

    # Create room — only the supervisor is attached, workers stay external
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

    # 2. Manager delegates to researcher
    print("\n=== Manager delegates to researcher ===")
    _room_id_var.set("project-room")
    result = await manager.tool_handler(
        "delegate_to_agent-researcher",
        {"task": "Research the latest trends in AI for 2025."},
    )

    parsed = json.loads(result)
    print(f"  Delegation result: {parsed}")

    # 3. Show that workers are NOT in the main room
    print("\n=== Room bindings ===")
    bindings = await kit.store.list_bindings("project-room")
    print(f"  Channels in room: {[b.channel_id for b in bindings]}")
    print("  (Workers run in child rooms, not the main room)")

    # 4. Show delegation tool availability
    print("\n=== Supervisor's delegation tools ===")
    for tool in manager._injected_tools:
        print(f"  Tool: {tool.name}")
        print(f"    Description: {tool.description[:60]}...")

    # 5. Show conversation state
    print("\n=== Conversation State ===")
    room = await kit.get_room("project-room")
    state = get_conversation_state(room)
    print(f"  Phase: {state.phase}")
    print(f"  Active agent: {state.active_agent_id}")

    # Cleanup
    await kit.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
