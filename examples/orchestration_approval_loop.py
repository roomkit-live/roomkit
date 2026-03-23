"""Loop orchestration strategy example.

Demonstrates a produce/review loop where a writer generates content,
a reviewer evaluates it, and the cycle repeats until the reviewer
approves or max iterations are reached.

The framework controls the flow — agents just produce content.
No handoff tools, no approve_output tool.

Run with:
    uv run python examples/orchestration_approval_loop.py
"""

from __future__ import annotations

import asyncio
import logging

logging.getLogger("roomkit").setLevel(logging.ERROR)
logging.getLogger("roomkit.orchestration.strategies.loop").setLevel(logging.INFO)

from roomkit import Agent, InboundMessage, Loop, RoomKit, TextContent, WebSocketChannel
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.event import RoomEvent
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
    # Writer — produces content (two versions for the loop)
    writer = Agent(
        "agent-writer",
        provider=MockAIProvider(
            responses=[
                "RoomKit is a Python library for multi-channel conversations.",
                "RoomKit is a pure async Python library for orchestrating "
                "multi-channel conversations with rooms, hooks, and pluggable backends.",
            ]
        ),
        role="Technical writer",
        system_prompt="You write concise technical content.",
        memory=SlidingWindowMemory(max_events=50),
    )

    # Reviewer — first rejects, then approves
    reviewer = Agent(
        "agent-reviewer",
        provider=MockAIProvider(
            responses=[
                "Needs more detail: mention async and the plugin architecture.",
                "APPROVED. Clear, accurate, and concise.",
            ]
        ),
        role="Content reviewer",
        system_prompt="You review content for quality.",
        memory=SlidingWindowMemory(max_events=50),
    )

    # Loop strategy: writer → reviewer → writer → reviewer (approved)
    kit = RoomKit(
        orchestration=Loop(
            agent=writer,
            reviewers=[reviewer],
            max_iterations=3,
        ),
    )

    # Transport channel
    ws = WebSocketChannel("ws-user")
    inbox: list[RoomEvent] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user", on_receive)
    kit.register_channel(ws)

    await kit.create_room(room_id="loop-room")
    await kit.attach_channel("loop-room", "ws-user")

    # --- Run the loop --------------------------------------------------------

    print("=== Loop Orchestration ===\n")

    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Write a one-line description of RoomKit."),
        )
    )

    reply = find_reply(inbox, "agent-writer", mark)
    if reply:
        print(f"Final output: {reply.content.body}")  # type: ignore[union-attr]

    # Show final state
    room = await kit.get_room("loop-room")
    state = get_conversation_state(room)
    print(f"\nApproved: {state.context.get('_loop_approved')}")
    print(f"Iterations: {state.context.get('_loop_iteration')}")

    await kit.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
