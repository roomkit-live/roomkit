"""Loop orchestration strategy example.

Demonstrates a produce/review loop where a writer generates content,
a reviewer evaluates it, and the cycle repeats until the reviewer
calls ``approve_output`` or max iterations are reached.

Uses the ``Loop`` orchestration strategy which automatically wires
the handoff cycle and injects the ``approve_output`` tool into the
reviewer.

Run with:
    uv run python examples/orchestration_approval_loop.py
"""

from __future__ import annotations

import asyncio
import json
import logging

# Suppress chain-depth warnings from AI-to-AI reentry (expected in multi-agent setups)
logging.getLogger("roomkit").setLevel(logging.ERROR)

from roomkit import Agent, InboundMessage, Loop, RoomKit, TextContent, WebSocketChannel
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
    # Writer agent — produces content
    writer = Agent(
        "agent-writer",
        provider=MockAIProvider(
            responses=[
                "Draft v1: RoomKit is a Python library for multi-channel conversations.",
                "Draft v2: RoomKit is a pure async Python library for orchestrating "
                "multi-channel conversations with rooms, hooks, and pluggable backends.",
            ]
        ),
        role="Technical writer",
        description="Writes technical documentation and blog posts",
        system_prompt="You write concise technical content.",
        memory=SlidingWindowMemory(max_events=50),
    )

    # Reviewer agent — evaluates content and approves or requests changes
    reviewer = Agent(
        "agent-reviewer",
        provider=MockAIProvider(
            responses=[
                "Needs more detail: mention async and the plugin architecture.",
                "Excellent revision! Approving.",
            ]
        ),
        role="Content reviewer",
        description="Reviews content for quality, clarity, and accuracy",
        system_prompt="You review content. Approve when quality standards are met.",
        memory=SlidingWindowMemory(max_events=50),
    )

    # Loop strategy: writer produces → reviewer evaluates → repeat until approved
    kit = RoomKit(
        orchestration=Loop(
            agent=writer,
            reviewer=reviewer,
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

    # Create room — Loop wires handoff + approve_output automatically
    await kit.create_room(room_id="loop-room")
    await kit.attach_channel("loop-room", "ws-user")

    # --- Simulate produce/review cycle ----------------------------------------

    # 1. User triggers the writer
    print("=== Iteration 1: Writer produces ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Write a one-line description of RoomKit."),
        )
    )
    reply = find_reply(inbox, "agent-writer", mark)
    print(f"  Writer: {reply.content.body}")  # type: ignore[union-attr]

    # 2. Hand off to reviewer (in production, the writer's AI would do this)
    _room_id_var.set("loop-room")
    await writer.tool_handler(
        "handoff_conversation",
        {
            "target": "agent-reviewer",
            "reason": "Draft ready for review",
            "summary": "First draft of one-line RoomKit description.",
        },
    )

    # 3. Reviewer evaluates (first pass — requests changes)
    print("\n=== Iteration 1: Reviewer evaluates ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Please review the draft."),
        )
    )
    reply = find_reply(inbox, "agent-reviewer", mark)
    print(f"  Reviewer: {reply.content.body}")  # type: ignore[union-attr]

    # 4. Reviewer sends back to writer
    _room_id_var.set("loop-room")
    await reviewer.tool_handler(
        "handoff_conversation",
        {
            "target": "agent-writer",
            "reason": "Needs revision",
            "summary": "Add async and plugin architecture details.",
        },
    )

    room = await kit.get_room("loop-room")
    state = get_conversation_state(room)
    print(f"\n  Loop iteration: {state.context.get('_loop_iteration', 0)}")
    print(f"  Approved: {state.context.get('_loop_approved', False)}")

    # 5. Writer revises
    print("\n=== Iteration 2: Writer revises ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Please revise based on feedback."),
        )
    )
    reply = find_reply(inbox, "agent-writer", mark)
    print(f"  Writer: {reply.content.body}")  # type: ignore[union-attr]

    # Hand off to reviewer again
    _room_id_var.set("loop-room")
    await writer.tool_handler(
        "handoff_conversation",
        {
            "target": "agent-reviewer",
            "reason": "Revised draft ready",
            "summary": "Updated description with async and plugin details.",
        },
    )

    # 6. Reviewer evaluates (second pass — approves!)
    print("\n=== Iteration 2: Reviewer approves ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="How does the revision look?"),
        )
    )
    reply = find_reply(inbox, "agent-reviewer", mark)
    print(f"  Reviewer: {reply.content.body}")  # type: ignore[union-attr]

    # Reviewer calls approve_output (Loop strategy injected this tool)
    _room_id_var.set("loop-room")
    approval = await reviewer.tool_handler(
        "approve_output",
        {"reason": "Clear, accurate, and concise."},
    )
    print(f"  Approval result: {json.loads(approval)}")

    # --- Results -------------------------------------------------------------

    print("\n=== Final State ===")
    room = await kit.get_room("loop-room")
    state = get_conversation_state(room)
    print(f"  Approved: {state.context.get('_loop_approved')}")
    print(f"  Loop iteration: {state.context.get('_loop_iteration')}")
    print(f"  Handoff count: {state.handoff_count}")

    print("\n=== Reviewer's Tools ===")
    for tool in reviewer._injected_tools:
        print(f"  {tool.name}: {tool.description[:60]}...")

    # Cleanup
    await kit.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
