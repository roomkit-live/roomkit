"""Loop orchestration strategy example.

Demonstrates a produce/review loop where a writer generates content,
a reviewer evaluates it, and the cycle repeats until the reviewer
approves or max iterations are reached.

The framework controls the flow — agents just produce content. The
reviewer gives its verdict by calling the ``submit_review`` tool the loop
hands it (approved, feedback): its words never approve, so a review that
says "Not APPROVED yet" sends the work back.

Run with:
    uv run python examples/orchestration_approval_loop.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    Agent,
    EventType,
    InboundMessage,
    Loop,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.memory import SlidingWindowMemory
from roomkit.providers.ai import AIResponse, AIToolCall, MockAIProvider

# Warnings from everything, plus the loop strategy's per-iteration progress.
setup_logging("example.approval_loop", level=logging.WARNING)
logging.getLogger("roomkit.orchestration.strategies.loop").setLevel(logging.INFO)

# --- Helpers -----------------------------------------------------------------


def find_reply(events: list[RoomEvent], agent_id: str, start: int = 0) -> RoomEvent | None:
    """Find the first message from a specific agent after `start` index.

    Only MESSAGE events: an agent's channel also carries system notices
    (a handoff announcement, for one) that are not what it said.
    """
    for event in events[start:]:
        if event.source.channel_id == agent_id and event.type == EventType.MESSAGE:
            return event
    return None


def review(approved: bool, feedback: str) -> list[AIResponse]:
    """A scripted review: the reviewer's submit_review call, then its closing line."""
    verdict = AIToolCall(
        id=f"review-{approved}",
        name="submit_review",
        arguments={"approved": approved, "feedback": feedback},
    )
    return [AIResponse(content="", tool_calls=[verdict]), AIResponse(content="Reviewed.")]


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

    # Reviewer — first rejects, then approves, each time through submit_review
    reviewer = Agent(
        "agent-reviewer",
        provider=MockAIProvider(
            ai_responses=[
                *review(False, "Not APPROVED yet: mention async and the plugin architecture."),
                *review(True, ""),
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

    ws.register_connection("user", on_receive, room_id="loop-room")
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
    if reply is None:
        print("No output from the writer.")
    else:
        print(f"\nFinal output: {reply.content.body}")  # type: ignore[union-attr]
        # The loop's verdict rides on the delivered output.
        print(f"\nApproved: {reply.metadata.get('approved')}")
        print(f"Iterations: {reply.metadata.get('iteration')}")

    await kit.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
