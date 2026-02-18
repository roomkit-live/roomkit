"""Pipeline with coder/reviewer loop.

Demonstrates a development pipeline where a reviewer can send work
back to the coder for fixes using ``can_return_to``. The conversation
loops between coding and review phases until approved, then proceeds
to the report stage.

Run with:
    uv run python examples/orchestration_loop.py
"""

from __future__ import annotations

import asyncio
import logging

# Suppress chain-depth warnings from AI-to-AI reentry (expected in multi-agent setups)
logging.getLogger("roomkit").setLevel(logging.ERROR)

from roomkit import (
    Agent,
    ChannelCategory,
    ConversationPipeline,
    ConversationState,
    HandoffMemoryProvider,
    InboundMessage,
    PipelineStage,
    RoomKit,
    SlidingWindowMemory,
    TextContent,
    WebSocketChannel,
    get_conversation_state,
    set_conversation_state,
)
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.mock import MockAIProvider

# --- Pipeline definition (with loop) ----------------------------------------


pipeline = ConversationPipeline(
    stages=[
        PipelineStage(phase="analysis", agent_id="agent-discuss", next="coding"),
        PipelineStage(phase="coding", agent_id="agent-coder", next="review"),
        PipelineStage(
            phase="review",
            agent_id="agent-reviewer",
            next="report",
            can_return_to={"coding"},  # Reviewer can send back to coder
        ),
        PipelineStage(phase="report", agent_id="agent-writer", next=None),
    ],
)


# --- Helpers -----------------------------------------------------------------


def find_reply(events: list[RoomEvent], agent_id: str, start: int = 0) -> RoomEvent | None:
    """Find the first event from a specific agent after `start` index."""
    for event in events[start:]:
        if event.source.channel_id == agent_id:
            return event
    return None


# --- Main --------------------------------------------------------------------


async def main() -> None:
    kit = RoomKit()

    # Transport channel
    ws = WebSocketChannel("ws-dev")
    inbox: list[RoomEvent] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("dev", on_receive)
    kit.register_channel(ws)

    # AI agents — each with multiple mock responses for the loop
    ai_discuss = Agent(
        "agent-discuss",
        provider=MockAIProvider(
            responses=["I've identified the root cause: PR #847 added auth middleware."]
        ),
        role="Bug analyst",
        description="Analyzes bugs and identifies root causes",
        system_prompt="You analyze bugs and identify root causes.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    ai_coder = Agent(
        "agent-coder",
        provider=MockAIProvider(
            responses=[
                "I've written the fix: skip auth for webhook callbacks.",
                "Fixed both issues: added IP allowlist and test coverage.",
            ]
        ),
        role="Code author",
        description="Writes code fixes based on analysis",
        system_prompt="You write code fixes.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    ai_reviewer = Agent(
        "agent-reviewer",
        provider=MockAIProvider(
            responses=[
                "Found 2 issues: missing IP allowlist and no test coverage.",
                "Code approved. All issues resolved.",
            ]
        ),
        role="Code reviewer",
        description="Reviews code changes for quality and correctness",
        system_prompt="You review code changes.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    ai_writer = Agent(
        "agent-writer",
        provider=MockAIProvider(responses=["Bug report published. JIRA: DEV-1234."]),
        role="Technical writer",
        description="Writes documentation and creates tickets",
        system_prompt="You write documentation and create tickets.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )

    for ch in [ai_discuss, ai_coder, ai_reviewer, ai_writer]:
        kit.register_channel(ch)

    # One-liner: registers routing hook + wires handoff on all agents
    router, handoff_handler = pipeline.install(kit, [ai_discuss, ai_coder, ai_reviewer, ai_writer])

    # Create room and attach channels
    await kit.create_room(room_id="dev-room")
    await kit.attach_channel("dev-room", "ws-dev")
    for agent_id in ["agent-discuss", "agent-coder", "agent-reviewer", "agent-writer"]:
        await kit.attach_channel("dev-room", agent_id, category=ChannelCategory.INTELLIGENCE)

    # Initialize to the pipeline's first stage
    room = await kit.get_room("dev-room")
    initial_state = ConversationState(phase="analysis", active_agent_id="agent-discuss")
    room = set_conversation_state(room, initial_state)
    await kit.store.update_room(room)

    # --- Simulate bug fix workflow -------------------------------------------

    # 1. Analysis phase
    print("=== Phase: Analysis ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-dev",
            sender_id="dev",
            content=TextContent(body="SMS webhooks return 401 since yesterday."),
        )
    )
    reply = find_reply(inbox, "agent-discuss", mark)
    print(f"  Discuss: {reply.content.body}")  # type: ignore[union-attr]

    # Handoff: discuss -> coder
    await handoff_handler.handle(
        room_id="dev-room",
        calling_agent_id="agent-discuss",
        arguments={
            "target": "agent-coder",
            "reason": "Root cause identified, needs code fix",
            "summary": "PR #847 auth middleware blocks Twilio callback. Need to skip auth.",
        },
    )

    # 2. Coding phase (first pass)
    print("\n=== Phase: Coding (pass 1) ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-dev",
            sender_id="dev",
            content=TextContent(body="Go ahead with the fix."),
        )
    )
    reply = find_reply(inbox, "agent-coder", mark)
    print(f"  Coder: {reply.content.body}")  # type: ignore[union-attr]

    # Handoff: coder -> reviewer
    await handoff_handler.handle(
        room_id="dev-room",
        calling_agent_id="agent-coder",
        arguments={
            "target": "agent-reviewer",
            "reason": "Fix ready for review",
            "summary": "Added auth bypass for /webhooks/twilio path using Twilio signature.",
        },
    )

    # 3. Review phase (first pass — finds issues)
    print("\n=== Phase: Review (pass 1) ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-dev",
            sender_id="dev",
            content=TextContent(body="What do you think?"),
        )
    )
    reply = find_reply(inbox, "agent-reviewer", mark)
    print(f"  Reviewer: {reply.content.body}")  # type: ignore[union-attr]

    # Handoff: reviewer -> coder (LOOP — sends back for fixes)
    print("\n  >> Loop: reviewer sends back to coder <<")
    result = await handoff_handler.handle(
        room_id="dev-room",
        calling_agent_id="agent-reviewer",
        arguments={
            "target": "agent-coder",
            "reason": "Two issues found, needs fixes",
            "summary": "Missing IP allowlist for webhook source. No test coverage.",
            "next_phase": "coding",  # explicit phase for the return trip
        },
    )
    print(f"  Accepted: {result.accepted}, Back to phase: {result.new_phase}")

    # 4. Coding phase (second pass — fixes issues)
    print("\n=== Phase: Coding (pass 2) ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-dev",
            sender_id="dev",
            content=TextContent(body="Please address the review feedback."),
        )
    )
    reply = find_reply(inbox, "agent-coder", mark)
    print(f"  Coder: {reply.content.body}")  # type: ignore[union-attr]

    # Handoff: coder -> reviewer (second review)
    await handoff_handler.handle(
        room_id="dev-room",
        calling_agent_id="agent-coder",
        arguments={
            "target": "agent-reviewer",
            "reason": "Both issues addressed",
            "summary": "Added Twilio IP allowlist and tests for webhook auth bypass.",
        },
    )

    # 5. Review phase (second pass — approves)
    print("\n=== Phase: Review (pass 2) ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-dev",
            sender_id="dev",
            content=TextContent(body="How does it look now?"),
        )
    )
    reply = find_reply(inbox, "agent-reviewer", mark)
    print(f"  Reviewer: {reply.content.body}")  # type: ignore[union-attr]

    # Handoff: reviewer -> writer (approved, move forward)
    await handoff_handler.handle(
        room_id="dev-room",
        calling_agent_id="agent-reviewer",
        arguments={
            "target": "agent-writer",
            "reason": "Code approved after second review",
            "summary": "Fix approved: auth bypass + IP allowlist + tests. Ready for docs.",
        },
    )

    # 6. Report phase
    print("\n=== Phase: Report ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-dev",
            sender_id="dev",
            content=TextContent(body="Please document everything."),
        )
    )
    reply = find_reply(inbox, "agent-writer", mark)
    print(f"  Writer: {reply.content.body}")  # type: ignore[union-attr]

    # --- Results -------------------------------------------------------------

    print("\n=== Full Phase History ===")
    room = await kit.get_room("dev-room")
    state = get_conversation_state(room)
    for i, t in enumerate(state.phase_history, 1):
        print(f"  {i}. {t.from_phase} -> {t.to_phase} ({t.reason})")

    print(f"\n  Total handoffs: {state.handoff_count}")
    print(f"  Final phase: {state.phase}")

    # Verify the loop is visible in history
    phases_visited = [t.to_phase for t in state.phase_history]
    print(f"  Phases visited: {' -> '.join(phases_visited)}")

    # Cleanup
    for ch in [ai_discuss, ai_coder, ai_reviewer, ai_writer]:
        await ch.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
