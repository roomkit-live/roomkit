"""Multi-agent orchestration pipeline example.

Demonstrates a 3-agent pipeline where conversations flow from
triage -> handler -> resolver, with handoff between agents.

The router ensures only the targeted agent processes each user
message. Chain-depth limiting handles AI-to-AI reentry safely.

Run with:
    uv run python examples/orchestration_pipeline.py
"""

from __future__ import annotations

import asyncio
import logging

# Suppress chain-depth warnings from AI-to-AI reentry (expected in multi-agent setups)
logging.getLogger("roomkit").setLevel(logging.ERROR)

from roomkit import (
    AIChannel,
    ChannelCategory,
    ConversationPipeline,
    ConversationState,
    HandoffHandler,
    HandoffMemoryProvider,
    HookExecution,
    HookTrigger,
    InboundMessage,
    PipelineStage,
    RoomKit,
    SlidingWindowMemory,
    TextContent,
    WebSocketChannel,
    get_conversation_state,
    set_conversation_state,
    setup_handoff,
)
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.mock import MockAIProvider

# --- Pipeline definition -----------------------------------------------------


pipeline = ConversationPipeline(
    stages=[
        PipelineStage(phase="triage", agent_id="agent-triage", next="handling"),
        PipelineStage(phase="handling", agent_id="agent-handler", next="resolution"),
        PipelineStage(phase="resolution", agent_id="agent-resolver", next=None),
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

    # Transport channel for user messages
    ws = WebSocketChannel("ws-user")
    inbox: list[RoomEvent] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user", on_receive)
    kit.register_channel(ws)

    # AI agents — each with a distinct mock response
    ai_triage = AIChannel(
        "agent-triage",
        provider=MockAIProvider(responses=["I'll transfer you to our specialist."]),
        system_prompt="You triage incoming requests.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    ai_handler = AIChannel(
        "agent-handler",
        provider=MockAIProvider(responses=["Let me resolve this for you."]),
        system_prompt="You handle requests.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    ai_resolver = AIChannel(
        "agent-resolver",
        provider=MockAIProvider(responses=["All done! Issue resolved."]),
        system_prompt="You resolve and close requests.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )

    for ch in [ai_triage, ai_handler, ai_resolver]:
        kit.register_channel(ch)

    # Generate router from pipeline and install as hook
    router = pipeline.to_router()
    kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.SYNC, priority=-100)(
        router.as_hook()
    )

    # Wire handoff into each agent (with transition enforcement from pipeline)
    handoff_handler = HandoffHandler(
        kit=kit,
        router=router,
        phase_map=pipeline.get_phase_map(),
        allowed_transitions=pipeline.get_allowed_transitions(),
    )
    for ch in [ai_triage, ai_handler, ai_resolver]:
        setup_handoff(ch, handoff_handler)

    # Create room and attach all channels
    await kit.create_room(room_id="support-room")
    await kit.attach_channel("support-room", "ws-user")
    for agent_id in ["agent-triage", "agent-handler", "agent-resolver"]:
        await kit.attach_channel("support-room", agent_id, category=ChannelCategory.INTELLIGENCE)

    # Initialize conversation state to the pipeline's first stage
    room = await kit.get_room("support-room")
    initial_state = ConversationState(phase="triage", active_agent_id="agent-triage")
    room = set_conversation_state(room, initial_state)
    await kit.store.update_room(room)

    # --- Simulate conversation ------------------------------------------------

    print("=== Phase 1: Triage ===")
    room = await kit.get_room("support-room")
    state = get_conversation_state(room)
    print(f"  Phase: {state.phase}, Active agent: {state.active_agent_id}")

    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="I need help with my billing."),
        )
    )
    reply = find_reply(inbox, "agent-triage", mark)
    print(f"  Triage replied: {reply.content.body}")  # type: ignore[union-attr]

    # Handoff: triage -> handler
    print("\n=== Handoff: triage -> handler ===")
    result = await handoff_handler.handle(
        room_id="support-room",
        calling_agent_id="agent-triage",
        arguments={
            "target": "agent-handler",
            "reason": "Billing issue needs specialist",
            "summary": "User has a billing question about their account.",
        },
    )
    print(f"  Accepted: {result.accepted}, New phase: {result.new_phase}")

    room = await kit.get_room("support-room")
    state = get_conversation_state(room)
    print(
        f"  State: phase={state.phase}, agent={state.active_agent_id}, "
        f"handoffs={state.handoff_count}"
    )

    # User message — now routes to handler
    print("\n=== Phase 2: Handling ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="My last invoice looks wrong."),
        )
    )
    reply = find_reply(inbox, "agent-handler", mark)
    print(f"  Handler replied: {reply.content.body}")  # type: ignore[union-attr]

    # Handoff: handler -> resolver
    print("\n=== Handoff: handler -> resolver ===")
    result = await handoff_handler.handle(
        room_id="support-room",
        calling_agent_id="agent-handler",
        arguments={
            "target": "agent-resolver",
            "reason": "Invoice corrected, needs final confirmation",
            "summary": "Adjusted invoice #4521. User confirmed the new amount.",
        },
    )
    print(f"  Accepted: {result.accepted}")

    room = await kit.get_room("support-room")
    state = get_conversation_state(room)
    print(
        f"  State: phase={state.phase}, agent={state.active_agent_id}, "
        f"handoffs={state.handoff_count}"
    )

    # Final message — routes to resolver
    print("\n=== Phase 3: Resolution ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Looks good, thanks!"),
        )
    )
    reply = find_reply(inbox, "agent-resolver", mark)
    print(f"  Resolver replied: {reply.content.body}")  # type: ignore[union-attr]

    # Show full phase history
    print("\n=== Phase History ===")
    room = await kit.get_room("support-room")
    state = get_conversation_state(room)
    for t in state.phase_history:
        print(f"  {t.from_phase} -> {t.to_phase} ({t.reason})")

    # Cleanup
    for ch in [ai_triage, ai_handler, ai_resolver]:
        await ch.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
