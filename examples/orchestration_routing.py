"""Rule-based routing with ConversationRouter.

Demonstrates direct ConversationRouter usage (without ConversationPipeline)
with multiple routing rules, conditions, agent aliases, and a muted
supervisor.

The router picks an agent in this order: the active agent (affinity) while
it is in the room, else the first matching rule, else the default agent.
Phases 2-3 show affinity; phase 4 shows an intent rule deciding, once no
agent is active.

The supervisor is attached muted: it reads every exchange and its model
still runs on each, but what it says is stored BLOCKED, never delivered
(muting silences the voice, not the brain). The handoff log below comes
from an ON_HANDOFF hook, not from the supervisor.

Run with:
    uv run python examples/orchestration_routing.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    Agent,
    ChannelCategory,
    EventType,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.event import RoomEvent
from roomkit.orchestration.handoff import HandoffMemoryProvider
from roomkit.orchestration.router import ConversationRouter, RoutingConditions, RoutingRule
from roomkit.orchestration.state import get_conversation_state, save_conversation_state
from roomkit.providers.ai.mock import MockAIProvider

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


# --- Main --------------------------------------------------------------------


async def main() -> None:
    kit = RoomKit()

    # Transport channel
    ws = WebSocketChannel("ws-user")
    inbox: list[RoomEvent] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user", on_receive, room_id="support-room")
    kit.register_channel(ws)

    # AI agents — each specialized for a domain
    ai_triage = Agent(
        "agent-triage",
        provider=MockAIProvider(responses=["Let me route you to the right team."]),
        role="Triage agent",
        description="Routes incoming requests to billing or tech support",
        system_prompt="You triage incoming requests.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    ai_billing = Agent(
        "agent-billing",
        provider=MockAIProvider(
            responses=[
                "I can help with your invoice.",
                "Done: one day of service credited for the outage.",
            ]
        ),
        role="Billing specialist",
        description="Handles billing and invoice questions",
        system_prompt="You handle billing questions.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    ai_tech = Agent(
        "agent-tech",
        provider=MockAIProvider(responses=["Let me troubleshoot that for you."]),
        role="Tech support",
        description="Handles technical issues and troubleshooting",
        system_prompt="You handle technical issues.",
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )
    supervisor_ai = MockAIProvider(responses=["[Supervisor note: monitoring.]"])
    ai_supervisor = Agent(
        "agent-supervisor",
        provider=supervisor_ai,
        role="Supervisor",
        description="Monitors conversations for quality assurance",
        system_prompt="You monitor conversations for quality.",
    )

    for ch in [ai_triage, ai_billing, ai_tech, ai_supervisor]:
        kit.register_channel(ch)

    # --- Router with explicit rules (no pipeline) ----------------------------

    router = ConversationRouter(
        rules=[
            # In intake phase → triage handles
            RoutingRule(
                agent_id="agent-triage",
                conditions=RoutingConditions(phases={"intake"}),
                priority=0,
            ),
            # In handling phase with billing intent → billing agent
            RoutingRule(
                agent_id="agent-billing",
                conditions=RoutingConditions(
                    phases={"handling"},
                    intents={"billing"},
                ),
                priority=0,
            ),
            # In handling phase with tech intent → tech agent
            RoutingRule(
                agent_id="agent-tech",
                conditions=RoutingConditions(
                    phases={"handling"},
                    intents={"tech"},
                ),
                priority=0,
            ),
        ],
        default_agent_id="agent-triage",
        supervisor_id="agent-supervisor",
    )

    # One-liner: registers routing hook + wires handoff on all agents
    handoff_handler = router.install(
        kit,
        [ai_triage, ai_billing, ai_tech],
        agent_aliases={
            "billing": "agent-billing",
            "tech": "agent-tech",
            "triage": "agent-triage",
        },
        phase_map={
            "agent-billing": "handling",
            "agent-tech": "handling",
            "agent-triage": "intake",
        },
    )

    # --- Log handoffs ---------------------------------------------------------

    handoff_log: list[str] = []

    @kit.hook(HookTrigger.ON_HANDOFF, execution=HookExecution.ASYNC)
    async def log_handoff(event: RoomEvent, _ctx: object) -> None:
        meta = event.metadata or {}
        handoff_log.append(
            f"Handoff: {meta.get('from_agent')} -> {meta.get('to_agent')} "
            f"(phase={meta.get('new_phase')})"
        )

    # --- Create room and attach channels -------------------------------------

    await kit.create_room(room_id="support-room")
    await kit.attach_channel("support-room", "ws-user")
    for agent_id in ["agent-triage", "agent-billing", "agent-tech"]:
        await kit.attach_channel("support-room", agent_id, category=ChannelCategory.INTELLIGENCE)
    # Supervisor is muted: it reads every exchange, its answers are blocked
    await kit.attach_channel(
        "support-room",
        "agent-supervisor",
        category=ChannelCategory.INTELLIGENCE,
        muted=True,
    )

    # --- Simulate conversation -----------------------------------------------

    # 1. Intake — triage routes by default
    print("=== Phase 1: Intake (triage) ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Hi, I have a problem with my last invoice."),
        )
    )
    reply = find_reply(inbox, "agent-triage", mark)
    print(f"  Triage: {reply.content.body}")  # type: ignore[union-attr]

    # 2. Triage hands off to billing using alias
    print("\n=== Handoff: triage -> billing (via alias) ===")
    result = await handoff_handler.handle(
        room_id="support-room",
        calling_agent_id="agent-triage",
        arguments={
            "target": "billing",  # alias, not full channel ID
            "reason": "User has an invoice issue",
            "summary": "Customer reports incorrect last invoice amount.",
        },
    )
    print(f"  Accepted: {result.accepted}, New agent: {result.new_agent_id}")

    # 3. User message — routes to billing via affinity. The intent tag does
    #    not decide here: the active agent outranks every rule.
    print("\n=== Phase 2: Handling (billing via affinity) ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="The amount is $200 too high."),
            metadata={"intent": "billing"},
        )
    )
    reply = find_reply(inbox, "agent-billing", mark)
    print(f"  Billing: {reply.content.body}")  # type: ignore[union-attr]

    # 4. Billing hands off to tech (customer also has a technical issue)
    print("\n=== Handoff: billing -> tech (via alias) ===")
    result = await handoff_handler.handle(
        room_id="support-room",
        calling_agent_id="agent-billing",
        arguments={
            "target": "tech",
            "reason": "Invoice fixed, but user also reports webhook failures",
            "summary": "Invoice corrected. User says delivery webhooks stopped working.",
        },
    )
    print(f"  Accepted: {result.accepted}, New agent: {result.new_agent_id}")

    # 5. User message — routes to tech via affinity (intent tag ignored)
    print("\n=== Phase 3: Handling (tech via affinity) ===")
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="The webhooks return 401 since yesterday."),
            metadata={"intent": "tech"},
        )
    )
    reply = find_reply(inbox, "agent-tech", mark)
    print(f"  Tech: {reply.content.body}")  # type: ignore[union-attr]

    # 6. Tech is done and releases the conversation: still in the handling
    #    phase, with no active agent, so the rules decide the next message.
    print("\n=== Phase 4: Handling (billing via intent rule, no active agent) ===")
    room = await kit.get_room("support-room")
    state = get_conversation_state(room).transition(
        "handling", to_agent=None, reason="Webhooks fixed, conversation released"
    )
    await save_conversation_state(kit.store, "support-room", state)
    mark = len(inbox)
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Can I get a credit for the outage?"),
            metadata={"intent": "billing"},
        )
    )
    reply = find_reply(inbox, "agent-billing", mark)
    if reply is not None and isinstance(reply.content, TextContent):
        print(f"  Billing: {reply.content.body}")
    answered = sorted({e.source.channel_id for e in inbox[mark:] if e.type == EventType.MESSAGE})
    print(f"  Answered by: {answered}")

    # --- Results -------------------------------------------------------------

    print("\n=== Conversation State ===")
    room = await kit.get_room("support-room")
    state = get_conversation_state(room)
    print(f"  Phase: {state.phase}")
    print(f"  Active agent: {state.active_agent_id}")
    print(f"  Handoff count: {state.handoff_count}")

    print("\n=== Phase History ===")
    for t in state.phase_history:
        print(f"  {t.from_phase} -> {t.to_phase}: {t.from_agent} -> {t.to_agent} ({t.reason})")

    print("\n=== Handoff Log (ON_HANDOFF hook) ===")
    for entry in handoff_log:
        print(f"  {entry}")

    print("\n=== Muted Supervisor ===")
    delivered = sum(1 for e in inbox if e.source.channel_id == "agent-supervisor")
    print(f"  Model calls: {len(supervisor_ai.calls)}, delivered to the user: {delivered}")

    # Cleanup
    for ch in [ai_triage, ai_billing, ai_tech, ai_supervisor]:
        await ch.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
