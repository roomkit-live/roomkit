"""Interactive swarm orchestration with specialist agents.

Demonstrates ``Swarm`` with real LLM agents and CLI. Three specialist
agents handle different aspects of a customer conversation. Any agent
can hand off to any other — the AI decides when a topic change requires
a different specialist.

    Human → Sales ↔ Support ↔ Billing → Human

Requires ``ANTHROPIC_API_KEY`` environment variable.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/orchestration_swarm_cli.py
"""

from __future__ import annotations

import asyncio
import logging

from shared.env import require_env

from roomkit import Agent, CLIChannel, HookExecution, HookTrigger, RoomKit, Swarm
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.orchestration.handoff import HandoffMemoryProvider
from roomkit.orchestration.state import get_conversation_state
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig

logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("roomkit").setLevel(logging.WARNING)


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")

    haiku_config = AnthropicConfig(
        api_key=env["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",
    )

    # --- Three specialist agents ---------------------------------------------

    sales = Agent(
        "agent-sales",
        provider=AnthropicAIProvider(haiku_config),
        role="Sales agent",
        description="Handles product inquiries, pricing, and upselling",
        system_prompt=(
            "You are a sales agent. Help customers with product information "
            "and pricing. If the customer has a technical issue, hand off to "
            "support. If they have a billing question, hand off to billing."
        ),
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )

    support = Agent(
        "agent-support",
        provider=AnthropicAIProvider(haiku_config),
        role="Support agent",
        description="Handles technical issues and troubleshooting",
        system_prompt=(
            "You are a technical support agent. Help customers troubleshoot "
            "issues. If the issue is billing-related, hand off to billing. "
            "If they want to purchase something, hand off to sales."
        ),
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )

    billing = Agent(
        "agent-billing",
        provider=AnthropicAIProvider(haiku_config),
        role="Billing agent",
        description="Handles billing, invoices, and payment issues",
        system_prompt=(
            "You are a billing agent. Help customers with invoices, payments, "
            "and account charges. If they have a technical issue, hand off to "
            "support. If they want to buy more, hand off to sales."
        ),
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )

    # --- Swarm setup ---------------------------------------------------------

    kit = RoomKit(
        orchestration=Swarm(
            agents=[sales, support, billing],
            entry="agent-sales",
        ),
    )

    # --- Observability hooks -------------------------------------------------

    @kit.hook(HookTrigger.ON_HANDOFF, execution=HookExecution.ASYNC)
    async def on_handoff(event: RoomEvent, ctx: RoomContext) -> None:
        room = await kit.get_room(event.room_id)
        state = get_conversation_state(room)
        print(f"\n\033[35m[handoff] Active agent: {state.active_agent_id}\033[0m")

    cli = CLIChannel("cli")
    kit.register_channel(cli)

    await kit.create_room(room_id="swarm-room")
    await kit.attach_channel("swarm-room", "cli")

    await cli.run(
        kit,
        room_id="swarm-room",
        welcome=(
            "=== Swarm Orchestration ===\n"
            "Chat with the sales agent. It can hand off to support or billing.\n"
            "Type 'quit' to exit.\n"
        ),
    )

    room = await kit.get_room("swarm-room")
    state = get_conversation_state(room)
    print(f"\nFinal state: active={state.active_agent_id}, handoffs={state.handoff_count}")

    await kit.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
