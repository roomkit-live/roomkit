"""Interactive pipeline triage with CLI.

Demonstrates ``Pipeline`` with real LLM agents — a triage agent routes
the user to a specialist, who can then hand off to a resolver.

    Human → Triage → Handler → Resolver → Human

Requires ``ANTHROPIC_API_KEY`` environment variable.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/orchestration_pipeline_cli.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, setup_logging

from roomkit import Agent, CLIChannel, HookExecution, HookTrigger, Pipeline, RoomKit
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.orchestration.handoff import HandoffMemoryProvider
from roomkit.orchestration.state import get_conversation_state
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig

setup_logging("pipeline_cli")
logging.getLogger("roomkit").setLevel(logging.WARNING)


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")

    haiku_config = AnthropicConfig(
        api_key=env["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",
    )

    # --- Three agents in a linear pipeline -----------------------------------

    triage = Agent(
        "agent-triage",
        provider=AnthropicAIProvider(haiku_config),
        role="Triage receptionist",
        description="Routes incoming requests to the right specialist",
        system_prompt=(
            "You are a triage receptionist. Greet the user and understand "
            "their request. When you understand the issue, hand off to the "
            "handler specialist."
        ),
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )

    handler = Agent(
        "agent-handler",
        provider=AnthropicAIProvider(haiku_config),
        role="Request handler",
        description="Handles and investigates customer requests",
        system_prompt=(
            "You are a request handler. Investigate the customer's issue "
            "in detail. Once you have a solution, hand off to the resolver "
            "to finalize."
        ),
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )

    resolver = Agent(
        "agent-resolver",
        provider=AnthropicAIProvider(haiku_config),
        role="Resolution specialist",
        description="Confirms resolution and closes requests",
        system_prompt=(
            "You are a resolution specialist. Confirm the solution with "
            "the customer and close the request."
        ),
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=50)),
    )

    # --- Pipeline setup ------------------------------------------------------

    kit = RoomKit(
        orchestration=Pipeline(agents=[triage, handler, resolver]),
    )

    # --- Observability hooks -------------------------------------------------

    @kit.hook(HookTrigger.ON_HANDOFF, execution=HookExecution.ASYNC)
    async def on_handoff(event: RoomEvent, ctx: RoomContext) -> None:
        room = await kit.get_room(event.room_id)
        state = get_conversation_state(room)
        print(f"\n\033[35m[handoff] Active: {state.active_agent_id}\033[0m")

    cli = CLIChannel("cli")
    kit.register_channel(cli)

    await kit.create_room(room_id="triage-room")
    await kit.attach_channel("triage-room", "cli")

    await cli.run(
        kit,
        room_id="triage-room",
        welcome=(
            "=== Pipeline Triage ===\n"
            "Talk to the triage receptionist. They'll route you.\n"
            "Type 'quit' to exit.\n"
        ),
    )

    room = await kit.get_room("triage-room")
    state = get_conversation_state(room)
    print(f"\nFinal: active={state.active_agent_id}, handoffs={state.handoff_count}")

    await kit.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
