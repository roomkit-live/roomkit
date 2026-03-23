"""Interactive loop orchestration with writer and reviewer.

Demonstrates ``Loop`` orchestration with real LLM agents and CLI.
A writer produces content, a reviewer evaluates it, and the cycle
repeats until the reviewer approves or max iterations are reached.

    Human → Writer → Reviewer → Writer → ... → Reviewer (approves) → Human

Requires ``ANTHROPIC_API_KEY`` environment variable.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/orchestration_loop_cli.py
"""

from __future__ import annotations

import asyncio
import logging

from shared.env import require_env

from roomkit import Agent, CLIChannel, HookExecution, HookTrigger, Loop, RoomKit
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.orchestration.state import get_conversation_state
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig

logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("roomkit").setLevel(logging.WARNING)
logging.getLogger("roomkit.orchestration.strategies.loop").setLevel(logging.INFO)


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")

    haiku_config = AnthropicConfig(
        api_key=env["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",
    )

    writer = Agent(
        "agent-writer",
        provider=AnthropicAIProvider(haiku_config),
        role="Technical writer",
        system_prompt=(
            "You write concise technical content. Write immediately "
            "without asking questions. Keep it short and focused."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    reviewer = Agent(
        "agent-reviewer",
        provider=AnthropicAIProvider(haiku_config),
        role="Content reviewer",
        system_prompt=(
            "You are a strict content reviewer. Review for quality, "
            "clarity, accuracy, and conciseness. On the first review, "
            "always request at least one improvement. Only say APPROVED "
            "after seeing a revision that addresses your feedback."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    kit = RoomKit(
        orchestration=Loop(
            agent=writer,
            reviewer=reviewer,
            max_iterations=3,
            auto_cycle=True,
        ),
    )

    # --- Observability hooks -------------------------------------------------

    @kit.hook(HookTrigger.ON_PHASE_TRANSITION, execution=HookExecution.ASYNC)
    async def on_phase(event: RoomEvent, ctx: RoomContext) -> None:
        m = event.metadata
        from_phase = m.get("from_phase", "?")
        to_phase = m.get("to_phase", "?")
        reason = m.get("reason", "")
        print(f"\n\033[35m[phase] {from_phase} → {to_phase} ({reason})\033[0m")

    @kit.hook(HookTrigger.ON_HANDOFF, execution=HookExecution.ASYNC)
    async def on_handoff(event: RoomEvent, ctx: RoomContext) -> None:
        room = await kit.get_room(event.room_id)
        state = get_conversation_state(room)
        iteration = state.context.get("_loop_iteration", 0)
        approved = state.context.get("_loop_approved", False)
        print(f"\033[35m[loop] iteration={iteration} approved={approved}\033[0m")

    cli = CLIChannel("cli")
    kit.register_channel(cli)

    await kit.create_room(room_id="loop-room")
    await kit.attach_channel("loop-room", "cli")

    await cli.run(
        kit,
        room_id="loop-room",
        welcome=(
            "=== Loop Orchestration ===\n"
            "Ask the writer to produce content. The reviewer will evaluate it.\n"
            "Type 'quit' to exit.\n"
        ),
    )

    # Show final state
    room = await kit.get_room("loop-room")
    state = get_conversation_state(room)
    print(
        f"\nFinal state: approved={state.context.get('_loop_approved')}, "
        f"iterations={state.context.get('_loop_iteration')}, "
        f"handoffs={state.handoff_count}"
    )

    await kit.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
