"""Interactive loop orchestration with multiple reviewers.

Demonstrates ``Loop`` with a writer and 3 parallel reviewers.
The writer produces content, all reviewers evaluate in parallel,
and the cycle repeats until all approve or max iterations are reached.

    Human → Writer → [Quality | Accuracy | Style] → Writer → ... → Human

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

    # --- Producer agent ------------------------------------------------------

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

    # --- Three parallel reviewers --------------------------------------------

    quality = Agent(
        "agent-quality",
        provider=AnthropicAIProvider(haiku_config),
        role="Quality reviewer",
        system_prompt=(
            "You review content for overall quality and clarity. "
            "If the content is clear and well-structured, say APPROVED. "
            "Otherwise, provide specific feedback."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    accuracy = Agent(
        "agent-accuracy",
        provider=AnthropicAIProvider(haiku_config),
        role="Accuracy reviewer",
        system_prompt=(
            "You review content for factual accuracy. "
            "If all facts are correct, say APPROVED. "
            "Otherwise, point out inaccuracies."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    style = Agent(
        "agent-style",
        provider=AnthropicAIProvider(haiku_config),
        role="Style reviewer",
        system_prompt=(
            "You review content for writing style and tone. "
            "If the style is appropriate and consistent, say APPROVED. "
            "Otherwise, suggest improvements."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    # --- Loop with parallel reviewers ----------------------------------------

    kit = RoomKit(
        orchestration=Loop(
            agent=writer,
            reviewers=[quality, accuracy, style],
            strategy="parallel",
            max_iterations=3,
        ),
    )

    # --- Observability hooks -------------------------------------------------

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def on_delegated(event: RoomEvent, ctx: RoomContext) -> None:
        agent = event.metadata.get("agent_id", "?")
        print(f"\n\033[35m[delegated] {agent}\033[0m")

    reviewer_ids = {"agent-quality", "agent-accuracy", "agent-style"}

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_completed(event: RoomEvent, ctx: RoomContext) -> None:
        agent = event.metadata.get("agent_id", "?")
        duration = event.metadata.get("duration_ms", 0)
        body = getattr(event.content, "body", "")
        if agent in reviewer_ids:
            approved = "APPROVED" in body.upper() if body else False
            color = "\033[32m" if approved else "\033[33m"
            status = "approved" if approved else "feedback"
            print(f"{color}[{status}] {agent} ({duration:.0f}ms)\033[0m")
        else:
            preview = body[:80] + "..." if len(body) > 80 else body
            print(f"\033[36m[produced] {agent} ({duration:.0f}ms) {preview}\033[0m")

    cli = CLIChannel("cli")
    kit.register_channel(cli)

    await kit.create_room(room_id="loop-room")
    await kit.attach_channel("loop-room", "cli")

    await cli.run(
        kit,
        room_id="loop-room",
        welcome=(
            "=== Multi-Reviewer Loop ===\n"
            "Ask the writer to produce content. 3 reviewers evaluate in parallel.\n"
            "Type 'quit' to exit.\n"
        ),
    )

    room = await kit.get_room("loop-room")
    state = get_conversation_state(room)
    print(
        f"\nFinal state: approved={state.context.get('_loop_approved')}, "
        f"iterations={state.context.get('_loop_iteration')}"
    )

    await kit.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
