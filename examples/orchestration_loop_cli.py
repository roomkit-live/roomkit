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
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, setup_logging

from roomkit import (
    Agent,
    CLIChannel,
    HookExecution,
    HookTrigger,
    Loop,
    RoomContext,
    RoomEvent,
    RoomKit,
    get_conversation_state,
)
from roomkit.memory import SlidingWindowMemory
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig

setup_logging("loop_cli")
logging.getLogger("roomkit").setLevel(logging.WARNING)
logging.getLogger("roomkit.orchestration.strategies.loop").setLevel(logging.INFO)


def _approves(body: str) -> bool:
    """Whether a review's result, its submit_review verdict as JSON, approves."""
    try:
        verdict = json.loads(body)
    except ValueError:
        return False
    return isinstance(verdict, dict) and verdict.get("approved") is True


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")

    haiku_config = AnthropicConfig(
        api_key=env["ANTHROPIC_API_KEY"],
        model="claude-haiku-5-5",
    )

    # --- Producer agent ------------------------------------------------------

    writer = Agent(
        "agent-writer",
        provider=AnthropicAIProvider(haiku_config),
        role="Technical writer",
        system_prompt=(
            "You write technical content. On your first attempt, write "
            "a quick rough draft, and make mistake — prioritize speed over polish. When "
            "you receive revision feedback, improve based on that feedback."
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
            "Approve the content if it is clear and well-structured. "
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
            "Approve the content if all facts are correct. "
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
            "Approve the content if its style is appropriate and consistent. "
            "Otherwise, suggest improvements."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    # --- Loop with parallel reviewers ----------------------------------------

    kit = RoomKit(
        process_timeout=120.0,
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
            # A review's result is its submit_review verdict, as JSON.
            approved = _approves(body)
            color = "\033[32m" if approved else "\033[33m"
            status = "approved" if approved else "feedback"
            print(f"{color}[{status}] {agent} ({duration:.0f}ms)\033[0m")
        else:
            preview = body[:80] + "..." if len(body) > 80 else body
            print(f"\033[36m[produced] {agent} ({duration:.0f}ms) {preview}\033[0m")

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def on_loop_result(event: RoomEvent, ctx: RoomContext) -> None:
        if "approved" not in (event.metadata or {}):
            return
        approved = event.metadata["approved"]
        iteration = event.metadata.get("iteration", "?")
        if approved:
            print(f"\n\033[32m[result] Approved after {iteration} iteration(s)\033[0m")
        else:
            print(f"\n\033[31m[result] Not approved after {iteration} iteration(s)\033[0m")

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
