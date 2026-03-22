"""Parallel analysis workflow.

Demonstrates ``strategy="parallel"`` — the framework runs all workers
concurrently on the same task. The supervisor talks to the user and
delegates work; the framework controls the execution flow.

    Human → Supervisor → [Technical | Business] → Supervisor → Human

Requires ``ANTHROPIC_API_KEY`` environment variable.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/orchestration_parallel_tasks.py
"""

from __future__ import annotations

import asyncio
import logging

from shared.env import require_env

from roomkit import Agent, CLIChannel, HookExecution, HookTrigger, RoomKit, Supervisor
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig

logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("roomkit").setLevel(logging.WARNING)
logging.getLogger("roomkit.tasks").setLevel(logging.DEBUG)


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")

    haiku_config = AnthropicConfig(
        api_key=env["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",
    )

    # --- Three agents --------------------------------------------------------

    supervisor = Agent(
        "agent-supervisor",
        provider=AnthropicAIProvider(haiku_config),
        role="Project supervisor",
        system_prompt="You coordinate analysis. Present a combined summary to the user.",
        memory=SlidingWindowMemory(max_events=50),
    )

    technical = Agent(
        "agent-technical",
        provider=AnthropicAIProvider(haiku_config),
        role="Technical analyst",
        system_prompt=(
            "You are a technical analyst. Analyze the given topic: "
            "architecture, implementation, scalability, trade-offs. "
            "Be concise (3-4 points)."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    business = Agent(
        "agent-business",
        provider=AnthropicAIProvider(haiku_config),
        role="Business analyst",
        system_prompt=(
            "You are a business analyst. Analyze the given topic: "
            "market impact, competitive positioning, revenue, strategy. "
            "Be concise (3-4 points)."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    # --- Supervisor setup ----------------------------------------------------
    #
    # strategy="parallel" injects a single delegate_workers tool.
    # The framework runs both analysts concurrently via asyncio.gather.

    kit = RoomKit(
        orchestration=Supervisor(
            supervisor=supervisor,
            workers=[technical, business],
            strategy="parallel",
        ),
    )

    # --- State hooks ---------------------------------------------------------

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def on_delegated(event: RoomEvent, ctx: RoomContext) -> None:
        agent = event.metadata.get("agent_id", "?")
        task_id = event.metadata.get("task_id", "?")
        print(f"\n\033[35m[state] Delegating to {agent} (task {task_id})\033[0m")

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_completed(event: RoomEvent, ctx: RoomContext) -> None:
        agent = event.metadata.get("agent_id", "?")
        status = event.metadata.get("task_status", "?")
        duration = event.metadata.get("duration_ms", 0)
        print(f"\n\033[35m[state] {agent} completed ({status}, {duration:.0f}ms)\033[0m")

    cli = CLIChannel("cli")
    kit.register_channel(cli)

    await kit.create_room(room_id="analysis-room")
    await kit.attach_channel("analysis-room", "cli")

    await cli.run(
        kit,
        room_id="analysis-room",
        welcome=(
            "=== Parallel Analysis Workflow ===\n"
            "Ask the supervisor to analyze a topic. Both analysts run in parallel.\n"
            "Type 'quit' to exit.\n"
        ),
    )

    await kit.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
