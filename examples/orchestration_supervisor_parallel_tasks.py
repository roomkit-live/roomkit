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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, setup_logging

from roomkit import Agent, CLIChannel, HookExecution, HookTrigger, RoomKit, Supervisor
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig

setup_logging("parallel_tasks")
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
            "Need a technical deep dive report. Abuse of technical information. "
            "Don't return anything about business."
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
    # auto_delegate=True: framework triggers workers automatically.
    # strategy="parallel": both analysts run concurrently.
    # No tool, no AI choice — fully framework-driven.

    kit = RoomKit(
        orchestration=Supervisor(
            supervisor=supervisor,
            workers=[technical, business],
            strategy="parallel",
            auto_delegate=True,
            refine_task=False,
        ),
    )

    # --- Observability hooks ----------------------------------------------------
    #
    # Hooks use enriched metadata: task_id, child_room_id, parent_room_id,
    # agent_id, task_input, task_status, duration_ms, error.

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def on_delegated(event: RoomEvent, ctx: RoomContext) -> None:
        m = event.metadata
        agent = m.get("agent_id", "?")
        task_id = m.get("task_id", "?")
        child = m.get("child_room_id", "?")
        task_input = m.get("task_input", "")
        preview = task_input[:80] + "..." if len(task_input) > 80 else task_input
        print(
            f"\n\033[35m[delegated] {agent}\033[0m"
            f"\n  task:  {task_id}"
            f"\n  room:  {child}"
            f"\n  input: {preview}"
        )

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_completed(event: RoomEvent, ctx: RoomContext) -> None:
        m = event.metadata
        agent = m.get("agent_id", "?")
        status = m.get("task_status", "?")
        duration = m.get("duration_ms", 0)
        error = m.get("error")
        child = m.get("child_room_id", "?")
        color = "\033[32m" if status == "completed" else "\033[31m"
        body = getattr(event.content, "body", "")
        preview = body[:80] + "..." if len(body) > 80 else body
        print(
            f"\n{color}[completed] {agent}\033[0m"
            f"\n  status:   {status}"
            f"\n  duration: {duration:.0f}ms"
            f"\n  output:   {preview}"
        )
        if error:
            print(f"  error:    {error}")

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
