"""Sequential content creation workflow.

Demonstrates ``strategy="sequential"`` — the framework chains workers
in order: researcher → writer. The coordinator talks to the user and
delegates work; the framework controls the execution flow.

    Human → Coordinator → [Researcher → Writer] → Coordinator → Human

Requires ``ANTHROPIC_API_KEY`` environment variable.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/orchestration_content_workflow.py
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

# Show task runner errors — important for debugging delegation
setup_logging("content_workflow")
logging.getLogger("roomkit").setLevel(logging.WARNING)
logging.getLogger("roomkit.tasks").setLevel(logging.DEBUG)


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")

    haiku_config = AnthropicConfig(
        api_key=env["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",
    )

    # --- Three agents --------------------------------------------------------

    coordinator = Agent(
        "agent-coordinator",
        provider=AnthropicAIProvider(haiku_config),
        role="Project coordinator",
        system_prompt="You coordinate content creation. Present the final article to the user.",
        memory=SlidingWindowMemory(max_events=50),
    )

    researcher = Agent(
        "agent-researcher",
        provider=AnthropicAIProvider(haiku_config),
        role="Research analyst",
        system_prompt=(
            "You are a research analyst. Research the given topic and "
            "provide 4-5 key findings. Be concise and factual."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    writer = Agent(
        "agent-writer",
        provider=AnthropicAIProvider(haiku_config),
        role="Content writer",
        system_prompt=(
            "You are a content writer. Write a clear, engaging article "
            "based on the research provided. Use markdown headings. "
            "Keep it under 500 words."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    # --- Supervisor setup ----------------------------------------------------

    kit = RoomKit(
        orchestration=Supervisor(
            supervisor=coordinator,
            workers=[researcher, writer],
            strategy="sequential",
            auto_delegate=True,
        ),
    )

    # --- State change hook — prints when agents are delegated to -------------

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

    await kit.create_room(room_id="content-room")
    await kit.attach_channel("content-room", "cli")

    await cli.run(
        kit,
        room_id="content-room",
        welcome=(
            "=== Content Creation Workflow ===\n"
            "Ask the coordinator to write an article. Type 'quit' to exit.\n"
        ),
    )

    await kit.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
