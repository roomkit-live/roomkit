"""Interactive content creation workflow with 3 agents.

Demonstrates a supervisor-style workflow with a coordinator, a
researcher, and a writer — each running as a separate agent.
The user chats via an interactive CLI powered by ``CLIChannel``.

The flow:

    Human (CLI) → Coordinator (supervisor)
                    ↓ delegate_to_agent-researcher
                  Researcher (worker) → researches topic
                    ↓ returns findings
                  Coordinator ← receives research
                    ↓ delegate_to_agent-writer (with research)
                  Writer (worker) → writes article
                    ↓ returns article
                  Coordinator ← receives article
                    ↓ delivers to human
    Human ← Coordinator

Uses ``wait_for_result=True`` so the coordinator gets each worker's
output during the same tool-use turn and can chain calls.

Requires ``ANTHROPIC_API_KEY`` environment variable.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/orchestration_content_workflow.py
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

# Show task runner errors — important for debugging delegation
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

    coordinator = Agent(
        "agent-coordinator",
        provider=AnthropicAIProvider(haiku_config),
        role="Project coordinator",
        description="Coordinates content creation by delegating to researcher and writer",
        system_prompt=(
            "You are a project coordinator for content creation. "
            "When the user asks for an article:\n"
            "1. First, delegate research to the researcher using "
            "delegate_to_agent-researcher with the topic.\n"
            "2. Then, delegate article writing to the writer using "
            "delegate_to_agent-writer — pass the researcher's findings "
            "as the task.\n"
            "3. Present the writer's article to the user.\n\n"
            "Always use both workers in sequence. You coordinate, "
            "you don't write or research yourself."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    researcher = Agent(
        "agent-researcher",
        provider=AnthropicAIProvider(haiku_config),
        role="Research analyst",
        description="Researches topics and provides detailed findings",
        system_prompt=(
            "You are a research analyst. Research the given topic and "
            "provide 4-5 key findings with clear structure. Be concise "
            "and factual."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    writer = Agent(
        "agent-writer",
        provider=AnthropicAIProvider(haiku_config),
        role="Content writer",
        description="Writes clear, engaging articles based on research",
        system_prompt=(
            "You are a content writer. Write a clear, engaging, and "
            "well-structured article based on the research provided. "
            "Use markdown headings, make it accessible to a broad "
            "audience, and keep it under 500 words."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    # --- Supervisor setup ----------------------------------------------------

    kit = RoomKit(
        orchestration=Supervisor(
            supervisor=coordinator,
            workers=[researcher, writer],
            wait_for_result=True,
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
