"""AI Planning — structured task tracking with _plan_tasks tool.

When ``enable_planning=True``, AIChannel exposes a ``_plan_tasks`` tool
that lets the AI create and update structured task lists. The current
plan is automatically injected into the system prompt on every turn so
the AI always knows where it left off.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/ai_planning.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import log_tool_call, require_env

from roomkit import (
    ChannelCategory,
    CLIChannel,
    HookTrigger,
    RoomKit,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")

    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=AnthropicAIProvider(AnthropicConfig(api_key=env["ANTHROPIC_API_KEY"])),
        system_prompt=(
            "You are a project planning assistant.\n"
            "When the user asks you to work on something with multiple steps, "
            "you MUST call the plan_tasks tool to create a structured plan. "
            "Do NOT describe the plan in text — always use the tool. "
            "Update the plan as you make progress by calling plan_tasks again "
            "with updated statuses (in_progress, completed)."
        ),
        enable_planning=True,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    @kit.hook(HookTrigger.ON_TOOL_CALL)
    async def show_tool_call(event, _ctx):
        return log_tool_call(event, tool_names=["plan_tasks"], label="plan")

    await kit.create_room(room_id="planning-room")
    await kit.attach_channel("planning-room", "cli")
    await kit.attach_channel(
        "planning-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE
    )

    await cli.run(
        kit,
        room_id="planning-room",
        welcome=(
            "\nPlanning demo — the AI can create and update task plans.\n"
            'Try: "Plan how to build a REST API with authentication"\n'
        ),
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
