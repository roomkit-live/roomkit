"""Agent Skills — give AI channels specialized knowledge packages.

Demonstrates how to use the Agent Skills standard (https://agentskills.io)
with AIChannel. When skills are registered, RoomKit automatically provides
the AI with tools to activate skills and read their references. The AI
decides when to use them based on the conversation.

Uses CLIChannel for interactive exploration. Try asking:
  - "Review this function: def get_user(id): return db.query(f'SELECT * FROM users WHERE id={id}')"
  - "Write tests for a function that validates email addresses"

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/agent_skills.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.env import require_env
from shared.hooks import log_tool_call

from roomkit import (
    ChannelCategory,
    CLIChannel,
    HookTrigger,
    RoomKit,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.skills import SkillRegistry

SKILLS_DIR = Path(__file__).parent / "skills"


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")

    # --- Discover skills from examples/skills/ ---
    registry = SkillRegistry()
    count = registry.discover(SKILLS_DIR)
    print(f"Discovered {count} skills:")
    for meta in registry.all_metadata():
        print(f"  - {meta.name}: {meta.description}")

    # --- Set up RoomKit ---
    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=AnthropicAIProvider(AnthropicConfig(api_key=env["ANTHROPIC_API_KEY"])),
        system_prompt=(
            "You are a senior Python developer assistant.\n"
            "You have access to Agent Skills. When the user asks for code "
            "review or writing tests, activate the relevant skill first and "
            "read its references before answering."
        ),
        skills=registry,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    # Show skill tool invocations in the terminal
    @kit.hook(HookTrigger.ON_TOOL_CALL)
    async def show_tool_call(event, _ctx):
        return log_tool_call(event, label="skill")

    await kit.create_room(room_id="skills-room")
    await kit.attach_channel("skills-room", "cli")
    await kit.attach_channel("skills-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    await cli.run(
        kit,
        room_id="skills-room",
        welcome=(
            "\nSkills demo — the AI has access to: "
            + ", ".join(registry.skill_names)
            + "\nType a request and watch the AI use skill tools.\n"
        ),
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
