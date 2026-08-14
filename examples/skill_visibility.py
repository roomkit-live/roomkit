"""Skill visibility — advertise less than you can activate.

Demonstrates the three visibility states of a registry skill:

- available   — listed in the prompt manifest and activatable (the default)
- unlisted    — absent from the manifest, still activatable by name
- unavailable — not activatable, named in the manifest with a reason

The catalogue here holds four skills but advertises two. `legacy-csv-import`
is niche enough to drown the manifest, so it is marked unlisted; a
BEFORE_BROADCAST hook plays the host's recommender and, when a message smells
like legacy CSV work, appends a note naming the skill — which the AI can then
activate even though it was never advertised. `deploy-helper` only works
through its scripts and this example configures no ScriptExecutor, so it is
marked unavailable with that reason and the AI can say why instead of
guessing at a name that answers "not found".

Uses CLIChannel for interactive exploration. Try:
  - "What can you help me with?"          -> only the two advertised skills
  - "Help me import a legacy CSV export"  -> the nudge fires, the unlisted
    skill activates
  - "Can you deploy the app?"             -> the AI explains why deploy-helper
    is unavailable

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/skill_visibility.py
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import log_tool_call, require_env

from roomkit import (
    ChannelCategory,
    CLIChannel,
    HookResult,
    HookTrigger,
    RoomKit,
    TextContent,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.event import RoomEvent
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.skills import SkillRegistry

SKILLS_DIR = Path(__file__).parent / "skills"

NUDGE_PATTERN = re.compile(r"\b(csv|legacy)\b", re.IGNORECASE)
NUDGE_NOTE = (
    "\n\n[host note: the unlisted skill 'legacy-csv-import' handles this — activate it by name.]"
)

_CYAN = "\033[36m"
_RESET = "\033[0m"


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")

    # --- Build the catalogue: four skills, two advertised ---
    registry = SkillRegistry()
    registry.discover(SKILLS_DIR)
    registry.mark_unlisted("legacy-csv-import")
    registry.mark_unavailable(
        "deploy-helper", "its scripts need a ScriptExecutor, which this host does not configure"
    )

    print(f"Registered (activatable): {registry.skill_names}")
    print(f"Advertised (in manifest): {registry.listed_names}")
    print(f"Unavailable with reason:  {list(registry.unavailable_skills)}")
    print("\nWhat the system prompt will carry:\n")
    print(registry.to_prompt_xml())

    # --- Set up RoomKit ---
    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=AnthropicAIProvider(
            AnthropicConfig(api_key=env["ANTHROPIC_API_KEY"], model="claude-opus-5")
        ),
        system_prompt=(
            "You are a data-engineering assistant.\n"
            "You have access to Agent Skills. When a skill fits the user's "
            "request, activate it first and follow its instructions. Host "
            "notes appended to a message may name a skill the manifest does "
            "not show — those names are activatable too."
        ),
        skills=registry,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    # --- The host's recommender: nudge with the unlisted skill's name ---
    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="skill_recommender")
    async def skill_recommender(event: RoomEvent, _ctx) -> HookResult:
        if event.source.channel_id != "cli" or not isinstance(event.content, TextContent):
            return HookResult.allow()
        if not NUDGE_PATTERN.search(event.content.body):
            return HookResult.allow()
        print(f"{_CYAN}  [recommender] nudging with 'legacy-csv-import'{_RESET}")
        modified = event.model_copy(
            update={"content": TextContent(body=event.content.body + NUDGE_NOTE)}
        )
        return HookResult.modify(modified)

    # Show skill tool invocations in the terminal
    @kit.hook(HookTrigger.ON_TOOL_CALL)
    async def show_tool_call(event, _ctx):
        return log_tool_call(event, label="skill")

    await kit.create_room(room_id="visibility-room")
    await kit.attach_channel("visibility-room", "cli")
    await kit.attach_channel(
        "visibility-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE
    )

    await cli.run(
        kit,
        room_id="visibility-room",
        welcome=(
            "\nSkill visibility demo — the AI is told about "
            f"{len(registry.listed_names)} of {registry.skill_count} activatable skills.\n"
            "Mention a legacy CSV import and watch the recommender surface the hidden one.\n"
        ),
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
