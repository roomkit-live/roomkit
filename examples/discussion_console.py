"""Talk with a team of agents from a full-screen terminal: Discussion + DiscussionConsole.

Three agents share one room with you. Name one with @ to ask it; a message
that names nobody goes to the agents that asked you something, else to all of
them, one at a time. The right-hand panel shows each agent, its model and its
live state (speaking, next, listening, asking you); the bottom line shows the
speak queue.

  @analyst what could cause a sudden drop in sign-ups?
  /listen @critic     the critic only answers when you name it
  /talk @critic       it talks again
  /quit

Logs go to ``discussion_console.log`` while the screen is up.

Requires the console extra and an Anthropic key:
    pip install roomkit[console,anthropic]
    ANTHROPIC_API_KEY=... uv run python examples/discussion_console.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env

from roomkit import Agent, Discussion, RoomKit, WebSocketChannel
from roomkit.console import DiscussionConsole
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig

ROOM = "team-room"

TEAM = {
    "analyst": (
        "Analyst",
        "Looks at the facts first and says what they show",
        "You reason from data. Say what you would check and what it would tell you.",
    ),
    "builder": (
        "Builder",
        "Turns a direction into concrete next steps",
        "You propose concrete, small next steps and who should take them.",
    ),
    "critic": (
        "Critic",
        "Looks for what could go wrong",
        "You point out risks and weak assumptions, briefly and constructively.",
    ),
}


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")
    agents = [
        Agent(
            handle,
            provider=AnthropicAIProvider(
                AnthropicConfig(api_key=env["ANTHROPIC_API_KEY"], model="claude-sonnet-5-5")
            ),
            role=role,
            description=description,
            system_prompt=f"{prompt} Keep answers short: this is a chat.",
        )
        for handle, (role, description, prompt) in TEAM.items()
    ]

    kit = RoomKit()
    you = WebSocketChannel("you")
    kit.register_channel(you)
    await kit.create_room(
        room_id=ROOM,
        orchestration=Discussion(agents=agents, people=["you"], max_depth=6),
    )
    await kit.attach_channel(ROOM, "you")

    console = DiscussionConsole(
        kit,
        ROOM,
        channel_id="you",
        title="a team of three agents",
        log_file="discussion_console.log",
    )
    try:
        await console.run()
    finally:
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
