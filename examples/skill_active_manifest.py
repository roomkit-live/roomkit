"""RoomKit — a host manifest that knows what the room already loaded.

``skills_in_prompt=False`` hands the skills manifest to the host, and a
catalogue is only half of what a manifest needs: ``SkillRegistry`` lists what is
*available*, never what a given room has already activated. Rendered from the
catalogue alone every row reads "available" — including the skill whose
instructions the system prompt is already carrying — so the manifest pushes the
model to load rules that are in front of it. The model obeys: an
``activate_skill`` round answered by an ack, and a user watching the same skill
load twice.

``AIChannel.active_skill_names(room_id)`` is the missing half. This example
renders the same room's manifest both ways, before and after one activation, so
the difference is a printed block rather than a paragraph. It runs against a
mock provider — no API key, no network.

Run with:
    uv run python examples/skill_active_manifest.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills import SkillRegistry

SKILLS_DIR = Path(__file__).parent / "skills"
ROOM_ID = "manifest-room"
SKILL = "code-review"


class ActivateThenAnswer(MockAIProvider):
    """Calls ``activate_skill`` on the first turn, then answers in text.

    Stands in for a real model doing what a manifest asks of it, so the example
    is deterministic: the interesting part is what the host renders next.
    """

    def __init__(self, skill_name: str) -> None:
        super().__init__(responses=["Reviewed — the query interpolates user input."])
        self._skill_name = skill_name
        self._activated = False

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if not self._activated:
            self._activated = True
            return AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    AIToolCall(
                        id="tc1", name="activate_skill", arguments={"name": self._skill_name}
                    )
                ],
            )
        return AIResponse(content=self.responses[0], finish_reason="stop")


def render_manifest(registry: SkillRegistry, active: set[str]) -> str:
    """The host's own manifest — the thing ``skills_in_prompt=False`` asks for."""
    lines = ["<available_skills>"]
    for meta in registry.all_metadata():
        state = "loaded" if meta.name in active else "available"
        lines.append(f"  <skill name={meta.name!r} state={state!r}>{meta.description}</skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def show(title: str, body: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}\n{body}")


async def main() -> None:
    registry = SkillRegistry()
    registry.discover(SKILLS_DIR)

    kit = RoomKit()
    ws = WebSocketChannel("ws-user")
    ai = AIChannel(
        "ai-assistant",
        provider=ActivateThenAnswer(SKILL),
        system_prompt="You are a code-review assistant.",
        skills=registry,
        # The host renders the manifest, so the channel does not put one in the
        # prompt. Active bodies are injected either way — that is runtime state
        # a host cannot know, which is the whole point of this example.
        skills_in_prompt=False,
    )
    kit.register_channel(ws)
    kit.register_channel(ai)

    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "ws-user")
    await kit.attach_channel(ROOM_ID, "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    show(
        "Before the first message — nothing is loaded yet",
        render_manifest(registry, ai.active_skill_names(ROOM_ID)),
    )

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(
                body="Review this: db.query(f'SELECT * FROM users WHERE id={id}')"
            ),
        )
    )

    active = ai.active_skill_names(ROOM_ID)
    print(f"\nThe model activated: {sorted(active)}")

    show(
        "After the activation — rendered from the catalogue alone (the bug)",
        render_manifest(registry, set()),
    )
    show(
        "After the activation — rendered with active_skill_names (the fix)",
        render_manifest(registry, active),
    )

    print(
        f"\nOnly the second block tells the model that '{SKILL}' is already binding.\n"
        "Rendered the first way, the manifest asks it to activate what it is\n"
        "already following — one wasted round, answered by an ack.\n"
    )

    # Runtime state, keyed on the room: another room has loaded nothing.
    print(f"active_skill_names('other-room') -> {ai.active_skill_names('other-room')}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
