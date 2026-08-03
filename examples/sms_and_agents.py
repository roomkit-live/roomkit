"""A colleague on SMS in the same Room as your coding agent.

    terminal ──▶ CLIChannel "you" ──▶ Room ──┬──▶ AIChannel  "assistant"
                                              └──▶ SMSChannel "sms"

You are working with an agent. A colleague texts you. Both live in the same
Room, and the point of this example is that neither drowns the other:

- **Her message does not wake the agent.** The SMS binding's visibility is
  ``"transport"``, so what she writes reaches the humans in the room and no
  intelligence channel — a confidentiality boundary, not just noise control.
- **The agent's answers do not reach her phone.** Its binding's visibility
  is your console, so its prose, its thinking and its tool activity stay
  where you are reading them.
- **You choose, per line, who sees what.** By default everything you type is
  visible to the whole room, colleague included. ``@marie`` scopes a line to
  her alone — question *and* the answer it might get.

A leading ``@`` names someone, and what it *does* follows from who they are:
an agent is **asked** to act (``addressed_to``), a person is **written to**
(``visibility``). You do not solicit a colleague, and no agent should answer
in her place — so a line to Marie addresses nobody.
- **The transcript names people.** She shows as ``@marie · sms``, not as the
  channel that carried her words.

No credentials: the SMS provider is RoomKit's mock, which prints what would
have been sent instead of sending it, and ``/sms`` fakes a message coming
back. The agent is a mock AI for the same reason — swap it for an
``ACPChannel`` (see ``acp_claude_code.py``) and nothing else changes.

Requires:
    pip install "roomkit[console]"

Run with:
    CONSOLE=1 uv run python examples/sms_and_agents.py

At the prompt:
    what does this project do?     to the agent — Marie sees it too
    @marie running 10 minutes late  to Marie alone — she sees it, no agent does
    @assistant summarise that        ask the agent explicitly
    /sms can you review my PR?     pretend Marie just texted you
    quit
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import console_enabled, setup_logging

from roomkit import (
    ChannelCategory,
    CLIChannel,
    RoomKit,
    SMSChannel,
    Visibility,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.participant import Participant
from roomkit.providers.ai import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider

COLLEAGUE_NUMBER = "+15551234567"
COLLEAGUE_NAME = "Marie"

# Who a leading @handle can name. A person is written *to*; an agent is
# *asked*. Same syntax, and the meaning follows from which side you are on.
PEOPLE = {"marie": "sms"}
AGENTS = {"assistant"}


class PrintingSMSProvider(MockSMSProvider):
    """The mock, made visible: what would leave shows up in the transcript."""

    async def send(self, event: RoomEvent, to: str, from_: str | None = None):
        result = await super().send(event, to, from_)
        body = getattr(event.content, "body", "") or "(no text)"
        print(f"\n  📱 → {to}: {body}\n")
        return result


async def main(args: argparse.Namespace) -> None:
    kit = RoomKit()

    cli = CLIChannel("you", markdown=True, console=console_enabled())
    assistant = AIChannel(
        "assistant",
        provider=MockAIProvider(
            responses=[
                "RoomKit orchestrates multi-channel conversations — rooms, "
                "channels, hooks and pluggable backends.",
                "Yes: a room can hold humans and agents at once.",
            ]
        ),
        system_prompt="You are a concise assistant.",
    )
    sms = SMSChannel("sms", provider=PrintingSMSProvider())

    for channel in (cli, assistant, sms):
        kit.register_channel(channel)

    room_id = "work-session"
    await kit.create_room(room_id=room_id)
    await kit.attach_channel(room_id, "you")

    # The agent speaks to your console and to nobody else: its answers, its
    # thinking and its tool activity all carry this scope.
    await kit.attach_channel(
        room_id,
        "assistant",
        category=ChannelCategory.INTELLIGENCE,
        visibility="you",
    )

    # What Marie writes reaches the humans in the room and no agent. Her words
    # never enter a model's context unless you decide otherwise. The recipient
    # address lives on the binding — that is where the transport reads it.
    await kit.attach_channel(
        room_id,
        "sms",
        visibility=Visibility.TRANSPORT,
        metadata={"phone_number": COLLEAGUE_NUMBER},
    )

    # A name for the transcript: without a participant the console can only
    # show the channel, and "@marie · sms" beats "@marie" meaning the wire.
    await kit.store.add_participant(
        Participant(
            id=COLLEAGUE_NUMBER,
            room_id=room_id,
            channel_id="sms",
            display_name=COLLEAGUE_NAME,
        )
    )

    def mention(line: str) -> tuple[str, str] | None:
        """Split a leading ``@handle`` off the line, if it names someone here."""
        if not line.startswith("@"):
            return None
        handle, _, rest = line[1:].partition(" ")
        handle = handle.strip().casefold()
        if handle not in PEOPLE and handle not in AGENTS:
            return None
        return handle, rest.strip()

    def strip_mention(line: str) -> TextContent | None:
        """The handle addresses; it is not part of what you said."""
        named = mention(line)
        return TextContent(body=named[1] if named else line)

    def asked(line: str) -> list[str] | None:
        """@ names someone — what it *means* follows from who they are.

        An agent is asked to act. A person is not: you do not solicit a
        colleague, you write to them, and no agent should answer in their
        place either.
        """
        named = mention(line)
        if named is None:
            return None  # unaddressed: every agent in the room may answer
        handle, _ = named
        return [] if handle in PEOPLE else [handle]

    def scope(line: str) -> list[str] | None:
        """Who may see this line, and whatever answer it draws."""
        named = mention(line)
        if named is None or named[0] not in PEOPLE:
            return None  # the default: the whole room, colleague included
        return ["you", PEOPLE[named[0]]]

    async def incoming_sms(text: str) -> None:
        """``/sms`` — pretend Marie just texted, to see where it lands."""
        if not text:
            print("\nUsage: /sms <what Marie writes>\n")
            return
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms",
                sender_id=COLLEAGUE_NUMBER,
                content=TextContent(body=text),
            )
        )

    try:
        await cli.run(
            kit,
            room_id=room_id,
            sender_id="you",
            content_factory=strip_mention,
            addressed_to=asked,
            visibility=scope,
            commands={"/sms": incoming_sms},
            welcome=(
                f"{COLLEAGUE_NAME} ({COLLEAGUE_NUMBER}) is in this room, on SMS.\n"
                "Type freely — the assistant answers, and she sees it too.\n"
                "'@marie <text>' writes to her alone; '@assistant <text>' asks the\n"
                "agent explicitly. '/sms <text>' fakes a text from her, 'quit' exits."
            ),
        )
    finally:
        await kit.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A colleague on SMS sharing a Room with your coding agent."
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging("sms_and_agents")
    try:
        asyncio.run(main(_parse_args()))
    except KeyboardInterrupt:
        pass
