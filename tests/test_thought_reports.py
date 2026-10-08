"""What ON_THOUGHT reports when the agent speaks while its thinker is still
thinking (RMK-627, RFC §6.4): the thought a late call replaces is the one the
agent spoke with, emptied, not the one the call started from."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import MockSpeakPolicy, Thinker, Thought, ThoughtEvent
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIContext
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import respond

_BINDING = ChannelBinding(
    channel_id="ai1",
    room_id="r1",
    channel_type=ChannelType.AI,
    category=ChannelCategory.INTELLIGENCE,
)
_SMS = ChannelBinding(
    channel_id="sms1",
    room_id="r1",
    channel_type=ChannelType.SMS,
    category=ChannelCategory.TRANSPORT,
)
PRICE = Thought("They look for the price; I know it.", ("It costs 1,200 $ a year.",))
LATER = Thought("They settled on ten licences.", ("Ten gets the discount.",))


class _SlowSecondCall(Thinker):
    """The first call brings PRICE at once; the second takes 0.3 s and fails,
    keeps the thought (``same``), or brings LATER."""

    def __init__(self, second: str) -> None:
        self.second = second
        self.calls = 0

    async def think(self, previous: Thought, context: AIContext) -> Thought:
        self.calls += 1
        if self.calls == 1:
            return PRICE
        await asyncio.sleep(0.3)
        if self.second == "fails":
            raise RuntimeError("model down")
        return previous if self.second == "same" else LATER


def _context(*events: Any) -> RoomContext:
    return RoomContext(room=Room(id="r1"), bindings=[_BINDING, _SMS], recent_events=list(events))


@pytest.mark.parametrize(
    ("second", "late_report"),
    [
        ("fails", []),
        ("same", []),
        ("new", [(LATER.said(), PRICE.said())]),
    ],
)
async def test_a_call_that_ends_after_the_agent_spoke_reports_against_what_it_spoke_with(
    second: str, late_report: list[tuple[Thought, Thought]]
) -> None:
    thinker = _SlowSecondCall(second)
    channel = AIChannel(
        "ai1",
        provider=MockAIProvider(responses=["Yes?"]),
        speak_policy=MockSpeakPolicy(["silent", "silent", "silent", "speak"]),
        thinker=thinker,
        think_wait=0.05,
    )
    seen: list[ThoughtEvent] = []

    async def hook(event: ThoughtEvent) -> None:
        seen.append(event)

    channel._thought_hook = hook
    said = [make_event(body=b, channel_id="sms1", room_id="r1") for b in ("a", "b", "Nova?")]
    for event in said[:2]:  # the first call brings PRICE, the second starts
        await channel.on_event(event, _BINDING, _context(*said[:2]))
    await respond(channel, said[2], _BINDING, _context(*said))  # speaks during the call
    await asyncio.sleep(0.4)
    await channel.close()

    assert thinker.calls == 2
    assert [(e.thought, e.previous) for e in seen] == [
        (PRICE, Thought()),
        (PRICE.said(), PRICE),
        *late_report,
    ]
    assert seen[1].duration_ms is None
