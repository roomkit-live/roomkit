"""Public text witnesses for isolated, coalesced and cancelled thinker work."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

import pytest

from roomkit import Agent, AIChannel, HookExecution, HookTrigger, MockSpeakPolicy, RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import AIContext
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.speaking import Thinker, Thought, ThoughtEvent
from tests.test_framework import SimpleChannel


class HeldThinker(Thinker):
    """The first call per room really parks; later contexts use the next call."""

    def __init__(self) -> None:
        self.entered: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self.release: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self.cut: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self.calls: dict[str, list[tuple[Thought, AIContext]]] = defaultdict(list)
        self.closed = 0

    async def think(self, previous: Thought, context: AIContext) -> Thought:
        assert context.room is not None and context.room.room is not None
        room_id = context.room.room.id
        self.calls[room_id].append((previous, context))
        number = len(self.calls[room_id])
        if number == 1:
            self.entered[room_id].set()
            try:
                await self.release[room_id].wait()
            except asyncio.CancelledError:
                self.cut[room_id].set()
                raise
        return Thought(f"{room_id} thought {number}")

    async def close(self) -> None:
        self.closed += 1


async def _kit(channel_kind: type[AIChannel], thinker: Thinker) -> tuple[RoomKit, MockAIProvider]:
    provider = MockAIProvider(responses=["Unexpected answer"])
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(
        channel_kind(
            "ai",
            provider=provider,
            speak_policy=MockSpeakPolicy(["silent"]),
            thinker=thinker,
            think_wait=0,
        )
    )
    for room in ("one", "two"):
        await kit.create_room(room_id=room)
        await kit.attach_channel(room, "sms")
        await kit.attach_channel(room, "ai", category=ChannelCategory.INTELLIGENCE)
    return kit, provider


async def _say(kit: RoomKit, room: str, body: str) -> None:
    await asyncio.wait_for(
        kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id="alice", content=TextContent(body=body)),
            room_id=room,
        ),
        3,
    )


@pytest.mark.parametrize("channel_kind", [AIChannel, Agent], ids=["channel", "agent"])
async def test_real_messages_keep_room_thoughts_separate_and_coalesce_pending_contexts(
    channel_kind: type[AIChannel],
) -> None:
    thinker = HeldThinker()
    kit, provider = await _kit(channel_kind, thinker)
    reports: list[ThoughtEvent] = []
    reported = asyncio.Event()

    @kit.hook(HookTrigger.ON_THOUGHT, execution=HookExecution.ASYNC)
    async def observe(event: ThoughtEvent, context: Any) -> None:
        reports.append(event)
        if len(reports) == 3:
            reported.set()

    try:
        await _say(kit, "one", "first")
        await asyncio.wait_for(thinker.entered["one"].wait(), 3)
        await _say(kit, "two", "other room")
        await asyncio.wait_for(thinker.entered["two"].wait(), 3)
        await _say(kit, "one", "superseded")
        await _say(kit, "one", "latest")
        assert {room: len(calls) for room, calls in thinker.calls.items()} == {"one": 1, "two": 1}
        thinker.release["one"].set()
        thinker.release["two"].set()
        await asyncio.wait_for(reported.wait(), 3)
        await asyncio.wait_for(kit.close(), 3)
    finally:
        thinker.release["one"].set()
        thinker.release["two"].set()
        await asyncio.wait_for(kit.close(), 3)

    assert provider.calls == []
    assert {room: len(calls) for room, calls in thinker.calls.items()} == {"one": 2, "two": 1}
    first, latest = thinker.calls["one"]
    assert first[0] == Thought() and latest[0] == Thought("one thought 1")
    assert "latest" in str(latest[1].messages[-1].content)
    assert thinker.calls["two"][0][0] == Thought()
    assert all(
        context.room.room.id == room
        for room, calls in thinker.calls.items()
        for _, context in calls
    )
    assert sorted((e.room_id, e.thought.text, e.previous.text) for e in reports) == [
        ("one", "one thought 1", ""),
        ("one", "one thought 2", "one thought 1"),
        ("two", "two thought 1", ""),
    ]
    assert thinker.closed == 0  # caller-owned resource; the channel cancels its own calls


@pytest.mark.parametrize("channel_kind", [AIChannel, Agent], ids=["channel", "agent"])
async def test_close_cuts_a_thinker_that_really_entered_without_reporting_a_new_thought(
    channel_kind: type[AIChannel],
) -> None:
    thinker = HeldThinker()
    kit, provider = await _kit(channel_kind, thinker)
    reports: list[ThoughtEvent] = []

    @kit.hook(HookTrigger.ON_THOUGHT, execution=HookExecution.ASYNC)
    async def observe(event: ThoughtEvent, context: Any) -> None:
        reports.append(event)

    try:
        await _say(kit, "one", "listen to this")
        await asyncio.wait_for(thinker.entered["one"].wait(), 3)
        assert not thinker.release["one"].is_set()
        await asyncio.wait_for(kit.close(), 3)
        assert thinker.cut["one"].is_set()
    finally:
        thinker.release["one"].set()
        await asyncio.wait_for(kit.close(), 3)
    assert reports == [] and provider.calls == []
    assert len(thinker.calls["one"]) == 1 and thinker.closed == 0
