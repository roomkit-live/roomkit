"""A turn's live ephemerals reach no one its answer would not reach (RFC §8.4, RMK-952).

Subscribers of a room's ephemerals are not bound channels: the owner cannot
test them against an answer's visibility, so a turn whose answer the room's
visibility narrows publishes none — its reasoning, its tool calls' arguments
and results would otherwise reach a customer's browser the answer is kept
from.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from roomkit import (
    Access,
    ChannelBinding,
    ChannelCategory,
    ChannelType,
    EventSource,
    InboundMessage,
    Room,
    RoomContext,
    RoomEvent,
    RoomKit,
    SMSChannel,
    TextContent,
)
from roomkit.channels.agent import Agent
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.turns import InMemoryTurnBackend, OwnerEvents, TurnClient, TurnExecutor
from roomkit.turns._outlet import TurnRealtime
from roomkit.turns._owner_between import publish_between_turns
from roomkit.turns._rooms import RoomRecords
from roomkit.turns.wire import Ephemeral, PreviousTurn, RoomPublication

CUSTOMER = "+15550100001"


async def _turn_ephemerals(
    *, visibility: str = "all", access: Access = Access.READ_WRITE, scope: str | None = None
) -> tuple[list[str], list[str]]:
    """The ephemerals a subscriber of the room receives during an agent's
    turn, and what the customer was texted."""
    sms = MockSMSProvider()
    kit = RoomKit()
    kit.register_channel(SMSChannel("sms", provider=sms))
    answer = AIResponse(content="the answer", thinking="the reasoning")
    kit.register_channel(
        Agent("assistant", provider=MockAIProvider(ai_responses=[answer], streaming=True))
    )
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms", metadata={"phone_number": CUSTOMER})
    await kit.attach_channel(
        "r",
        "assistant",
        category=ChannelCategory.INTELLIGENCE,
        visibility=visibility,
        access=access,
    )
    seen: list[str] = []
    received = asyncio.Event()

    async def subscriber(event) -> None:
        seen.append(event.type.value)
        received.set()

    await kit.realtime.subscribe_to_room("r", subscriber)
    try:
        await kit.process_inbound(
            InboundMessage(
                channel_id="sms",
                sender_id=CUSTOMER,
                content=TextContent(body="hi"),
                response_visibility=scope,
            ),
            room_id="r",
        )
        # A turn's ephemerals are published by a task of its own: wait for
        # the first, or let a gated turn's (none) the same bound.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(received.wait(), 0.05)
    finally:
        await kit.close()
    return seen, [item["event"].content.body for item in sms.sent]


async def test_a_turn_whose_answer_reaches_the_room_publishes_its_reasoning():
    seen, texted = await _turn_ephemerals()

    assert "thinking_delta" in seen
    assert texted == ["the answer"]


@pytest.mark.parametrize(
    "narrowed",
    [
        {"visibility": "intelligence"},
        {"scope": "intelligence"},
        {"access": Access.READ_ONLY},
    ],
    ids=["agent binding scoped", "trigger's answer scope", "agent cannot write"],
)
async def test_a_turn_whose_answer_the_room_narrows_publishes_no_ephemeral(narrowed):
    seen, texted = await _turn_ephemerals(**narrowed)

    assert texted == []
    assert seen == []


class _Events:
    """The owner's publishing side, with the channel's binding as the room
    holds it at publish time."""

    def __init__(self, *, reaches_room: bool) -> None:
        self.published: list[str] = []
        self._reaches_room = reaches_room

    async def publish(self, channel: str, event) -> None:
        self.published.append(event.type.value)

    async def emit(self, *args, **kwargs) -> None:
        return None

    async def reaches_room(self, room_id: str, channel_id: str) -> bool:
        return self._reaches_room


def _ephemeral(kind: EphemeralEventType) -> EphemeralEvent:
    return EphemeralEvent(
        room_id="r",
        type=kind,
        user_id="assistant",
        channel_id="assistant",
        data={"text": "after the turn"},
    )


def _publication(
    turn_id: str = "t1", kind: EphemeralEventType = EphemeralEventType.THINKING_DELTA
) -> RoomPublication:
    event = _ephemeral(kind)
    return RoomPublication(
        turn_id=turn_id,
        executor_id="x",
        room_id="r",
        channel_id="assistant",
        frame=Ephemeral(channel="room:r", event=event.to_dict()),
    )


_THINKING = EphemeralEventType.THINKING_DELTA


@pytest.mark.parametrize(
    ("named_turn", "binding_reaches_room", "kind", "published"),
    [
        ("t2", True, _THINKING, ["thinking_delta"]),
        # An earlier turn of the executor, whose trigger scoped its answers:
        # its work is not published under the later turn's scope.
        ("t1", True, _THINKING, []),
        ("t0", True, _THINKING, []),
        # The binding narrowed since the turn began: decided at publish time.
        ("t2", False, _THINKING, []),
        # An answer's text is published only while a trace reads its turn
        # (RFC §6.4, RMK-972).
        ("t2", True, EphemeralEventType.TEXT_DELTA, []),
    ],
    ids=[
        "the turn it names is open",
        "an earlier turn that was scoped",
        "a turn the owner does not know",
        "binding narrowed after the turn began",
        "an answer's text",
    ],
)
async def test_work_a_turn_began_publishes_after_it_as_that_turn_and_the_room_allow(
    named_turn, binding_reaches_room, kind, published
):
    previous: RoomRecords[PreviousTurn] = RoomRecords()
    earlier = PreviousTurn(turn_id="t1", executor_id="x")
    earlier._record_scope("t1", unscoped=False, response_depth=1)
    latest = PreviousTurn(turn_id="t2", executor_id="x")
    latest._inherit_scopes(earlier)
    latest._record_scope("t2", unscoped=True, response_depth=1)
    previous.put("assistant", "r", latest)
    events = _Events(reaches_room=binding_reaches_room)

    await publish_between_turns(previous, events, _publication(named_turn, kind))

    assert events.published == published


class _KitRealtime:
    def __init__(self) -> None:
        self.published: list[str] = []

    async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
        self.published.append(event.type.value)


async def test_an_answers_text_published_outside_a_turn_is_dropped():
    """Outside a turn, an executor publishes on the kit's realtime, with no
    owner to judge it: an answer's text, which only a trace's owner
    publishes (RFC §6.4, RMK-972), is dropped there, nothing else."""
    kit_realtime = _KitRealtime()
    realtime = TurnRealtime(lambda: kit_realtime)  # ty: ignore[invalid-argument-type]

    for kind in (EphemeralEventType.TEXT_DELTA, _THINKING):
        await realtime.publish_to_room("r", _ephemeral(kind))

    assert kit_realtime.published == ["thinking_delta"]


class _HostEvents(OwnerEvents):
    """A host's own publishing side, written before ``reaches_room()``."""

    def __init__(self) -> None:
        self.published: list[str] = []

    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        self.published.append(event.type.value)

    async def emit(self, event_type: str, **fields) -> None:
        return None


class _ReachingHostEvents(_HostEvents):
    async def reaches_room(self, room_id: str, channel_id: str) -> bool:
        return True


@pytest.mark.parametrize(
    ("events", "publishes"),
    [(_HostEvents, False), (_ReachingHostEvents, True)],
    ids=["without reaches_room", "with reaches_room"],
)
async def test_a_host_s_owner_events_publishes_a_turn_s_ephemerals_only_with_reaches_room(
    events: type[_HostEvents], publishes: bool
) -> None:
    """The migration RMK-952 asks of a host that gives ``TurnClient(events=)``
    its own ``OwnerEvents``: ``reaches_room()`` defaults to False, so
    without it no turn ephemeral is published at all."""
    answer = AIResponse(content="the answer", thinking="the reasoning")
    agent = Agent("assistant", provider=MockAIProvider(ai_responses=[answer], streaming=True))
    channels = {agent.channel_id: agent}
    owner = events()
    client = TurnClient(
        InMemoryTurnBackend(),
        executor=TurnExecutor(channels.get, hosts=channels.__contains__),
        events=owner,
    )
    binding = ChannelBinding(
        channel_id="assistant",
        room_id="r",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    trigger = RoomEvent(
        room_id="r",
        source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
        content=TextContent(body="hi"),
    )
    try:
        output = await client.run(
            trigger, binding, RoomContext(room=Room(id="r"), bindings=[binding])
        )
        assert [item async for item in output.response_stream]
    finally:
        await client.close()

    assert ("thinking_delta" in owner.published) is publishes
