"""A call into a room from the room's own work (RFC §9.5.1 Reentrance, §8.3,
§10.5, §10.1 steps 16 and 18).

Made under the room lock, from an off-lock hook's body or from a message's
parse, a call is adopted by the event whose work made it and draws on that
event's reentry budget; made from a turn, a delivery or an ASYNC hook's body
the cascade fires once an event's delivery set has run, it is a new arrival
that draws on the budget of that cascade, at least as deep as the turn's
response. Each loop a host can write by calling into its room on every event
therefore stops, at the reentry cap or the chain-depth limit.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest

from roomkit import (
    ConferenceRealtimeConfig,
    MockConferenceBackend,
    ReentrantCallError,
    ReentryLoopCapError,
    RoomKit,
)
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.base import Channel
from roomkit.channels.conference import ConferenceChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.channels.voice import VoiceChannel
from roomkit.core.inbound_router import InboundRoomRouter
from roomkit.delivery.memory import InMemoryDeliveryBackend
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import DeleteContent, EditContent, RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter
from roomkit.models.streaming import SegmentBreakMarker
from roomkit.orchestration.strategies.discussion import Discussion
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall, ServedCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.mock import MockTelemetryProvider
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import AgentReasoningBackend
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.tts.mock import MockTTSProvider
from tests.conference.test_conference_realtime import ROOM, _Source, realtime_kit, until
from tests.conference.test_conference_tool_calls import LOOKUP
from tests.test_framework import AILikeChannel, SimpleChannel

# max_chain_depth=1: a reentry budget of 10 passes per cascade.
BUDGET = 10


def _message(body: str = "hi") -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body=body))


def _body(event: RoomEvent) -> str | None:
    return event.content.body if isinstance(event.content, TextContent) else None


async def _room(kit: RoomKit, *channels: SimpleChannel) -> RoomKit:
    await kit.create_room(room_id="r1")
    for channel in (SimpleChannel("sms"), *channels):
        kit.register_channel(channel)
        await kit.attach_channel("r1", channel.channel_id)
    return kit


async def _messages(kit: RoomKit) -> list[RoomEvent]:
    events = await kit.store.list_events(
        "r1", limit=1000, event_filter=EventFilter(include_blocked=True)
    )
    return [e for e in events if e.type == EventType.MESSAGE]


def _capped(events: list[RoomEvent]) -> list[RoomEvent]:
    return [e for e in events if e.blocked_by == "reentry_loop_cap"]


class TestAnInjectionMadeFromTheRoomsOwnWork:
    """§9.5.1: under the lock, or from an off-lock hook's body, the
    injection draws on the budget of the event whose work made it."""

    @pytest.mark.parametrize("needs_lock", [True, False], ids=["locked hook", "off-lock hook"])
    async def test_a_hook_that_injects_on_every_event_stops_at_the_cap(
        self, needs_lock: bool
    ) -> None:
        kit = await _room(RoomKit(max_chain_depth=1))

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="echo", needs_lock=needs_lock)
        async def echo(event: RoomEvent, ctx: RoomContext) -> HookResult:
            await kit.send_event("r1", "sms", TextContent(body="again"))
            return HookResult.allow()

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            events = await _messages(kit)
            assert len(events) == 1 + BUDGET + 1
            assert len(_capped(events)) == 1
        finally:
            await kit.close()

    async def test_a_replay_draws_nothing_and_returns_its_original(self) -> None:
        """A hook sends with the same idempotency key more times than its
        cascade's budget holds passes, then a new event: a replay is no new
        pass and draws nothing, so each replay returns the event the key
        committed, past the cap too, nothing stored and nothing raised, and
        the new event still has its pass (RFC §13.4, §8.3, §9.5.1)."""
        kit = await _room(RoomKit(max_chain_depth=1))
        sent: list[RoomEvent] = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="send")
        async def send(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) == "hi":
                for n in range(BUDGET + 2):
                    sent.append(
                        await kit.send_event(
                            "r1", "sms", TextContent(body=f"sent {n}"), idempotency_key="k"
                        )
                    )
                sent.append(await kit.send_event("r1", "sms", TextContent(body="new")))
            return HookResult.allow()

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            assert len(sent) == BUDGET + 3
            assert {e.id for e in sent[:-1]} == {sent[0].id}
            assert [(_body(e), e.blocked_by) for e in await _messages(kit)] == [
                ("sent 0", None),
                ("new", None),
                ("hi", None),
            ]
        finally:
            await kit.close()


def _order(events: list[RoomEvent]) -> list[str]:
    return [f"{e.index}:{e.source.channel_id}" for e in events]


class TestACallFromATaskALockedHookStarted:
    """§9.5.1, §13.5 (RMK-978): a task a locked hook starts acts for the hook
    while the hook runs, awaited or not; once the hook is over, its calls are
    new arrivals, though the event's pass still holds the room lock for its
    next hook."""

    async def test_while_the_hook_runs_its_call_is_adopted(self) -> None:
        kit = await _room(RoomKit(), SimpleChannel("ws"))
        sent = asyncio.Event()
        started: list[asyncio.Task[None]] = []

        async def notify() -> None:
            await kit.send_event("r1", "ws", TextContent(body="heads up"))
            sent.set()

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="notifier")
        async def notifier(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.source.channel_id == "sms":
                started.append(asyncio.create_task(notify()))  # never awaited
                await asyncio.wait_for(sent.wait(), timeout=1)
            return HookResult.allow()

        await kit.process_inbound(_message("hello"))
        await asyncio.gather(*started)
        assert _order(await _messages(kit)) == ["2:ws", "3:sms"]
        await kit.close()

    @pytest.mark.parametrize("needs_lock", [True, False], ids=["locked", "off-lock"])
    async def test_once_the_hook_is_over_its_call_is_a_new_arrival(self, needs_lock: bool) -> None:
        """Locked or off-lock: a hook stands for the work it began only while
        its body runs. An off-lock hook's task, calling while the check's
        next hook runs, stands in no admission order once the body is over:
        it takes a ticket and commits after the event."""
        kit = await _room(RoomKit(), SimpleChannel("ws"))
        over, calling, committed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        started: list[asyncio.Task[int]] = []

        async def notify() -> int:
            await over.wait()
            calling.set()
            sent = await kit.send_event("r1", "ws", TextContent(body="heads up"))
            committed.set()
            return sent.index

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="notifier", needs_lock=needs_lock)
        async def notifier(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.source.channel_id == "sms":
                started.append(asyncio.create_task(notify()))
                over.set()
            return HookResult.allow()

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="next", priority=10, needs_lock=needs_lock)
        async def next_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            # The same pass's, or check's, next hook runs on once the task
            # calls: an adopted call commits meanwhile, a new arrival waits.
            if event.source.channel_id == "sms":
                await calling.wait()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(committed.wait(), timeout=0.02)
            return HookResult.allow()

        await kit.process_inbound(_message("hello"))
        sent = await asyncio.wait_for(asyncio.gather(*started), timeout=1)
        assert (sent, _order(await _messages(kit))) == ([3], ["2:sms", "3:ws"])
        await kit.close()

    async def test_a_later_hook_awaiting_it_waits_until_its_timeout(self) -> None:
        """The task's call once its hook is over is an ordinary new arrival,
        neither refused nor adopted: a later hook of the same pass that
        awaits the task waits for the lock its pass holds, until its own
        timeout, which cancels the task it awaited; this fail-open hook's
        event then commits."""
        kit = await _room(RoomKit(), SimpleChannel("ws"))
        over = asyncio.Event()
        started: list[asyncio.Task[int]] = []

        async def notify() -> int:
            await over.wait()
            return (await kit.send_event("r1", "ws", TextContent(body="heads up"))).index

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="notifier")
        async def notifier(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.source.channel_id == "sms":
                started.append(asyncio.create_task(notify()))
            return HookResult.allow()

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="waiter", priority=10, timeout=0.05)
        async def waiter(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.source.channel_id == "sms":
                over.set()
                await started[0]
            return HookResult.allow()

        result = await kit.process_inbound(_message("hello"))
        await asyncio.gather(*started, return_exceptions=True)
        outcome = (result.blocked, _order(await _messages(kit)), started[0].cancelled())
        assert outcome == (False, ["2:sms"], True)
        await kit.close()


class _Parser(SimpleChannel):
    """A transport whose parse sends a message into its room every time."""

    def __init__(self, channel_id: str, kit: RoomKit, *, routed: bool = False) -> None:
        super().__init__(channel_id)
        self._kit = kit
        self.routed = routed
        self.answers: list[InboundResult | BaseException] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        if not self.answers or isinstance(self.answers[-1], InboundResult):
            again = _message("again")
            try:
                room = None if self.routed else "r1"
                self.answers.append(await self._kit.process_inbound(again, room_id=room))
            except ReentrantCallError as refused:
                self.answers.append(refused)
        return await super().handle_inbound(message, context)


class TestACallFromAParse:
    """The parse stands in its message's admission order (§9.5.1): what it
    sends into its room is adopted beside, on the message's budget."""

    async def test_a_parse_that_calls_into_its_room_for_every_message_stops_at_the_cap(
        self,
    ) -> None:
        kit = RoomKit(max_chain_depth=1)
        parser = _Parser("sms", kit)
        await kit.create_room(room_id="r1")
        kit.register_channel(parser)
        await kit.attach_channel("r1", "sms")
        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            refused = [a for a in parser.answers if isinstance(a, InboundResult) and a.blocked]
            assert [a.reason for a in refused] == ["reentry_loop_cap"]
            assert len(await _messages(kit)) == 1 + BUDGET
        finally:
            await kit.close()


class _Looping(SimpleChannel):
    """A transport whose every delivery sends a message into its room; once
    one is refused, it sends *then* into the room too, past the budget."""

    def __init__(self, channel_id: str, kit: RoomKit, then: InboundMessage | None = None) -> None:
        super().__init__(channel_id)
        self._kit = kit
        self._then = then
        self.refused = asyncio.Event()
        self.then_answer: InboundResult | Exception | None = None

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        result = await self._kit.process_inbound(_message("again"), room_id="r1")
        if result.reason == "reentry_loop_cap" and not self.refused.is_set():
            if self._then is not None:
                try:
                    self.then_answer = await self._kit.process_inbound(self._then, room_id="r1")
                except Exception as error:
                    self.then_answer = error
            self.refused.set()
        return ChannelOutput.empty()


class TestACallFromADelivery:
    async def test_a_transport_calling_into_its_room_on_every_delivery_stops_at_the_cap(
        self,
    ) -> None:
        """A new arrival draws on the budget of the cascade that delivers:
        with a budget of its own, the room would fill forever."""
        kit = RoomKit(max_chain_depth=1)
        looping = _Looping("loop", kit)
        await _room(kit, looping)
        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(looping.refused.wait(), 5)
            assert len(await _messages(kit)) == 1 + BUDGET
        finally:
            await kit.close()

    @pytest.mark.parametrize("channel_id", ["ghost", "late"], ids=["unregistered", "unattached"])
    async def test_a_call_naming_its_room_past_the_budget_is_refused_before_anything_else(
        self, channel_id: str
    ) -> None:
        """RFC §10.1 step 1, §9.5.1 Past the cap: a message that names its
        room is refused as it starts, before its channel is looked up or
        attached: nothing created or stored."""
        telemetry = MockTelemetryProvider()
        kit = RoomKit(max_chain_depth=1, telemetry=telemetry)
        then = InboundMessage(channel_id=channel_id, sender_id="u", content=TextContent(body="x"))
        looping = _Looping("loop", kit, then=then)
        await _room(kit, looping)
        kit.register_channel(SimpleChannel("late"))
        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(looping.refused.wait(), 5)
            answer = looping.then_answer
            assert isinstance(answer, InboundResult), answer
            assert (answer.blocked, answer.reason, answer.event) == (
                True,
                "reentry_loop_cap",
                None,
            )
            assert await kit.store.get_binding("r1", channel_id) is None
            assert len(await _messages(kit)) == 1 + BUDGET
            # Still traced: a loop the cap cut is visible.
            [span] = [
                span
                for span in telemetry.get_spans(SpanKind.INBOUND_PIPELINE)
                if span.name == "framework.inbound" and span.channel_id == channel_id
            ]
            assert (span.room_id, span.attributes["blocked"]) == ("r1", True)
        finally:
            await kit.close()

    @pytest.mark.parametrize(
        "opens", ["no-room", "a-room-id-the-store-lacks", "a-named-room-the-store-lacks"]
    )
    async def test_a_transport_opening_a_room_on_every_delivery_stops_at_the_cap(
        self, opens: str
    ) -> None:
        """A message the room's work sends that creates a room is that
        work's: it draws on the budget of the cascade that delivers, whether
        routing opens the room (no answer, or an id the store lacks) or the
        call names a room the store lacks, so a transport whose every
        delivery opens a room stops at the cap."""
        router = _OPENING_ROUTERS.get(opens, _NoRoom)()
        kit = RoomKit(max_chain_depth=1, inbound_router=router)
        opening = _Opening("opener", kit, naming=opens == "a-named-room-the-store-lacks")
        kit.register_channel(opening)

        @kit.hook(HookTrigger.ON_ROOM_CREATED)
        async def bind_opener(event: object, ctx: RoomContext) -> None:
            await kit.attach_channel(ctx.room.id, "opener")

        await _room(kit)
        try:
            await asyncio.wait_for(kit.process_inbound(_message(), room_id="r1"), 5)
            await asyncio.wait_for(opening.refused.wait(), 5)
            rooms = await kit.store.list_rooms()
            assert len(rooms) == 1 + BUDGET
        finally:
            await kit.close()

    async def test_a_routed_message_charged_for_a_room_created_meanwhile_is_charged_once(
        self,
    ) -> None:
        """RFC §8.3, §9.5.1: a routed message charged for creating a room
        another call created first, then routed again into the room its work
        delivers in, is not charged a second time there: with one pass left
        on the budget, it lands."""
        kit = RoomKit(
            max_chain_depth=1, store=_CreatedMeanwhile(), inbound_router=_AbsentThenTheRoom()
        )
        spender = _SpendsThenRoutes("spender", kit, named=BUDGET - 1)
        await _room(kit, spender)
        try:
            await asyncio.wait_for(kit.process_inbound(_message("go"), room_id="r1"), 5)
            routed = spender.routed
            assert routed is not None and not routed.blocked, routed
            assert routed.event is not None and routed.event.room_id == "r1"
        finally:
            await kit.close()

    async def test_a_delivery_s_call_into_another_existing_room_has_its_own_budget(
        self,
    ) -> None:
        """A call into an existing room the delivery does not deliver in is
        none of its work: each has a budget of its own."""
        kit = RoomKit(max_chain_depth=1)
        fanning = _Fanning("fan", kit, sends=BUDGET + 1)
        await _room(kit, fanning)
        await kit.create_room(room_id="r2")
        await kit.attach_channel("r2", "sms")
        try:
            await asyncio.wait_for(kit.process_inbound(_message(), room_id="r1"), 5)
            assert [e.blocked_by for e in fanning.sent] == [None] * (BUDGET + 1)
        finally:
            await kit.close()


class _NoRoom(InboundRoomRouter):
    """Finds no room for any message: each one opens a new room."""

    async def route(self, channel_id: str, channel_type: Any, **kwargs: Any) -> str | None:
        return None


class _FreshRoom(InboundRoomRouter):
    """Answers a room id the store does not hold for every message."""

    def __init__(self) -> None:
        self._rooms = 0

    async def route(self, channel_id: str, channel_type: Any, **kwargs: Any) -> str | None:
        self._rooms += 1
        return f"fresh-{self._rooms}"


class _CreatedMeanwhile(InMemoryStore):
    """Another call creates room ``fresh`` right after a routed message read
    it absent, its binding of ``sms`` naming another correspondent: the
    message, charged for the creation, reads it as existing, is no one it
    admits, and is routed again."""

    async def room_exists(self, room_id: str) -> bool:
        exists = await super().room_exists(room_id)
        if room_id == "fresh" and not exists:
            await super().create_room(Room(id="fresh"))
            await super().add_binding(
                ChannelBinding(
                    channel_id="sms",
                    room_id="fresh",
                    channel_type=ChannelType.SMS,
                    participant_id="someone-else",
                )
            )
        return exists


class _AbsentThenTheRoom(InboundRoomRouter):
    """Answers ``fresh``, an id the store lacks, then ``r1``."""

    def __init__(self) -> None:
        self._answers = ["fresh"]

    async def route(self, channel_id: str, channel_type: Any, **kwargs: Any) -> str | None:
        return self._answers.pop() if self._answers else "r1"


class _SpendsThenRoutes(SimpleChannel):
    """A transport whose delivery of ``go`` sends *named* messages naming its
    room, then one routed."""

    def __init__(self, channel_id: str, kit: RoomKit, *, named: int) -> None:
        super().__init__(channel_id)
        self._kit, self._named = kit, named
        self.routed: InboundResult | None = None

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if _body(event) == "go":
            for _ in range(self._named):
                await self._kit.process_inbound(_message("named"), room_id="r1")
            self.routed = await self._kit.process_inbound(_message("routed"))
        return ChannelOutput.empty()


_OPENING_ROUTERS: dict[str, Callable[[], InboundRoomRouter]] = {
    "no-room": _NoRoom,
    "a-room-id-the-store-lacks": _FreshRoom,
}


class _Opening(SimpleChannel):
    """A transport whose every delivery sends a message that opens a room:
    routed, or naming a fresh room id when *naming*."""

    def __init__(self, channel_id: str, kit: RoomKit, *, naming: bool = False) -> None:
        super().__init__(channel_id)
        self._kit = kit
        self._naming = naming
        self._named = 0
        self.refused = asyncio.Event()

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self._named += 1
        room_id = f"named-{self._named}" if self._naming else None
        result = await self._kit.process_inbound(_message("again"), room_id=room_id)
        if result.reason == "reentry_loop_cap":
            self.refused.set()
        return ChannelOutput.empty()


class _Fanning(SimpleChannel):
    """A transport whose delivery of the first message sends *sends*
    messages into room ``r2``."""

    def __init__(self, channel_id: str, kit: RoomKit, sends: int) -> None:
        super().__init__(channel_id)
        self._kit, self._sends = kit, sends
        self.sent: list[RoomEvent] = []

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if _body(event) == "hi":
            for n in range(self._sends):
                self.sent.append(
                    await self._kit.send_event("r2", "sms", TextContent(body=f"there {n}"))
                )
        return ChannelOutput.empty()


class _Injecting(AILikeChannel):
    """An agent that, at each turn, sends into its room through the SMS
    channel and answers nothing."""

    def __init__(self, channel_id: str, kit: RoomKit) -> None:
        super().__init__(channel_id)
        self._kit = kit
        self.turns = 0

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.turns += 1
        await self._kit.send_event("r1", "sms", TextContent(body=f"turn {self.turns}"))
        return ChannelOutput.empty()


class TestACallFromATurn:
    @pytest.mark.in_process(
        "the agent calls the kit it holds, whose context an in-memory turn shares with "
        "the owner; across Redis it goes through its TurnHost"
    )
    async def test_an_agent_injecting_at_each_turn_stops_at_the_chain_depth_limit(self) -> None:
        """A call made within a turn is at least as deep as the turn's
        response (RFC §8.3, §10.6): with a budget of 30, only the depth limit
        of 3 stops the agent this soon."""
        kit = RoomKit(max_chain_depth=3)
        agent = _Injecting("ai", kit)
        await _room(kit)
        kit.register_channel(agent)
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(_settled(kit), 5)
            assert agent.turns == 2
        finally:
            await kit.close()

    @pytest.mark.parametrize(
        "work",
        [
            pytest.param(
                "agent",
                marks=pytest.mark.in_process(
                    "the agent's task calls the kit it holds, whose context an in-memory "
                    "turn shares with the owner; across Redis it goes through its TurnHost"
                ),
            ),
            "tool-handler",
            "conference-handler",
        ],
    )
    async def test_work_a_turn_began_starts_fresh_once_the_turn_is_over(self, work: str) -> None:
        """A task a turn, a tool handler or a conference's tool call spawned,
        sending once that work is over, is work of its own: depth 0 and a
        budget of its own for each call, as a call from outside the room's
        work (RFC §8.3, §9.5.1)."""
        sends = 2 * BUDGET + 1
        kit, start, later = await _WORK[work](sends)
        try:
            await asyncio.wait_for(start(), 5)
            later.over.set()
            sent = await asyncio.wait_for(later.sent, 5)
            assert [(e.chain_depth, e.blocked_by) for e in sent] == [(0, None)] * sends
        finally:
            await kit.close()

    async def test_a_delivery_queued_from_a_tool_call_carries_the_turn_s_depth(self) -> None:
        """A ``deliver()`` a tool handler makes while the turn runs continues
        at least the turn's chain (RFC §8.3): queued, its item carries the
        depth of the turn's response, 1 for an answer to a person's message,
        to the worker that runs it once the turn is over (none runs here)."""
        backend = _RecordingBackend()
        kit = await _room(RoomKit(max_chain_depth=3, delivery_backend=backend))

        async def lookup(name: str, arguments: dict[str, Any]) -> str:
            await kit.deliver("r1", "Found it.", channel_id="sms", addressed_to=[])
            return "ok"

        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[AIToolCall(id="t1", name="lookup", arguments={})],
                ),
                AIResponse(content="Done.", finish_reason="stop"),
            ]
        )
        kit.register_channel(
            AIChannel(
                "ai",
                provider=provider,
                tool_handler=lookup,
                tools=[AITool(name="lookup", description="d")],
            )
        )
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            assert backend.depths == [1]
        finally:
            await kit.close()

    @pytest.mark.in_process(
        "the agent calls the kit it holds, whose context an in-memory turn shares with "
        "the owner; across Redis it goes through its TurnHost"
    )
    async def test_a_call_into_another_room_during_the_turn_has_its_own_budget(self) -> None:
        """A turn's call into another room is none of this room's work: a
        budget of its own for each call; it is still at least as deep as the
        turn's response, as any call made while the turn runs. The agent is
        asked below the depth limit of 2, with a budget of 20 per cascade."""
        kit = await _room(RoomKit(max_chain_depth=2))
        await kit.create_room(room_id="r2")
        await kit.attach_channel("r2", "sms")
        agent = _Elsewhere("ai", kit, sends=2 * BUDGET + 1)
        kit.register_channel(agent)
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        try:
            await asyncio.wait_for(kit.process_inbound(_message(), room_id="r1"), 5)
            assert [(e.chain_depth, e.blocked_by) for e in agent.sent] == [(1, None)] * (
                2 * BUDGET + 1
            )
        finally:
            await kit.close()


class _RecordingBackend(InMemoryDeliveryBackend):
    """An in-memory delivery queue that records the depth of each item."""

    def __init__(self) -> None:
        super().__init__()
        self.depths: list[int] = []

    async def enqueue(self, item: Any) -> None:
        self.depths.append(item.chain_depth)
        await super().enqueue(item)


class _Later:
    """Work started inside a turn or a tool call: a task that sends *sends*
    messages into *room_id* through *channel_id* once told the work is
    over, each addressed to nobody, so that no agent answers it."""

    def __init__(self, kit: RoomKit, room_id: str, channel_id: str, sends: int) -> None:
        self._kit, self._room_id, self._channel_id, self._sends = kit, room_id, channel_id, sends
        self.over = asyncio.Event()
        self.sent: asyncio.Future[list[RoomEvent]] = asyncio.get_running_loop().create_future()
        self._tasks: set[asyncio.Task[None]] = set()

    def start(self) -> None:
        if self._tasks or self.sent.done():
            return
        task = asyncio.create_task(self._send())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _send(self) -> None:
        await self.over.wait()
        sent = [
            await self._kit.send_event(
                self._room_id, self._channel_id, TextContent(body=f"later {n}"), addressed_to=[]
            )
            for n in range(self._sends)
        ]
        self.sent.set_result(sent)


type _Work = tuple[RoomKit, Callable[[], Awaitable[Any]], _Later]


async def _agent_work(sends: int) -> _Work:
    """An agent whose turn starts the work. The agent is asked below the
    depth limit of 2, with a budget of 20 per cascade."""
    kit = await _room(RoomKit(max_chain_depth=2))
    later = _Later(kit, "r1", "sms", sends)
    kit.register_channel(_Starting("ai", later))
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
    return kit, lambda: kit.process_inbound(_message()), later


async def _tool_work(sends: int) -> _Work:
    """An AI channel's tool handler starts the work, during the turn."""
    kit = await _room(RoomKit(max_chain_depth=2))
    later = _Later(kit, "r1", "sms", sends)

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        later.start()
        return "ok"

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="t1", name="lookup", arguments={})],
            ),
            AIResponse(content="Done.", finish_reason="stop"),
        ]
    )
    agent = AIChannel(
        "ai",
        provider=provider,
        tool_handler=lookup,
        tools=[AITool(name="lookup", description="d")],
    )
    kit.register_channel(agent)
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
    return kit, lambda: kit.process_inbound(_message()), later


async def _conference_work(sends: int) -> _Work:
    """A conference's tool call starts the work, at the depth 3 of the
    answer that issued it."""
    provider = MockRealtimeProvider()
    holder: list[_Later] = []

    async def handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        holder[0].start()
        return "done"

    config = ConferenceRealtimeConfig(provider=provider, tools=[LOOKUP], tool_handler=handler)
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    later = _Later(kit, ROOM, "src", sends)
    holder.append(later)
    session = await channel._realtime.ensure_session(ROOM)  # noqa: SLF001
    assert session is not None

    async def start() -> None:
        await kit.send_event(ROOM, "src", TextContent(body="agent says"), chain_depth=2)
        await provider.simulate_tool_call(session, "c1", "lookup", {})
        await until(lambda: bool(provider.tool_results))

    return kit, start, later


_WORK: dict[str, Callable[[int], Awaitable[_Work]]] = {
    "agent": _agent_work,
    "tool-handler": _tool_work,
    "conference-handler": _conference_work,
}


class _Starting(AILikeChannel):
    """An agent whose turn on the first message starts *later*."""

    def __init__(self, channel_id: str, later: _Later) -> None:
        super().__init__(channel_id)
        self._later = later

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if _body(event) == "hi":
            self._later.start()
        return ChannelOutput.empty()


class _Deferring(AILikeChannel):
    """An agent whose turn on the first message starts a task that sends
    *sends* messages into the room once the turn is over."""

    def __init__(self, channel_id: str, kit: RoomKit, *, sends: int = 1) -> None:
        super().__init__(channel_id)
        self._kit = kit
        self._sends = sends
        self.turn_over = asyncio.Event()
        self.sent: asyncio.Future[list[RoomEvent]] = asyncio.get_running_loop().create_future()
        self._tasks: set[asyncio.Task[None]] = set()

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if _body(event) == "hi":
            task = asyncio.create_task(self._later())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return ChannelOutput.empty()

    async def _later(self) -> None:
        await self.turn_over.wait()
        sent = [
            await self._kit.send_event("r1", "sms", TextContent(body=f"later {n}"))
            for n in range(self._sends)
        ]
        self.sent.set_result(sent)


class _Elsewhere(AILikeChannel):
    """An agent whose turn on the first message sends *sends* messages into
    room ``r2``."""

    def __init__(self, channel_id: str, kit: RoomKit, *, sends: int) -> None:
        super().__init__(channel_id)
        self._kit = kit
        self._sends = sends
        self.sent: list[RoomEvent] = []

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if _body(event) == "hi":
            for n in range(self._sends):
                self.sent.append(
                    await self._kit.send_event("r2", "sms", TextContent(body=f"there {n}"))
                )
        return ChannelOutput.empty()


async def _settled(kit: RoomKit, room_id: str = "r1") -> None:
    """Wait until the room's lane has run everything queued."""
    await kit._lanes.lane(room_id).drain()  # noqa: SLF001


async def _observed(kit: RoomKit, room_id: str = "r1") -> None:
    """Wait until the kit's observers and the room's lane have run all."""
    while kit._hook_engine.running_observers():  # noqa: SLF001
        await kit._hook_engine.drain_observers()  # noqa: SLF001
        await _settled(kit, room_id)
    await _settled(kit, room_id)


class _AnsweringModel(MockRealtimeProvider):
    """A speech-to-speech model that answers each text injected into its
    session with a call to ``lookup``, or a delegation to its hosted backend
    when it *delegates*, as its receive loop would, in a context of its own;
    a silent injection asks it nothing. It stops on its own after *rounds*
    answers, so a loop nothing ends still ends."""

    def __init__(self, rounds: int, *, delegates: bool = False) -> None:
        super().__init__()
        self._rounds = rounds
        self.delegates = delegates
        self.answers: list[asyncio.Task[None]] = []
        self.silences: list[bool] = []
        """Whether each text injected into its session was silent, in order."""

    def answer(self, session: Any) -> None:
        if len(self.answers) < self._rounds:
            n = len(self.answers)
            if self.delegates:
                call = self.simulate_delegation(session, f"d{n}", "hosted")
            else:
                call = self.simulate_tool_call(session, f"c{n}", "lookup", {})
            loop = asyncio.get_running_loop()
            self.answers.append(loop.create_task(call, context=contextvars.Context()))

    async def inject_text(self, session: Any, text: str, **kwargs: Any) -> Any:
        result = await super().inject_text(session, text, **kwargs)
        self.silences.append(bool(kwargs.get("silent")))
        if not kwargs.get("silent"):
            self.answer(session)
        return result


class _TalkingModel(MockRealtimeProvider):
    """A speech-to-speech model that answers each text asked of it by saying
    something (its transcription, a partial first when *partials*), from its
    own receive loop; it stops on its own after *rounds* answers."""

    def __init__(self, rounds: int, *, partials: bool) -> None:
        super().__init__()
        self._rounds = rounds
        self._partials = partials
        self.answers: list[asyncio.Task[None]] = []
        self.silences: list[bool] = []

    def answer(self, session: Any) -> None:
        if len(self.answers) < self._rounds:
            loop = asyncio.get_running_loop()
            say = self._say(session, len(self.answers) + 1)
            self.answers.append(loop.create_task(say, context=contextvars.Context()))

    async def _say(self, session: Any, n: int) -> None:
        if self._partials:
            await self.simulate_transcription(session, "well", "assistant", False)
        await self.simulate_transcription(session, f"answer {n}", "assistant", True)

    async def inject_text(self, session: Any, text: str, **kwargs: Any) -> Any:
        result = await super().inject_text(session, text, **kwargs)
        self.silences.append(bool(kwargs.get("silent")))
        if not kwargs.get("silent"):
            self.answer(session)
        return result


def _answering_agent(*, streaming: bool = False) -> AIChannel:
    """An agent that answers each message it is asked, "Answer n." in turn."""
    replies = [f"Answer {n}." for n in range(4 * BUDGET)]
    return AIChannel("ai", provider=MockAIProvider(responses=replies, streaming=streaming))


class _StreamingInputTTS(TTSProvider):
    """A TTS that reads a streamed reply sentence by sentence, one chunk each."""

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize(self, text: str, *, voice: str | None = None) -> Any:
        raise NotImplementedError

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        async for _ in text_stream:
            yield AudioChunk(data=b"\x00\x00" * 160, sample_rate=16000)


async def _voice_speaker(*, streaming: bool = False) -> RoomKit:
    """Room ``r1``: ``sms``, an agent, and a voice channel that speaks its
    answers, read sentence by sentence as they stream when *streaming*."""
    stt, backend = MockSTTProvider(transcripts=[]), MockVoiceBackend()
    tts = _StreamingInputTTS() if streaming else MockTTSProvider()
    kit = await _room(RoomKit(stt=stt, tts=tts, voice=backend, max_chain_depth=4))
    voice = VoiceChannel("voice", stt=stt, tts=tts, backend=backend)
    assert voice.supports_streaming_delivery is streaming
    kit.register_channel(voice)
    kit.register_channel(_answering_agent(streaming=streaming))
    await kit.attach_channel("r1", "voice")
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
    await kit.join("r1", "voice", participant_id="caller")
    return kit


async def _conference_speaker() -> RoomKit:
    """Room ``r1``: ``sms``, an agent, and a conference whose bot speaks its answers."""
    kit = await _room(RoomKit(max_chain_depth=4))
    conference = ConferenceChannel("conf", backend=MockConferenceBackend(), tts=MockTTSProvider())
    kit.register_channel(conference)
    kit.register_channel(_answering_agent())
    await kit.attach_channel("r1", "conf")
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
    return kit


_SPEAKERS: dict[str, Callable[[], Awaitable[RoomKit]]] = {
    "a voice channel": _voice_speaker,
    "a streamed reply": lambda: _voice_speaker(streaming=True),
    "a conference": _conference_speaker,
}


async def _until_the_model_rests(kit: RoomKit, model: _AnsweringModel, room_id: str) -> None:
    """Wait until every call the model made is answered (a delegation to
    its hosted backend waits for nothing), its observers have run and its
    room's lane is drained, and it made no call since."""
    while True:
        made = len(model.answers)
        await asyncio.gather(*model.answers)
        await until(lambda made=made: model.delegates or len(model.tool_results) == made)
        await _observed(kit, room_id)
        if len(model.answers) == made:
            return


async def _realtime_model_host(model: _AnsweringModel) -> tuple[RoomKit, str, str, Any, Any]:
    """A realtime voice channel serving *model* in room ``r1``, ``sms`` to post."""

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        return "found"

    kit = await _room(RoomKit(max_chain_depth=4))
    channel = RealtimeVoiceChannel(
        "rt",
        provider=model,
        transport=MockRealtimeTransport(),
        tools=[LOOKUP],
        tool_handler=lookup,
    )
    kit.register_channel(channel)
    await kit.attach_channel("r1", "rt")
    return kit, "r1", "sms", channel, await channel.start_session("r1", "u", object())


async def _conference_model_host(model: _AnsweringModel) -> tuple[RoomKit, str, str, Any, Any]:
    """A conference whose realtime configuration serves *model*, ``src`` to post."""

    async def lookup(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        return "found"

    config = ConferenceRealtimeConfig(provider=model, tools=[LOOKUP], tool_handler=lookup)
    channel = ConferenceChannel("conf", backend=MockConferenceBackend(), realtime=config)
    kit = RoomKit(max_chain_depth=4)
    kit.register_channel(channel)
    kit.register_channel(_Source("src", ChannelType.AI))
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    await kit.attach_channel(ROOM, "src")
    session = await channel._realtime.ensure_session(ROOM)  # noqa: SLF001
    return kit, ROOM, "src", channel, session


def _realtime_reported_outside(
    report: str,
) -> tuple[RealtimeVoiceChannel, MockRealtimeProvider, asyncio.Event]:
    """A realtime channel whose call is reported outside its run: its handler
    never returns, so the session's end cuts it; or a reasoning backend whose
    own provider serves the call it makes."""
    entered = asyncio.Event()

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        entered.set()
        await asyncio.Event().wait()
        return "never"

    if report == "the session's end":
        provider = MockRealtimeProvider()
        channel = RealtimeVoiceChannel(
            "rt",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=[LOOKUP],
            tool_handler=lookup,
        )
        return channel, provider, entered
    served = AIToolCall(
        id="p1", name="web_search", arguments={"q": "x"}, served=ServedCall(result="3 hits")
    )
    ai = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[served]),
            AIResponse(content="done"),
        ]
    )
    provider = MockRealtimeProvider(full_duplex=True)
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[LOOKUP],
        tool_handler=lookup,
        reasoning_backend=AgentReasoningBackend(Agent("reasoner", provider=ai)),
    )
    return channel, provider, entered


_MODEL_HOSTS: dict[
    str, Callable[[_AnsweringModel], Awaitable[tuple[RoomKit, str, str, Any, Any]]]
] = {
    "realtime": _realtime_model_host,
    "conference": _conference_model_host,
}


class _SilenceRecorder(MockRealtimeProvider):
    """A speech-to-speech model that notes whether each text it took was
    silent: one past the depth limit asks it nothing."""

    def __init__(self) -> None:
        super().__init__()
        self.silent: dict[str, bool] = {}

    async def inject_text(self, session: Any, text: str, **kwargs: Any) -> Any:
        self.silent[text.rsplit(" ", 1)[-1]] = bool(kwargs.get("silent"))
        return await super().inject_text(session, text, **kwargs)


class TestACallFromAnAsyncHook:
    """RFC §10.1 step 16, §10.3, §9.5.1: while the body of an ASYNC hook its
    cascade fires runs (``AFTER_BROADCAST`` once an event's delivery set has
    run, a change's mutation trigger), a call it makes into the room is a
    new arrival that draws on that cascade's budget, as a delivery's does; a
    task it starts calls as work of its own once the body has returned.

    Each looping hook stops itself well past the cap, so that a loop the cap
    does not stop fails at once rather than run forever."""

    async def test_a_hook_posting_after_every_event_stops_at_the_cap(self) -> None:
        """No agent in the room: the hook's own post fires it again."""
        kit = await _room(RoomKit(max_chain_depth=1))
        posted: list[RoomEvent] = []

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="notice")
        async def notice(event: RoomEvent, ctx: RoomContext) -> None:
            if event.type is EventType.MESSAGE and len(posted) <= 2 * BUDGET:
                posted.append(await kit.post("r1", f"Noted {len(posted)}."))

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(_settled(kit), 5)
            assert [e.blocked_by for e in posted] == [None] * BUDGET + ["reentry_loop_cap"]
        finally:
            await kit.close()

    async def test_an_observer_of_a_sync_trigger_posting_on_every_event_stops_at_the_cap(
        self,
    ) -> None:
        """RMK-978: an ASYNC hook on a SYNC trigger observes the payload the
        chain left, and is the cascade's work as an ``AFTER_BROADCAST``
        body is: what it sends into the room draws one of the cascade's
        hook calls, its own post's ``BEFORE_BROADCAST`` firing it again."""
        kit = await _room(RoomKit(max_chain_depth=1))
        posted: list[RoomEvent] = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.ASYNC, name="echo")
        async def echo(event: RoomEvent, ctx: RoomContext) -> None:
            if event.type is EventType.MESSAGE and len(posted) <= 2 * BUDGET:
                posted.append(await kit.send_event("r1", "sms", TextContent(body="posted")))

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(_observed(kit), 5)
            # In index order: the post past the cap returns first, unwaited.
            in_order = sorted(posted, key=lambda e: e.index)
            assert [e.blocked_by for e in in_order] == [None] * BUDGET + ["reentry_loop_cap"]
        finally:
            await kit.close()

    @pytest.mark.in_process(
        "the observer runs on the turn's executor and calls the owner's kit it holds, "
        "whose context an in-memory turn shares; across Redis it holds no frame of the "
        "room's work, and its posts take a budget of their own, as on 2d6e6267"
    )
    @pytest.mark.parametrize(
        "trigger",
        [HookTrigger.ON_TOOL_CALL, HookTrigger.BEFORE_TOOL_USE, HookTrigger.AFTER_TOOL_ROUND],
    )
    async def test_an_observer_of_a_trigger_per_tool_call_runs_as_the_turn_s_task(
        self, trigger: HookTrigger
    ) -> None:
        """RMK-978, as on 2d6e6267: the observers of the triggers that fire
        per tool call or per round are not the cascade's work, as the other
        observers are, but run as any task the turn starts. A post they
        make while the turn's delivery runs draws one of the cascade's
        passes and stops at the cap; a direct change draws nothing."""
        # The agent answers at depth 1: max_chain_depth=2, 20 passes.
        kit = await _room(RoomKit(max_chain_depth=2), SimpleChannel("log"))
        rounds = 2 * BUDGET + 5

        async def lookup(name: str, arguments: dict[str, Any]) -> str:
            return f"found {arguments['q']}"

        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[AIToolCall(id=f"t{n}", name="lookup", arguments={"q": n})],
                )
                for n in range(rounds)
            ]
            + [AIResponse(content="Done.", finish_reason="stop")]
        )
        tools = [AITool(name="lookup", description="d")]
        kit.register_channel(AIChannel("ai", provider=provider, tool_handler=lookup, tools=tools))
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        note = await kit.send_event("r1", "log", TextContent(body="card"), visibility="log")
        posted: list[RoomEvent] = []
        updated: list[RoomEvent | None] = []

        @kit.hook(trigger, execution=HookExecution.ASYNC, name="audit")
        async def audit(event: Any, ctx: RoomContext) -> None:
            updated.append(await kit.update_event("r1", note.id, metadata={"n": len(updated)}))
            posted.append(
                await kit.send_event("r1", "log", TextContent(body="called"), visibility="log")
            )

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(_observed(kit), 5)
            blocked = [e.blocked_by for e in sorted(posted, key=lambda e: e.index)]
            capped = blocked.index("reentry_loop_cap")
            assert blocked == [None] * capped + ["reentry_loop_cap"] * (rounds - capped)
            assert (0 < capped < rounds, len([u for u in updated if u])) == (True, rounds)
        finally:
            await kit.close()

    @pytest.mark.parametrize("when", ["during the turn", "after the turn"])
    async def test_an_observer_of_a_tool_call_its_agent_answers_ends_at_the_depth_limit(
        self, when: str
    ) -> None:
        """RMK-978: a loop of observer post, the agent's answer, its tool
        call, the next post. The post is at least as deep as the turn was
        when the call was made, once the turn is over too, so each round is
        one deeper and the loop ends at ``max_chain_depth``."""
        kit = await _room(RoomKit(max_chain_depth=3))
        posted: list[RoomEvent] = []
        rounds = 4 * BUDGET

        async def lookup(name: str, arguments: dict[str, Any]) -> str:
            return f"found {len(posted)}"

        provider = MockAIProvider(
            ai_responses=[
                response
                for n in range(rounds)
                for response in (
                    AIResponse(
                        content="",
                        finish_reason="tool_calls",
                        tool_calls=[AIToolCall(id=f"t{n}", name="lookup", arguments={})],
                    ),
                    AIResponse(content=f"Answer {n}.", finish_reason="stop"),
                )
            ]
        )
        tools = [AITool(name="lookup", description="d")]
        kit.register_channel(AIChannel("ai", provider=provider, tool_handler=lookup, tools=tools))
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
        async def audit(event: Any, ctx: RoomContext) -> None:
            if when == "after the turn":
                # Once the turn's delivery has run, as a slow progress card.
                await _settled(kit)
            if len(posted) < rounds:
                posted.append(await kit.send_event("r1", "sms", TextContent(body="called")))

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(_observed(kit), 5)
            assert [e.chain_depth for e in sorted(posted, key=lambda e: e.index)] == [1, 2]
        finally:
            await kit.close()

    @pytest.mark.parametrize("tells", ["a post", "an injection"])
    @pytest.mark.parametrize(
        "trigger",
        [
            HookTrigger.ON_TOOL_CALL,
            HookTrigger.BEFORE_TOOL_USE,
            HookTrigger.ON_REALTIME_DELEGATION,
        ],
    )
    @pytest.mark.parametrize("door", ["realtime", "conference"])
    async def test_an_observer_of_a_realtime_tool_call_its_model_answers_ends_at_the_depth_limit(
        self, door: str, trigger: HookTrigger, tells: str
    ) -> None:
        """RMK-983: a loop of observer telling the model, its answer, its
        tool call, the observer telling it again, through a post into the
        room or straight through ``inject_text()``. What the observer sends
        is at least as deep as the answer that issued the call (RFC §8.3),
        and a text past the limit enters the model's session silently, so it
        answers nothing and the loop ends at ``max_chain_depth``: three
        calls, the third one's text silent. So does a model's delegation,
        whose hooks ran at depth 0 (RMK-998)."""
        model = _AnsweringModel(
            rounds=4 * BUDGET, delegates=trigger is HookTrigger.ON_REALTIME_DELEGATION
        )
        kit, room_id, poster, channel, session = await _MODEL_HOSTS[door](model)
        posted: list[RoomEvent] = []

        @kit.hook(trigger, execution=HookExecution.ASYNC, name="audit")
        async def audit(event: Any, ctx: RoomContext) -> None:
            if tells == "a post":
                posted.append(await kit.send_event(room_id, poster, TextContent(body="called")))
            else:
                await channel.inject_text(session, "called")

        try:
            model.answer(session)
            await asyncio.wait_for(_until_the_model_rests(kit, model, room_id), 5)
            assert (len(model.answers), model.silences) == (3, [False, False, True])
            depths = [e.chain_depth for e in sorted(posted, key=lambda e: e.index)]
            assert depths == ([1, 2, 3] if tells == "a post" else [])
        finally:
            await kit.close()

    @pytest.mark.parametrize("tells", ["a post", "an injection"])
    @pytest.mark.parametrize(
        ("trigger", "execution"),
        [
            (HookTrigger.ON_TRANSCRIPTION, HookExecution.SYNC),
            (HookTrigger.ON_TRANSCRIPTION, HookExecution.ASYNC),
            (HookTrigger.ON_PARTIAL_TRANSCRIPTION, HookExecution.ASYNC),
        ],
    )
    async def test_a_hook_on_the_model_s_words_that_tells_it_again_ends_at_the_depth_limit(
        self, trigger: HookTrigger, execution: HookExecution, tells: str
    ) -> None:
        """RMK-983: a hook on what the speech-to-speech model says (its
        transcription, final or partial) tells it something each time, and
        the model answers that. The hook runs at the depth of the answer it
        hears and sends at least that deep, an ASYNC one once the words'
        processing is over too, so the third text is past the limit and asks
        the model nothing."""
        model = _TalkingModel(rounds=12, partials=trigger is not HookTrigger.ON_TRANSCRIPTION)
        kit = await _room(RoomKit(max_chain_depth=4))
        channel = RealtimeVoiceChannel("rt", provider=model, transport=MockRealtimeTransport())
        kit.register_channel(channel)
        await kit.attach_channel("r1", "rt")
        session = await channel.start_session("r1", "u", object())

        @kit.hook(trigger, execution=execution, name="echo")
        async def echo(event: Any, ctx: Any) -> HookResult | None:
            if event.role == "assistant":
                if trigger is HookTrigger.ON_TRANSCRIPTION and execution is HookExecution.ASYNC:
                    # An observer, not awaited: a slow one acts once the
                    # words were processed (a partial's hooks are awaited).
                    async with channel._transcription_order_locks[session.id]:  # noqa: SLF001
                        pass
                if tells == "a post":
                    await kit.send_event("r1", "sms", TextContent(body="heard you"))
                else:
                    await channel.inject_text(session, "heard you")
            return HookResult.allow() if execution is HookExecution.SYNC else None

        try:
            model.answer(session)
            await until(lambda: (model.silences or [False])[-1] or len(model.answers) >= 12)
            await asyncio.gather(*model.answers)
            await _observed(kit)
            assert (len(model.answers), model.silences) == (3, [False, False, True])
        finally:
            await kit.close()

    @pytest.mark.parametrize("brings", ["inject_text()", "deliver()"])
    async def test_a_hook_on_an_injection_injects_at_its_body_s_depth(self, brings: str) -> None:
        """RMK-983: an observer's body two deep brings a note to the
        speech-to-speech model, straight or through ``kit.deliver()``, and an
        ``ON_REALTIME_TEXT_INJECTED`` hook answers the note by injecting a
        reminder: the reminder is as deep as that body, three, so past the
        limit it enters silently. A delivery's injection took its own depth
        and did not lend that to the hooks it fired."""
        model = _SilenceRecorder()
        kit = await _room(RoomKit(max_chain_depth=4))
        channel = RealtimeVoiceChannel("rt", provider=model, transport=MockRealtimeTransport())
        kit.register_channel(channel)
        await kit.attach_channel("r1", "rt")
        session = await channel.start_session("r1", "u", object())
        reminded = asyncio.Event()

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="bring")
        async def bring(event: RoomEvent, ctx: RoomContext) -> None:
            if getattr(event.content, "body", None) != "deep":
                return
            if brings == "deliver()":
                await kit.deliver("r1", "note", channel_id="rt", instruction=False)
            else:
                await channel.inject_text(session, "note", chain_depth=3)

        @kit.hook(HookTrigger.ON_REALTIME_TEXT_INJECTED, execution=HookExecution.ASYNC)
        async def remind(event: Any, ctx: Any) -> None:
            if event.content.body.endswith("note"):
                await channel.inject_text(session, "reminder")
                reminded.set()

        try:
            await kit.send_event(
                "r1", "sms", TextContent(body="deep"), chain_depth=2, addressed_to=[]
            )
            await asyncio.wait_for(reminded.wait(), 5)
            assert model.silent["reminder"] is True
        finally:
            await kit.close()

    @pytest.mark.parametrize(
        ("trigger", "execution", "speaker"),
        [
            (trigger, execution, speaker)
            for speaker in ("a voice channel", "a streamed reply", "a conference")
            for trigger, execution in (
                (HookTrigger.AFTER_TTS, HookExecution.ASYNC),
                (HookTrigger.BEFORE_TTS, HookExecution.SYNC),
                (HookTrigger.BEFORE_TTS, HookExecution.ASYNC),
            )
            # A SYNC BEFORE_TTS hook that awaits a post into its room waits
            # on itself on a streamed reply: RMK-1006.
            if (speaker, execution) != ("a streamed reply", HookExecution.SYNC)
        ],
    )
    async def test_a_hook_on_what_a_voice_agent_says_that_tells_it_again_ends_at_the_depth_limit(
        self, trigger: HookTrigger, execution: HookExecution, speaker: str
    ) -> None:
        """RMK-998: a hook on what an STT, LLM and TTS agent says (before or
        after its synthesis) posts at each of its answers, and the agent
        answers that. The hooks run at the depth of the answer they speak,
        on a voice channel, a streamed reply and a conference's voice lane,
        so the third post is past the limit and asks the agent nothing.
        They ran at depth 0, and only the reentry cap ended the loop."""
        kit = await _SPEAKERS[speaker]()
        posted: list[RoomEvent] = []
        told: set[str] = set()
        ended = _ends_at_the_depth_limit(kit)

        @kit.hook(trigger, execution=execution, name="echo")
        async def echo(text: Any, ctx: Any) -> HookResult | None:
            if len(posted) >= 2 * BUDGET:
                ended.set()
            elif isinstance(text, str) and text.startswith("Answer") and text not in told:
                told.add(text)  # once per answer: a streamed one's hooks run per sentence too
                posted.append(await kit.send_event("r1", "sms", TextContent(body="Heard you.")))
            return HookResult.allow() if execution is HookExecution.SYNC else None

        try:
            await kit.send_event("r1", "sms", TextContent(body="Hello."))
            await asyncio.wait_for(ended.wait(), 5)
            await asyncio.wait_for(_observed(kit), 5)
            assert sorted(e.chain_depth for e in posted) == [1, 2, 3]
        finally:
            await kit.close()

    async def test_a_task_one_observer_of_the_model_s_words_starts_calls_as_work_of_its_own(
        self,
    ) -> None:
        """RMK-1004: the floor of an observer of the model's words is its own
        body's, not its siblings': a task one observer starts posts at depth
        0 once that observer returned, though another observer of the same
        words still runs (RFC §9.5.1). It posted at the answer's depth, 1."""
        model = _TalkingModel(rounds=1, partials=False)
        kit = await _room(RoomKit(max_chain_depth=10))
        channel = RealtimeVoiceChannel("rt", provider=model, transport=MockRealtimeTransport())
        kit.register_channel(channel)
        await kit.attach_channel("r1", "rt")
        session = await channel.start_session("r1", "u", object())
        returned, sent = asyncio.Event(), asyncio.Event()
        posted: list[RoomEvent] = []
        tasks: set[asyncio.Task[None]] = set()

        async def later() -> None:
            await returned.wait()
            posted.append(await kit.send_event("r1", "sms", TextContent(body="later")))
            sent.set()

        @kit.hook(HookTrigger.ON_TRANSCRIPTION, execution=HookExecution.ASYNC, name="starts")
        async def starts(event: Any, ctx: Any) -> None:
            if event.role == "assistant":
                task = asyncio.create_task(later())
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                returned.set()  # its body returns right after

        @kit.hook(HookTrigger.ON_TRANSCRIPTION, execution=HookExecution.ASYNC, name="slow")
        async def slow(event: Any, ctx: Any) -> None:
            if event.role == "assistant":
                await asyncio.wait_for(sent.wait(), 5)

        try:
            model.answer(session)
            await asyncio.wait_for(sent.wait(), 5)
            await _observed(kit)
            assert [e.chain_depth for e in posted] == [0]
        finally:
            await kit.close()

    @pytest.mark.parametrize(
        ("trigger", "posts"),
        [
            (HookTrigger.BEFORE_AI_GENERATION, "once the turn is over"),
            (HookTrigger.ON_AI_RESPONSE, "while the turn awaits it"),
        ],
    )
    async def test_a_hook_telling_the_agent_again_from_its_body_ends_at_the_depth_limit(
        self, trigger: HookTrigger, posts: str
    ) -> None:
        """RMK-1004: an ASYNC hook on a turn's trigger that posts from its
        body at each turn sends at least as deep as the turn was when the
        trigger fired, once the turn is over too: each round is one deeper
        and the loop ends at ``max_chain_depth`` (RFC §8.3, §9.5.1). A
        detached observer posting once the turn was over posted at depth 1
        every time, until the reentry cap."""
        kit = await _room(RoomKit(max_chain_depth=5))
        kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["ok"])))
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        posted: list[RoomEvent] = []
        ended = _ends_at_the_depth_limit(kit)

        @kit.hook(trigger, execution=HookExecution.ASYNC, name="remind")
        async def remind(event: Any, ctx: RoomContext) -> None:
            if len(posted) >= 2 * BUDGET:
                ended.set()
                return
            if posts == "once the turn is over":
                await _settled(kit)
            posted.append(await kit.send_event("r1", "sms", TextContent(body="Cite sources.")))

        try:
            await kit.send_event("r1", "sms", TextContent(body="What is RoomKit?"))
            await asyncio.wait_for(ended.wait(), 5)
            await asyncio.wait_for(_observed(kit), 5)
            assert sorted(e.chain_depth for e in posted) == [1, 2, 3, 4]
        finally:
            await kit.close()

    @pytest.mark.parametrize("report", ["the session's end", "the backend's provider"])
    async def test_an_observer_of_a_realtime_call_reported_outside_its_run_posts_at_its_depth(
        self, report: str
    ) -> None:
        """RMK-983: a realtime call reported outside its run, the session's
        end cutting it or the reasoning backend's own provider serving it,
        fires its observers at the depth of the answer that issued it: 3,
        the model having heard an event 2 deep (RFC §8.3, §9.5.1)."""
        kit = await _room(RoomKit(max_chain_depth=10))
        channel, provider, entered = _realtime_reported_outside(report)
        kit.register_channel(channel)
        await kit.attach_channel("r1", "rt")
        posted: list[RoomEvent] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
        async def audit(event: Any, ctx: RoomContext) -> None:
            posted.append(await kit.send_event("r1", "sms", TextContent(body="reported")))

        try:
            session = await channel.start_session("r1", "u", object())
            channel._session_answer_depth(session.id).injected(2)  # noqa: SLF001
            if report == "the session's end":
                await provider.simulate_tool_call(session, "c1", "lookup", {})
                await asyncio.wait_for(entered.wait(), 2)
                await channel.end_session(session)
            else:
                await provider.simulate_delegation(session, "d1", "integrator")
            await until(lambda: bool(posted))
            assert [e.chain_depth for e in posted] == [3]
        finally:
            await kit.close()

    async def test_a_hook_opening_a_room_after_every_event_stops_at_the_cap(self) -> None:
        """A message the hook sends that opens a room is the cascade's work
        too, charged before the room is created: a hook that opens a room
        after every event, each room's message firing it again, stops at the
        cap."""
        kit = RoomKit(max_chain_depth=1, inbound_router=_NoRoom())
        calls: list[None] = []
        answers: list[InboundResult] = []

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="open")
        async def open_room(event: RoomEvent, ctx: RoomContext) -> None:
            if event.type is EventType.MESSAGE and len(calls) <= 2 * BUDGET:
                calls.append(None)
                answers.append(await kit.process_inbound(_message("again")))

        await _room(kit)
        try:
            await asyncio.wait_for(kit.process_inbound(_message(), room_id="r1"), 5)
            await asyncio.wait_for(_observed(kit), 5)
            # No call waits for the hook of the room it opens (RFC §9.3), so
            # the calls answer in no set order: one past the cap.
            reasons = [a.reason for a in answers]
            assert (reasons.count(None), reasons.count("reentry_loop_cap")) == (BUDGET, 1)
            assert len(await kit.store.list_rooms()) == 1 + BUDGET
        finally:
            await kit.close()

    @pytest.mark.parametrize("room", ["plain", "discussion"])
    async def test_a_judge_posting_back_after_each_answer_stops_at_the_depth_limit(
        self, room: str
    ) -> None:
        """A hook that awaits each answer and posts back to the agent: each
        post is one deeper than the answer that fired the hook, so the round
        trips end at the depth limit, about half of it in answers (3 at the
        limit of 6), in a plain room as in a discussion, whose turns each
        have a budget of their own (RFC §8.3, §19.7.5 rule 11)."""
        kit = RoomKit(max_chain_depth=6)
        provider = MockAIProvider(responses=["Answer."])
        await _agent_room(kit, room, provider)
        posted: list[RoomEvent] = []
        ended = _ends_at_the_depth_limit(kit)

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="judge")
        async def judge(event: RoomEvent, ctx: RoomContext) -> None:
            if event.source.channel_id != "a" or event.blocked_by is not None:
                return
            if len(posted) >= 4 * BUDGET:
                ended.set()
                return
            posted.append(await kit.post("r1", "@a again", addressed_to=["a"]))

        try:
            await asyncio.wait_for(kit.process_inbound(_message("@a hello")), 5)
            await asyncio.wait_for(ended.wait(), 5)
            await asyncio.wait_for(_settled(kit), 5)
            assert (len(provider.calls), [e.chain_depth for e in posted]) == (3, [2, 4, 6])
        finally:
            await kit.close()

    async def test_a_hook_delivering_to_the_agent_through_a_queue_stops_at_the_depth_limit(
        self,
    ) -> None:
        """``deliver()`` from a hook's body continues the hook's chain even
        when a worker runs it later, outside the body: the queued item
        carries the depth, and the round trips end at the depth limit of 6,
        3 answers."""
        backend = InMemoryDeliveryBackend()
        kit = RoomKit(max_chain_depth=6, delivery_backend=backend)
        await backend.start(kit)
        provider = MockAIProvider(responses=["Answer."])
        await _agent_room(kit, "plain", provider)
        delivered: list[None] = []
        ended = _ends_at_the_depth_limit(kit)

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="again")
        async def again(event: RoomEvent, ctx: RoomContext) -> None:
            if event.source.channel_id != "a" or event.blocked_by is not None:
                return
            if len(delivered) >= 4 * BUDGET:
                ended.set()
                return
            delivered.append(None)
            await kit.deliver("r1", "again", channel_id="sms", addressed_to=["a"])

        try:
            await asyncio.wait_for(kit.process_inbound(_message("@a hello")), 5)
            await asyncio.wait_for(ended.wait(), 5)
            assert (len(provider.calls), len(delivered)) == (3, 3)
        finally:
            await kit.close()
            await backend.close()

    @pytest.mark.parametrize("room", ["plain", "discussion"])
    async def test_a_hook_regenerating_after_each_answer_stops_at_the_depth_limit(
        self, room: str
    ) -> None:
        """A hook that deletes each answer and regenerates it: each
        regeneration is one deeper than the answer that fired the hook, so
        the loop ends at the depth limit of 6 after 3 answers, in a plain
        room as in a discussion, whose regenerated turns each have a budget
        of their own (RFC §8.3, §10.3, §19.7.5 rule 11)."""
        kit = RoomKit(max_chain_depth=6)
        provider = MockAIProvider(responses=["Answer."])
        await _agent_room(kit, room, provider)
        regenerated: list[None] = []
        ended = _ends_at_the_depth_limit(kit)

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="redo")
        async def redo(event: RoomEvent, ctx: RoomContext) -> None:
            if event.source.channel_id != "a" or event.blocked_by is not None:
                return
            if len(regenerated) >= 4 * BUDGET:
                ended.set()
                return
            regenerated.append(None)
            await kit.delete_event("r1", event.id)
            await kit.regenerate_response("r1")

        try:
            await asyncio.wait_for(kit.process_inbound(_message("@a hello")), 5)
            await asyncio.wait_for(ended.wait(), 5)
            await asyncio.wait_for(_settled(kit), 5)
            assert (len(provider.calls), len(regenerated)) == (3, 3)
        finally:
            await kit.close()

    @pytest.mark.parametrize("change", ["edit", "delete"])
    async def test_a_mutation_hook_changing_again_stops_at_the_cap(self, change: str) -> None:
        """A change's mutation trigger is its cascade's work too: a hook that
        edits the event again each time it is edited (``ON_EVENT_UPDATED``),
        or sends a message and deletes it each time one is deleted
        (``ON_EVENT_DELETED``), stops at the cap of its hook calls."""
        kit = await _room(RoomKit(max_chain_depth=1))
        trigger, again = _CHANGED_AGAIN[change]
        sent: list[RoomEvent] = []

        @kit.hook(trigger, execution=HookExecution.ASYNC, name="again")
        async def change_again(event: RoomEvent, ctx: RoomContext) -> None:
            if len(sent) <= 2 * BUDGET:
                sent.extend(await again(kit, event, len(sent)))

        try:
            first = await kit.send_event("r1", "sms", TextContent(body="v"), participant_id="u")
            await asyncio.wait_for(_send_change(kit, _FIRST_CHANGE[change](first)), 5)
            await asyncio.wait_for(_settled(kit), 5)
            assert [e.blocked_by for e in sent] == [None] * BUDGET + ["reentry_loop_cap"]
        finally:
            await kit.close()

    async def test_a_hook_posting_once_per_message_posts_for_every_message(self) -> None:
        """A notice, one post for each message, draws one hook call of that
        message's cascade: more messages than a cascade holds hook calls
        each get theirs."""
        kit = await _room(RoomKit(max_chain_depth=1))
        posted: list[RoomEvent] = []

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="receipt")
        async def receipt(event: RoomEvent, ctx: RoomContext) -> None:
            if event.source.channel_id == "sms":
                posted.append(await kit.post("r1", "Received."))

        try:
            for n in range(BUDGET + 1):
                await asyncio.wait_for(kit.process_inbound(_message(f"message {n}")), 5)
            await asyncio.wait_for(_settled(kit), 5)
            assert [e.blocked_by for e in posted] == [None] * (BUDGET + 1)
        finally:
            await kit.close()

    @pytest.mark.parametrize("ends", ["returns", "raises", "times out"])
    async def test_a_task_the_hook_starts_calls_as_work_of_its_own(self, ends: str) -> None:
        """Once the hook's body is over, whether it returned, raised or ran
        past its timeout, a task it started sends at depth 0 with a budget
        of its own for each call, as a call from outside the room's work
        (RFC §9.5.1, as for a turn's task)."""
        kit = await _room(RoomKit(max_chain_depth=1))
        sends = 2 * BUDGET + 1
        later = _Later(kit, "r1", "sms", sends)
        never = asyncio.Event()

        @kit.hook(
            HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="start", timeout=0.01
        )
        async def start(event: RoomEvent, ctx: RoomContext) -> None:
            if _body(event) != "hi":
                return
            later.start()
            if ends == "raises":
                raise RuntimeError("the hook's body fails")
            if ends == "times out":
                await never.wait()

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(_observed(kit), 5)
            later.over.set()
            sent = await asyncio.wait_for(later.sent, 5)
            assert [(e.chain_depth, e.blocked_by) for e in sent] == [(0, None)] * sends
        finally:
            await kit.close()

    @pytest.mark.parametrize("execution", [HookExecution.ASYNC, HookExecution.SYNC])
    async def test_a_relaunch_a_turn_s_observer_starts_begins_a_new_chain(
        self, execution: HookExecution
    ) -> None:
        """RMK-1004, the RMK-930 decision for work that outlives what began
        it: an ``ON_AI_RESPONSE`` hook that relaunches the agent from a task
        it starts, as a quality gate does, relaunches it at depth 0 each time
        and every relaunch is answered, though the turn still runs when the
        relaunch is sent, waiting for a slower observer of the same answer
        (an audit write). With the turn's floor, each relaunch was one deeper
        and the fourth was the last answered. So for a hook registered SYNC,
        which an ASYNC trigger runs as its ASYNC hooks (RFC §8.3, §9.2, §9.5.1)."""
        kit = await _room(RoomKit(max_chain_depth=5))
        kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["ok"])))
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        relaunches: list[RoomEvent] = []
        launched: list[bool] = []
        decided: list[RoomEvent] = []
        one_decided, over = asyncio.Event(), asyncio.Event()
        tasks: set[asyncio.Task[None]] = set()

        async def relaunch() -> None:
            event = await kit.send_event("r1", "sms", TextContent(body="Gate failed: retry."))
            relaunches.append(event)

        @kit.on("chain_depth_exceeded")
        async def limited(event: object) -> None:
            over.set()

        @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=execution, name="gate")
        async def gate(event: Any, ctx: RoomContext) -> None:
            if len(launched) == 5:
                over.set()
                return
            launched.append(True)
            task = asyncio.create_task(relaunch())
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC, name="audit")
        async def audit(event: Any, ctx: RoomContext) -> None:
            if len(launched) > len(decided):  # the turn runs until it returns
                await asyncio.wait_for(one_decided.wait(), 5)
            one_decided.clear()

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="decided")
        async def seen(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) == "Gate failed: retry.":
                decided.append(event)  # the relaunch's depth is set
                one_decided.set()
            return HookResult.allow()

        try:
            await kit.send_event("r1", "sms", TextContent(body="Write the report."))
            await asyncio.wait_for(over.wait(), 5)
            await asyncio.wait_for(asyncio.gather(*tasks), 5)
            await asyncio.wait_for(_observed(kit), 5)
            assert [e.chain_depth for e in relaunches] == [0] * 5
        finally:
            await kit.close()

    async def test_a_task_the_hook_starts_continues_the_chain_it_names(self) -> None:
        """A task the hook starts begins at depth 0 unless it names the chain
        it continues: posting one deeper than the answer that fired the
        hook, the round trips end at ``max_chain_depth`` (4 here), the agent
        not asked about the second post (RFC §8.3, §10.5)."""
        kit = await _room(RoomKit(max_chain_depth=4))
        kit.register_channel(AILikeChannel("ai", response="Answer."))
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        ended = asyncio.Event()
        failures: list[Exception] = []
        tasks: set[asyncio.Task[None]] = set()

        async def follow_up(answer: RoomEvent) -> None:
            try:
                await kit.post(
                    "r1", "Try again.", addressed_to=["ai"], chain_depth=answer.chain_depth + 1
                )
            except Exception as error:
                failures.append(error)
                ended.set()

        @kit.on("chain_depth_exceeded")
        async def limited(event: object) -> None:
            ended.set()

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="judge")
        async def judge(event: RoomEvent, ctx: RoomContext) -> None:
            if event.source.channel_id == "ai" and event.blocked_by is None:
                task = asyncio.create_task(follow_up(event))
                tasks.add(task)
                task.add_done_callback(tasks.discard)

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(ended.wait(), 5)
            await asyncio.wait_for(_settled(kit), 5)
            assert failures == []
            assert [
                (e.source.channel_id, e.chain_depth, e.blocked_by) for e in await _messages(kit)
            ] == [
                ("sms", 0, None),
                ("ai", 1, None),
                ("host", 2, None),
                ("ai", 3, None),
                ("host", 4, None),
                ("ai", 5, "event_chain_depth_limit"),
            ]
        finally:
            await kit.close()


async def _send_change(kit: RoomKit, change: EditContent | DeleteContent) -> RoomEvent:
    """Its sender's edit or delete of an event in room ``r1``."""
    event_type = EventType.EDIT if isinstance(change, EditContent) else EventType.DELETE
    return await kit.send_event("r1", "sms", change, event_type, participant_id="u")


def _edit(event: RoomEvent, n: int = 0) -> EditContent:
    return EditContent(target_event_id=event.id, new_content=TextContent(body=f"v{n}"))


async def _edit_again(kit: RoomKit, event: RoomEvent, n: int) -> list[RoomEvent]:
    return [await _send_change(kit, _edit(event, n))]


async def _delete_another(kit: RoomKit, event: RoomEvent, n: int) -> list[RoomEvent]:
    sent = await kit.send_event(
        "r1", "sms", TextContent(body=f"m{n}"), participant_id="u", addressed_to=[]
    )
    if sent.blocked_by is not None:
        return [sent]
    return [sent, await _send_change(kit, DeleteContent(target_event_id=sent.id))]


type _Again = Callable[[RoomKit, RoomEvent, int], Awaitable[list[RoomEvent]]]

_CHANGED_AGAIN: dict[str, tuple[HookTrigger, _Again]] = {
    "edit": (HookTrigger.ON_EVENT_UPDATED, _edit_again),
    "delete": (HookTrigger.ON_EVENT_DELETED, _delete_another),
}
_FIRST_CHANGE: dict[str, Callable[[RoomEvent], EditContent | DeleteContent]] = {
    "edit": _edit,
    "delete": lambda event: DeleteContent(target_event_id=event.id),
}


class _Streaming(Channel):
    """An agent whose turn on a person's message streams *rows* segments,
    each committed as a row of its own."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, channel_id: str, rows: int) -> None:
        super().__init__(channel_id)
        self._rows = rows

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if event.source.channel_type is not ChannelType.SMS:
            return ChannelOutput.empty()
        return ChannelOutput(responded=True, response_stream=self._segments())

    async def _segments(self) -> Any:
        for n in range(self._rows):
            yield f"part {n}. "
            yield SegmentBreakMarker()

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class TestTheBudgetOfAHooksCalls:
    """A cascade counts the calls of its ASYNC hooks' bodies apart from its
    own passes, on a counter of the same size (``10 × max_chain_depth``): a
    hook's loop stops at its cap, its notices cost the agents no answer, and
    the agents' answers block no notice below it (RFC §8.3, §9.5.1)."""

    async def test_a_notice_per_answer_costs_the_agents_no_answer(self) -> None:
        """Three agents answer one another to the depth limit of 5, 45
        answers, and a hook posts a notice after each: 45 passes and 45 hook
        calls, each under its cap of 50, so every answer and every notice
        lands. On one shared budget, 34 answers landed and 18 notices were
        blocked."""
        kit = await _room(RoomKit(max_chain_depth=5))
        for n in range(3):
            kit.register_channel(AILikeChannel(f"agent:{n}", response="Answer."))
            await kit.attach_channel("r1", f"agent:{n}", category=ChannelCategory.INTELLIGENCE)
        notices = await _notice_per_agent_row(kit)

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(_settled(kit), 5)
            answers = [e for e in await _messages(kit) if e.source.channel_id.startswith("agent:")]
            assert [
                e.blocked_by for e in answers if e.blocked_by != "event_chain_depth_limit"
            ] == [None] * 45
            assert [e.blocked_by for e in notices] == [None] * 45
        finally:
            await kit.close()

    async def test_a_notice_per_streamed_row_costs_its_answers_nothing(self) -> None:
        """An agent streams 12 rows, a second agent answers the person and
        each row, and a hook posts a notice after each agent row: 13 passes
        for the answers and 25 hook calls, each under its cap of 30, so
        every row, every answer and every notice lands (depth limit 3). On
        one shared budget, 38 draws past 30, notices were blocked."""
        kit = await _room(RoomKit(max_chain_depth=3))
        kit.register_channel(_Streaming("agent:stream", rows=12))
        kit.register_channel(AILikeChannel("agent:answers", response="Answer."))
        for agent in ("agent:stream", "agent:answers"):
            await kit.attach_channel("r1", agent, category=ChannelCategory.INTELLIGENCE)
        notices = await _notice_per_agent_row(kit)

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(_settled(kit), 5)
            rows = [
                (e.source.channel_id, e.blocked_by)
                for e in await _messages(kit)
                if e.source.channel_id.startswith("agent:")
                and e.blocked_by != "event_chain_depth_limit"
            ]
            assert (
                sorted(rows, key=str)
                == [("agent:answers", None)] * 13 + [("agent:stream", None)] * 12
            )
            assert [e.blocked_by for e in notices] == [None] * 25
        finally:
            await kit.close()

    async def test_a_notice_per_row_past_the_hook_cap_is_blocked(self) -> None:
        """One pass per event on the cascade's hook calls: a streamed turn of
        22 rows, a notice after each, lands all its rows and 20 notices, the
        hook cap of 20 (depth limit 2), the last two notices blocked."""
        kit = await _room(RoomKit(max_chain_depth=2))
        kit.register_channel(_Streaming("agent:stream", rows=22))
        await kit.attach_channel("r1", "agent:stream", category=ChannelCategory.INTELLIGENCE)
        notices = await _notice_per_agent_row(kit)

        try:
            await asyncio.wait_for(kit.process_inbound(_message()), 5)
            await asyncio.wait_for(_settled(kit), 5)
            rows = [e for e in await _messages(kit) if e.source.channel_id == "agent:stream"]
            assert [e.blocked_by for e in rows] == [None] * 22
            assert [e.blocked_by for e in notices] == [None] * 20 + ["reentry_loop_cap"] * 2
        finally:
            await kit.close()


async def _notice_per_agent_row(kit: RoomKit) -> list[RoomEvent]:
    """A hook that posts a notice, answered by nobody, after each agent row
    of room ``r1``; the notices it sent."""
    notices: list[RoomEvent] = []

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="notice")
    async def notice(event: RoomEvent, ctx: RoomContext) -> None:
        if event.source.channel_id.startswith("agent:") and event.blocked_by is None:
            notices.append(
                await kit.send_event("r1", "sms", TextContent(body="Noted."), addressed_to=[])
            )

    return notices


class TestADirectChangeFromAHook:
    """A direct ``update_event()``, ``delete_event()`` or
    ``regenerate_response()`` an ASYNC hook's body makes draws one hook call
    of the budget that body runs on; the call's own triggers run their
    bodies on the budget of the work it came from, one of its own from
    outside. Past the cap it raises ``ReentryLoopCapError``, nothing written
    (RFC §8.3, §9.5.1, §10.3)."""

    @pytest.mark.parametrize(
        ("loop", "max_chain_depth", "runs"),
        [
            pytest.param("update", 1, 11, id="ON_EVENT_UPDATED updates again"),
            pytest.param("delete", 1, 6, id="ON_EVENT_DELETED sends and deletes again"),
        ],
    )
    async def test_a_hook_changing_its_room_on_every_change_stops_at_the_cap(
        self, loop: str, max_chain_depth: int, runs: int
    ) -> None:
        """Started from outside the room's work, each loop runs on the
        budget of its first call (a cap of 10 hook calls), and its last run's
        call raises. They solicit no agent, so no depth limit stops them."""
        kit = await _room(RoomKit(max_chain_depth=max_chain_depth))
        refused: list[ReentryLoopCapError] = []
        ran = await _DIRECT_LOOPS[loop](kit, refused)

        try:
            await asyncio.wait_for(_settled(kit), 5)
            assert (len(ran), len(refused)) == (runs, 1)
        finally:
            await kit.close()

    @pytest.mark.parametrize("change", ["update", "delete", "regenerate"])
    async def test_a_direct_change_past_the_cap_writes_nothing(self, change: str) -> None:
        """A hook's body spends the 30 hook calls of its cascade (depth limit
        3) updating another event, then makes one more direct change of the
        person's message the agent answered: refused once, nothing of it
        written, the message neither changed nor deleted, and the agent not
        asked again, though a regeneration of that message would ask it."""
        kit = await _room(RoomKit(max_chain_depth=3))
        provider = MockAIProvider(responses=["Answer."])
        kit.register_channel(AIChannel("ai", provider=provider))
        await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
        spare = await kit.send_event("r1", "sms", TextContent(body="spare"), addressed_to=[])
        refused: list[ReentryLoopCapError] = []

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="bulk")
        async def bulk(event: RoomEvent, ctx: RoomContext) -> None:
            if _body(event) != "hi":
                return
            for n in range(30):
                await kit.update_event("r1", spare.id, metadata={"n": n})
            try:
                await _ONE_MORE[change](kit, event)
            except ReentryLoopCapError as error:
                refused.append(error)

        try:
            hi = await kit.send_event("r1", "sms", TextContent(body="hi"))
            await asyncio.wait_for(_settled(kit), 5)
            assert len(refused) == 1
            stored = await kit.store.get_event(hi.id)
            assert stored is not None and stored.metadata == hi.metadata
            assert len(provider.calls) == 1
        finally:
            await kit.close()

    @pytest.mark.parametrize("work", ["a delivery", "a locked hook"])
    async def test_work_outside_a_hook_body_draws_nothing(self, work: str) -> None:
        """A transport's delivery, or a SYNC hook under the lock, that makes
        60 direct updates, six times the cap of 10, never raises: only an
        ASYNC hook's body draws on the hook calls (RFC §10.3)."""
        kit = await _room(RoomKit(max_chain_depth=1))
        target = await kit.send_event("r1", "sms", TextContent(body="target"))
        failures: list[Exception] = []

        async def update_sixty_times() -> None:
            try:
                for n in range(60):
                    await kit.update_event("r1", target.id, metadata={"n": n})
            except Exception as error:
                failures.append(error)

        if work == "a delivery":
            kit.register_channel(_Calling("calling", update_sixty_times))
            await kit.attach_channel("r1", "calling")
        else:

            @kit.hook(HookTrigger.BEFORE_BROADCAST, name="updates")
            async def updates(event: RoomEvent, ctx: RoomContext) -> HookResult:
                if _body(event) == "go":
                    await update_sixty_times()
                return HookResult.allow()

        try:
            await asyncio.wait_for(kit.send_event("r1", "sms", TextContent(body="go")), 5)
            await asyncio.wait_for(_settled(kit), 5)
            stored = await kit.store.get_event(target.id)
            assert (failures, stored.metadata if stored else None) == ([], {"n": 59})
        finally:
            await kit.close()

    async def test_a_cap_error_a_body_lets_escape_is_logged_as_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The hook engine logs a ``ReentryLoopCapError`` a body does not
        catch as a warning, without a traceback: the loop it stops is the
        hook's, not a failure of the framework."""
        kit = await _room(RoomKit(max_chain_depth=1))

        @kit.hook(HookTrigger.ON_EVENT_UPDATED, execution=HookExecution.ASYNC, name="again")
        async def again(event: RoomEvent, ctx: RoomContext) -> None:
            await kit.update_event("r1", event.id, metadata={"n": event.metadata["n"] + 1})

        try:
            first = await kit.send_event("r1", "sms", TextContent(body="v"))
            with caplog.at_level(logging.WARNING, logger="roomkit"):
                await asyncio.wait_for(kit.update_event("r1", first.id, metadata={"n": 0}), 5)
            [record] = [r for r in caplog.records if "again" in r.getMessage()]
            assert (record.levelno, record.exc_info) == (logging.WARNING, None)
        finally:
            await kit.close()


async def _updating(kit: RoomKit, refused: list[ReentryLoopCapError]) -> list[None]:
    """``ON_EVENT_UPDATED`` updates its event again; the first update is the test's."""
    ran: list[None] = []

    @kit.hook(HookTrigger.ON_EVENT_UPDATED, execution=HookExecution.ASYNC, name="again")
    async def again(event: RoomEvent, ctx: RoomContext) -> None:
        ran.append(None)
        if len(ran) > 4 * BUDGET:
            return
        try:
            await kit.update_event("r1", event.id, metadata={"n": len(ran)})
        except ReentryLoopCapError as error:
            refused.append(error)

    first = await kit.send_event("r1", "sms", TextContent(body="v"))
    await kit.update_event("r1", first.id, metadata={"n": 0})
    return ran


async def _deleting(kit: RoomKit, refused: list[ReentryLoopCapError]) -> list[None]:
    """``ON_EVENT_DELETED`` sends a message and deletes it; the first delete is the test's."""
    ran: list[None] = []

    @kit.hook(HookTrigger.ON_EVENT_DELETED, execution=HookExecution.ASYNC, name="again")
    async def again(event: RoomEvent, ctx: RoomContext) -> None:
        ran.append(None)
        if len(ran) > 4 * BUDGET:
            return
        sent = await kit.send_event("r1", "sms", TextContent(body="x"), addressed_to=[])
        if sent.blocked_by is not None:
            refused.append(ReentryLoopCapError("the send was capped"))
            return
        try:
            await kit.delete_event("r1", sent.id)
        except ReentryLoopCapError as error:
            refused.append(error)

    first = await kit.send_event("r1", "sms", TextContent(body="v"))
    await kit.delete_event("r1", first.id)
    return ran


async def _update_hi(kit: RoomKit, event: RoomEvent) -> None:
    await kit.update_event("r1", event.id, metadata={"changed": True})


async def _delete_hi(kit: RoomKit, event: RoomEvent) -> None:
    await kit.delete_event("r1", event.id)


async def _regenerate(kit: RoomKit, event: RoomEvent) -> None:
    await kit.regenerate_response("r1")


_ONE_MORE: dict[str, Callable[[RoomKit, RoomEvent], Awaitable[None]]] = {
    "update": _update_hi,
    "delete": _delete_hi,
    "regenerate": _regenerate,
}


class _Calling(SimpleChannel):
    """A transport whose delivery of ``go`` runs *work*."""

    def __init__(self, channel_id: str, work: Callable[[], Awaitable[None]]) -> None:
        super().__init__(channel_id)
        self._work = work

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if _body(event) == "go":
            await self._work()
        return ChannelOutput.empty()


async def _agent_room(kit: RoomKit, room: str, provider: MockAIProvider) -> None:
    """Room ``r1``, its transport ``sms`` and its agent ``a``: bound to a
    plain room, or the one agent of a discussion."""
    agent = AIChannel("a", provider=provider)
    kit.register_channel(SimpleChannel("sms"))
    if room == "discussion":
        await kit.create_room(room_id="r1", orchestration=Discussion([agent]))
    else:
        await kit.create_room(room_id="r1")
        kit.register_channel(agent)
        await kit.attach_channel("r1", "a", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("r1", "sms")


def _ends_at_the_depth_limit(kit: RoomKit) -> asyncio.Event:
    """An event set when the depth limit stops an agent in the kit's rooms."""
    ended = asyncio.Event()

    @kit.on("chain_depth_exceeded")
    async def limited(event: object) -> None:
        ended.set()

    return ended


_DIRECT_LOOPS: dict[str, Callable[[RoomKit, list[ReentryLoopCapError]], Awaitable[list[None]]]] = {
    "update": _updating,
    "delete": _deleting,
}


class TestARoutedMessageBackIntoItsRoom:
    """A message routed from under the lock, from an off-lock hook's body
    or from a parse lands where its caller's room is known only after
    routing: into that room, it fails rather than commit ahead of the event
    its caller works for (RFC §9.5.1, §10.1 step 2)."""

    @pytest.mark.parametrize("needs_lock", [True, False], ids=["locked hook", "off-lock hook"])
    async def test_from_a_hook_it_fails(self, needs_lock: bool) -> None:
        kit = await _room(RoomKit())
        answers: list[object] = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="router", needs_lock=needs_lock)
        async def route_back(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) == "hi":
                try:
                    answers.append(await kit.process_inbound(_message("again")))
                except ReentrantCallError as refused:
                    answers.append(refused)
            return HookResult.allow()

        try:
            result = await asyncio.wait_for(kit.process_inbound(_message()), 5)
            assert not result.blocked
            assert [type(a) for a in answers] == [ReentrantCallError]
            assert [_body(e) for e in await _messages(kit)] == ["hi"]
        finally:
            await kit.close()

    async def test_from_a_parse_it_fails(self) -> None:
        kit = RoomKit()
        parser = _Parser("sms", kit, routed=True)
        await kit.create_room(room_id="r1")
        kit.register_channel(parser)
        await kit.attach_channel("r1", "sms")
        try:
            await asyncio.wait_for(kit.process_inbound(_message(), room_id="r1"), 5)
            assert [type(a) for a in parser.answers] == [ReentrantCallError]
            assert [_body(e) for e in await _messages(kit)] == ["hi"]
        finally:
            await kit.close()
