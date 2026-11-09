"""A caller that stops waiting (RFC §13.6).

Once its call is decided, a caller cancelled (a client gone, an outer timeout
of its own) leaves the call's writes whole: the event a commit in flight
wrote is delivered, what its hooks injected is committed, and the room's
lane goes on. A room step 2 opens is completed whatever becomes of its
caller.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from roomkit import FrameworkEvent, RoomKit
from roomkit.core.inbound_router import InboundRoomRouter
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.identity import Identity
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter
from roomkit.store.memory import InMemoryStore
from tests.test_framework import SimpleChannel


def _body(event: RoomEvent) -> str:
    return getattr(event.content, "body", "")


def _msg(body: str, sender: str = "u1") -> InboundMessage:
    return InboundMessage(channel_id="sms1", sender_id=sender, content=TextContent(body=body))


class _SlowAnswer(InMemoryStore):
    """A commit whose answer travels back: the row is written and the call
    has not returned yet, as over a database round trip."""

    def __init__(self) -> None:
        super().__init__()
        self.held: str | None = None
        self.written = asyncio.Event()
        self.answer = asyncio.Event()

    async def commit_event(self, room_id: str, event: RoomEvent, **kw: Any) -> RoomEvent:
        committed = await super().commit_event(room_id, event, **kw)
        # Held by its text, a record by its code, a BLOCKED one by its block.
        names = (_body(event), getattr(event.content, "code", None), event.blocked_by)
        if self.held is not None and self.held in names:
            self.held = None
            self.written.set()
            await self.answer.wait()
        return committed


async def _cancelled_while(caller: asyncio.Task[Any], reached: asyncio.Event) -> None:
    """Cancel *caller* once it *reached* the point under test, and see the
    cancellation reach it."""
    await asyncio.wait_for(reached.wait(), 1)
    caller.cancel()


class TestACommitInFlight:
    async def test_a_retry_after_the_caller_leaves_returns_the_committed_event(self) -> None:
        """A caller gone during the commit can retry without a second delivery (§13.4)."""
        store = _SlowAnswer()
        kit = RoomKit(store=store)
        ws = SimpleChannel("ws1")
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")
        before = await kit.get_room("r1")
        message = _msg("first").model_copy(update={"idempotency_key": "k"})
        store.held = "first"
        try:
            caller = asyncio.create_task(kit.process_inbound(message, room_id="r1"))
            await _cancelled_while(caller, store.written)
            store.answer.set()
            with pytest.raises(asyncio.CancelledError):
                await caller
            replay = await asyncio.wait_for(kit.process_inbound(message, room_id="r1"), 5)
            after = await kit.get_room("r1")
            assert replay.duplicate and not replay.blocked
            assert replay.event is not None and replay.event.index == before.event_count
            held = await store.get_event_by_idempotency_key("r1", "k", channel_id="sms1")
            assert held is not None and replay.event.id == held.id
            assert (after.event_count, after.latest_index) == (
                before.event_count + 1,
                before.event_count,
            )
            assert [_body(event) for event in ws.delivered] == ["first"]
        finally:
            store.answer.set()
            await kit.close()

    @pytest.mark.parametrize("action", ["allow", "block"])
    async def test_its_writes_are_whole_and_the_lane_goes_on(self, action: str) -> None:
        """The caller is cancelled while its event's commit (a DELIVERED event
        or a BLOCKED record) is in flight: the event is delivered, what its
        hook injected is committed and delivered, and the room's next message
        waits for no gap (RFC §13.6, §9.4, §10.1 step 12)."""
        store = _SlowAnswer()
        kit = RoomKit(store=store, delivery_gap_timeout=0.2)
        ws = SimpleChannel("ws1")
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="notify")
        async def notify(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if _body(event) != "first":
                return HookResult.allow()
            notice = RoomEvent(
                room_id="r1",
                source=EventSource(channel_id="sms1", channel_type="sms"),
                content=TextContent(body="notice"),
            )
            return HookResult(
                action=action,
                reason="held" if action == "block" else None,
                injected_events=[InjectedEvent(event=notice, target_channel_ids=["ws1"])],
            )

        store.held = "first"
        try:
            caller = asyncio.create_task(kit.process_inbound(_msg("first"), room_id="r1"))
            await _cancelled_while(caller, store.written)
            store.answer.set()
            with pytest.raises(asyncio.CancelledError):
                await caller
            await asyncio.wait_for(kit.process_inbound(_msg("second"), room_id="r1"), 1)

            first = ["first"] if action == "allow" else []
            assert [_body(e) for e in ws.delivered] == [*first, "notice", "second"]
        finally:
            await kit.close()


class _FixedRouter(InboundRoomRouter):
    """Routes every message into room r1."""

    async def route(
        self,
        channel_id: str,
        channel_type: ChannelType,
        participant_id: str | None = None,
        channel_data: dict[str, Any] | None = None,
        *,
        organization_id: str | None = None,
        identity: Identity | None = None,
    ) -> str | None:
        return "r1"


async def _attach_in_a_named_room(kit: RoomKit) -> None:
    await kit.process_inbound(_msg("first"), room_id="r1")


async def _attach_in_a_routed_room(kit: RoomKit) -> None:
    await kit.process_inbound(_msg("first", "+15550000001"))


async def _commit_a_record(kit: RoomKit) -> None:
    await kit.attach_channel("r1", "sms1")
    record = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="sms1", channel_type="sms"),
        content=TextContent(body="record"),
    )
    await kit.commit_event("r1", record)


async def _mute(kit: RoomKit) -> None:
    await kit.attach_channel("r1", "sms1")
    await kit.mute("r1", "ws1")


async def _send_past_the_budget(kit: RoomKit) -> None:
    """A locked hook sends into its room on every event until its cascade's
    reentry budget is spent: the call past it is stored BLOCKED."""
    await kit.attach_channel("r1", "sms1")

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="echo")
    async def echo(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if _body(event) == "first":
            for n in range(kit._max_chain_depth * 10 + 1):
                await kit.send_event("r1", "sms1", TextContent(body=f"echo {n}"))
        return HookResult.allow()

    await kit.process_inbound(_msg("first"), room_id="r1")


class TestAnIndexedRecordInFlight:
    @pytest.mark.parametrize(
        ("held", "start"),
        [
            ("channel_attached", _attach_in_a_named_room),
            ("channel_attached", _attach_in_a_routed_room),
            ("record", _commit_a_record),
            ("channel_muted", _mute),
            ("reentry_loop_cap", _send_past_the_budget),
        ],
        ids=["step-2-attach", "step-2-routed-claim", "commit_event", "mute", "past-the-budget"],
    )
    async def test_reaches_the_lane_when_its_caller_leaves(self, held: str, start: Any) -> None:
        """A record nobody receives (step 2's attach in a room that exists, a
        host's record, a lifecycle record, a call past its budget) is
        committed and its caller cancelled before the commit returned: its
        index still reaches the room's lane, and the room's next message
        waits for no gap (RFC §13.5, §13.6). A routed message's sender is
        recorded on the binding its step 2 attached (§10.1 step 2)."""
        store = _SlowAnswer()
        kit = RoomKit(store=store, inbound_router=_FixedRouter(), delivery_gap_timeout=0.2)
        ws = SimpleChannel("ws1")
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        skipped: list[FrameworkEvent] = []
        kit.on("delivery_skipped")(_collect(skipped))

        store.held = held
        try:
            caller = asyncio.create_task(start(kit))
            await _cancelled_while(caller, store.written)
            store.answer.set()
            with pytest.raises(asyncio.CancelledError):
                await caller
            await asyncio.wait_for(kit.process_inbound(_msg("next"), room_id="r1"), 1)

            assert skipped == []
            assert _body(ws.delivered[-1]) == "next"
            binding = await kit.store.get_binding("r1", "sms1")
            assert binding is not None
            if start is _attach_in_a_routed_room:
                assert binding.participant_id == "+15550000001"
        finally:
            await kit.close()


class TestStepTwosAttachInARoomThatExists:
    @pytest.mark.parametrize(
        "start", [_attach_in_a_named_room, _attach_in_a_routed_room], ids=["named", "routed"]
    )
    async def test_runs_its_hooks_to_their_end_when_its_caller_leaves(self, start: Any) -> None:
        """Step 2 attaches the message's channel to a room that exists, and
        the caller is cancelled while an ON_CHANNEL_ATTACHED hook provisions
        the binding: the hook runs to its end, as a room step 2 opens does
        (RFC §13.6, §10.1 step 2)."""
        kit = RoomKit(inbound_router=_FixedRouter())
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1")
        entered, release = asyncio.Event(), asyncio.Event()
        provisioned: list[str] = []

        @kit.hook(HookTrigger.ON_CHANNEL_ATTACHED, name="provision")
        async def provision(event: RoomEvent, ctx: RoomContext) -> None:
            entered.set()
            await release.wait()
            provisioned.append(event.room_id)

        try:
            caller = asyncio.create_task(start(kit))
            await _cancelled_while(caller, entered)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert provisioned == ["r1"]
        finally:
            await kit.close()


def _collect(into: list[FrameworkEvent]) -> Any:
    async def collect(event: FrameworkEvent) -> None:
        into.append(event)

    return collect


class TestARoomStepTwoOpens:
    async def test_is_completed_when_its_caller_leaves_during_on_room_created(self) -> None:
        """A new sender's first message opens a room and its caller is
        cancelled while an ON_ROOM_CREATED hook provisions it: the room keeps
        its binding naming the sender, and their next message lands there
        (RFC §13.6, §10.1 step 2)."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        entered, release = asyncio.Event(), asyncio.Event()

        @kit.hook(HookTrigger.ON_ROOM_CREATED, name="provision")
        async def provision(event: RoomEvent, ctx: RoomContext) -> None:
            entered.set()
            await release.wait()

        try:
            caller = asyncio.create_task(kit.process_inbound(_msg("first", "+15550000001")))
            await _cancelled_while(caller, entered)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await caller
            second = await kit.process_inbound(_msg("second", "+15550000001"))

            [room] = await kit.store.list_rooms()
            assert second.event is not None and second.event.room_id == room.id
        finally:
            await kit.close()


class _SlowCreation(InMemoryStore):
    """A room's creation that waits, once armed, while another call creates
    the same room first."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = False
        self.creating = asyncio.Event()
        self.answer = asyncio.Event()

    async def create_room(self, room: Room) -> Room:
        if self.armed:
            self.armed = False
            self.creating.set()
            await self.answer.wait()
        return await super().create_room(room)


class TestARoomAnotherCallCreatedFirst:
    async def test_is_no_error_when_the_caller_left(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Step 2 creates the room the message names, another call creates it
        first, and the caller is gone meanwhile: the store's refusal is the
        race step 2 reads the room again for (RFC §10.1 step 2), never
        logged as a failed write."""
        store = _SlowCreation()
        kit = RoomKit(store=store)
        kit.register_channel(SimpleChannel("sms1"))
        store.armed = True
        try:
            caller = asyncio.create_task(kit.process_inbound(_msg("first"), room_id="r1"))
            await _cancelled_while(caller, store.creating)
            await kit.create_room(room_id="r1")
            with caplog.at_level(logging.DEBUG):
                store.answer.set()
                with pytest.raises(asyncio.CancelledError):
                    await caller
            assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []
        finally:
            await kit.close()


class _AnswerTimesOut(_SlowAnswer):
    """A commit whose answer never comes back: it times out, as a database
    round trip does, with a ``TimeoutError`` that carries no message."""

    async def commit_event(self, room_id: str, event: RoomEvent, **kw: Any) -> RoomEvent:
        if _body(event) == "first":
            self.written.set()
            await self.answer.wait()
            raise TimeoutError
        return await super().commit_event(room_id, event, **kw)


class TestAWriteThatFailsAfterItsCallerLeft:
    async def test_its_failure_is_logged_by_its_type(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A decided write fails once its caller was cancelled: the line it
        is logged under names the failure, its type when it has no message."""
        store = _AnswerTimesOut()
        kit = RoomKit(store=store)
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        try:
            caller = asyncio.create_task(kit.process_inbound(_msg("first"), room_id="r1"))
            await _cancelled_while(caller, store.written)
            with caplog.at_level(logging.ERROR, logger="roomkit"):
                store.answer.set()
                with pytest.raises(asyncio.CancelledError):
                    await caller
        finally:
            await kit.close()

        lines = [
            r.getMessage() for r in caplog.records if "a cancelled caller left" in r.getMessage()
        ]
        assert lines == ["Writes a cancelled caller left failed: TimeoutError"]


class TestTheKitsClose:
    async def test_the_writes_a_cancelled_caller_left_land_within_the_grace(
        self, close_grace: float
    ) -> None:
        """A host cancels a call whose commit is in flight and closes the kit
        at once: the writes are running work, so the close lets them end
        within its grace; the event is stored, its delivery, not started,
        dropped (RMK-1030, RFC §13.6)."""
        store = _SlowAnswer()
        kit = RoomKit(store=store)
        ws = SimpleChannel("ws1")
        kit.register_channel(SimpleChannel("sms1"))
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ws1")
        store.held = "first"

        caller = asyncio.create_task(kit.process_inbound(_msg("first"), room_id="r1"))
        await _cancelled_while(caller, store.written)
        closing = asyncio.ensure_future(kit.close())
        await asyncio.sleep(0)
        assert not closing.done()  # it gives the commit in flight its grace
        store.answer.set()
        await asyncio.wait_for(closing, 1)

        assert "first" in [_body(e) for e in await store.list_events("r1")]
        assert ws.delivered == []
        with pytest.raises(asyncio.CancelledError):
            await caller

    async def test_writes_outlasting_the_grace_are_cut(self, close_grace: float) -> None:
        """A decided write runs integrator code (an ``event_blocked`` handler
        that never answers): the close gives it its grace, then cuts it, the
        handler with it, and its caller reads the block's record stored,
        ``kit_closed`` (RMK-1030)."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        entered = asyncio.Event()
        steps: list[str] = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="block")
        async def block(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("spam")

        @kit.on("event_blocked")
        async def webhook(event: FrameworkEvent) -> None:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                steps.append("cut")
                raise

        caller = asyncio.create_task(kit.process_inbound(_msg("spam"), room_id="r1"))
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(kit.close(), 1)

        result = await asyncio.wait_for(caller, 1)
        assert steps == ["cut"]
        stored = await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))
        assert result.event is not None and result.event in stored
        assert (result.blocked, result.reason, result.cancellation_reason) == (
            True,
            "spam",
            "kit_closed",
        )
