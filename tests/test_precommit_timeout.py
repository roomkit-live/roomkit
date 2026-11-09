"""`process_timeout` bounds the whole pre-commit phase (RFC §13.6).

It bounded one region of it — the gates inside the room lock — and the setting
read as a promise it did not keep. Everything before the lock ran unbounded: the
context build, and `handle_inbound`, which is integrator code and can reach a
provider with no timeout of its own. So could the wait for the lock, which is
the pile-up the setting exists to stop: one stuck event holds a room's lock and
every later message queues behind it forever.

A store that hangs is not exotic — an exhausted connection pool and a network
that swallows packets without a reset both look exactly like this.
"""

from __future__ import annotations

import asyncio

import pytest

from roomkit import (
    FrameworkEvent,
    HookExecution,
    HookTrigger,
    KitClosedError,
    ProcessTimeoutError,
    RoomKit,
)
from roomkit.core.inbound_router import InboundRoomRouter
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.store.memory import InMemoryStore
from tests.test_framework import SimpleChannel

FAST = 0.2
# A budget the test watches run out: what is measured is that it does.
SHORT = 0.02


class HangingStore(InMemoryStore):
    """Healthy until armed, then one read never returns."""

    armed = False

    async def list_bindings(self, room_id: str):  # noqa: ANN201
        if self.armed:
            await asyncio.sleep(3600)
        return await super().list_bindings(room_id)


class HangingRoomRead(InMemoryStore):
    """Healthy until armed, then a read of a room never returns."""

    armed = False

    async def get_room(self, room_id: str):  # noqa: ANN201
        if self.armed:
            await asyncio.sleep(3600)
        return await super().get_room(room_id)


class HangingChannel(SimpleChannel):
    """A parser that reaches a provider without a timeout of its own."""

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


async def _kit(
    channel: SimpleChannel, store: InMemoryStore | None = None, budget: float = SHORT
) -> RoomKit:
    kit = RoomKit(store=store, process_timeout=budget)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    return kit


def _msg() -> InboundMessage:
    return InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="hi"))


async def _within_budget(kit: RoomKit):  # noqa: ANN202
    """Fails loudly rather than hanging the suite if the bound is gone."""
    return await asyncio.wait_for(kit.process_inbound(_msg()), timeout=5.0)


class TestTheWindowCoversWhatItClaims:
    async def test_a_store_that_hangs_during_the_context_build(self) -> None:
        store = HangingStore()
        kit = await _kit(SimpleChannel("sms1"), store)
        store.armed = True

        result = await _within_budget(kit)

        assert result.blocked is True
        assert result.reason == "process_timeout"

    async def test_a_channel_that_hangs_in_handle_inbound(self) -> None:
        kit = await _kit(HangingChannel("sms1"))

        result = await _within_budget(kit)

        assert result.blocked is True
        assert result.reason == "process_timeout"

    async def test_a_room_lock_held_by_something_stuck(self) -> None:
        """The pile-up: without this bound, every later message queues forever."""
        kit = await _kit(SimpleChannel("sms1"))

        holding = asyncio.Event()

        async def squatter() -> None:
            async with kit._lock_manager.locked("r1"):  # noqa: SLF001
                holding.set()
                await asyncio.sleep(3600)

        holder = asyncio.create_task(squatter())
        await asyncio.wait_for(holding.wait(), 5)
        try:
            result = await _within_budget(kit)
        finally:
            holder.cancel()

        assert result.blocked is True
        assert result.reason == "process_timeout"

    async def test_a_hook_that_hangs_inside_the_lock_still_times_out(self) -> None:
        """The region that already worked keeps working."""
        kit = await _kit(SimpleChannel("sms1"))

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def slow(event, ctx):  # noqa: ANN001, ANN202
            await asyncio.sleep(3600)

        result = await _within_budget(kit)

        assert result.blocked is True
        assert result.reason == "process_timeout"


class TestTheWindowStopsWhereItShould:
    async def test_a_healthy_message_commits(self) -> None:
        kit = await _kit(SimpleChannel("sms1"), budget=FAST)

        result = await _within_budget(kit)
        timeline = await kit.get_timeline("r1")

        assert result.blocked is False
        assert [e.id for e in timeline if e.id == result.event.id] == [result.event.id]

    async def test_a_refused_message_commits_nothing(self) -> None:
        """§13.6 forbids the converse too: no committed event reported blocked."""
        kit = await _kit(HangingChannel("sms1"))

        result = await _within_budget(kit)
        timeline = await kit.get_timeline("r1")

        assert result.blocked is True
        assert [getattr(e.content, "body", None) for e in timeline].count("hi") == 0

    @pytest.mark.wallclock
    async def test_the_budget_is_spent_once_across_both_regions(self) -> None:
        """A deadline, not the setting re-applied: 0.2s means 0.2s, not 0.4s."""
        store = HangingStore()
        kit = await _kit(SimpleChannel("sms1"), store, budget=FAST)
        store.armed = True

        started = asyncio.get_running_loop().time()
        await _within_budget(kit)
        elapsed = asyncio.get_running_loop().time() - started

        assert elapsed < FAST * 2


class TestDirectInjectionRaisesWhatItCannotReturn:
    """``send_event`` returns the committed event: a pre-commit expiry raises,
    the wait for the room lock bounded as on the inbound path (RMK-525, RFC
    §10.5, §13.6)."""

    @pytest.mark.parametrize("where", ["lock", "hook", "room read"])
    async def test_it_raises_and_writes_nothing(self, where: str) -> None:
        store = HangingRoomRead()
        kit = await _kit(SimpleChannel("sms1"), store)
        expired: list[FrameworkEvent] = []

        @kit.on("process_timeout")
        async def on_timeout(event: FrameworkEvent) -> None:
            expired.append(event)

        if where == "hook":

            @kit.hook(HookTrigger.BEFORE_BROADCAST)
            async def slow(event, ctx):  # noqa: ANN001, ANN202
                await asyncio.sleep(3600)

        holding = asyncio.Event()

        async def squatter() -> None:
            async with kit._lock_manager.locked("r1"):  # noqa: SLF001
                holding.set()
                await asyncio.sleep(3600)

        before = await kit.store.list_events("r1")
        store.armed = where == "room read"
        holder = asyncio.create_task(squatter()) if where == "lock" else None
        if holder is not None:
            await asyncio.wait_for(holding.wait(), 5)
        try:
            with pytest.raises(ProcessTimeoutError):
                await asyncio.wait_for(
                    kit.send_event("r1", "sms1", TextContent(body="hi")), timeout=5.0
                )
        finally:
            if holder is not None:
                holder.cancel()
        store.armed = False
        # A write still queued on the room's lock would land before this takes it.
        async with asyncio.timeout(5), kit._lock_manager.locked("r1"):  # noqa: SLF001
            after = await kit.store.list_events("r1")

        assert after == before
        assert [event.type for event in expired] == ["process_timeout"]


class HangingRouter(InboundRoomRouter):
    """A router whose lookup never returns."""

    async def route(self, *args: object, **kwargs: object) -> str | None:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


class HangingIdentityStore(InMemoryStore):
    """A store whose identity read for routing never returns."""

    async def resolve_identity(self, *args: object, **kwargs: object):  # noqa: ANN201
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


class SlowRouter(InboundRoomRouter):
    """A router that answers the one room after *delay* seconds."""

    def __init__(self, delay: float) -> None:
        self._delay = delay

    async def route(self, *args: object, **kwargs: object) -> str | None:
        await asyncio.sleep(self._delay)
        return "r1"


def _timeouts(kit: RoomKit) -> list[FrameworkEvent]:
    expired: list[FrameworkEvent] = []

    @kit.on("process_timeout")
    async def on_timeout(event: FrameworkEvent) -> None:
        expired.append(event)

    return expired


class TestOneDeadlineFromTheCallsStart:
    """RFC §13.6: the bound is one deadline from the call's start to its
    decision, routing included, so a router or a lock that hangs holds the
    caller no longer than the bound."""

    @pytest.mark.parametrize("hangs", ["the router", "the identity read"])
    async def test_routing_that_hangs_is_refused_with_no_room(self, hangs: str) -> None:
        store = HangingIdentityStore() if hangs == "the identity read" else None
        kit = RoomKit(
            store=store,
            process_timeout=SHORT,
            inbound_router=HangingRouter() if hangs == "the router" else None,
        )
        kit.register_channel(SimpleChannel("sms1"))
        expired = _timeouts(kit)

        result = await _within_budget(kit)

        assert (result.blocked, result.reason) == (True, "process_timeout")
        assert [(e.room_id, e.channel_id) for e in expired] == [(None, "sms1")]
        assert await kit.store.list_rooms() == []
        await kit.close()

    @pytest.mark.wallclock
    async def test_routing_spends_the_budget_the_rest_of_the_call_has(self) -> None:
        """A 40 ms router and a 30 ms hook do not fit in a 50 ms budget."""
        kit = await _kit(SimpleChannel("sms1"), budget=0.05)
        kit._inbound_router = SlowRouter(0.04)  # noqa: SLF001

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def slow(event, ctx):  # noqa: ANN001, ANN202
            await asyncio.sleep(0.03)

        result = await _within_budget(kit)

        assert (result.blocked, result.reason) == (True, "process_timeout")
        await kit.close()

    async def test_step_two_s_attach_is_bounded_on_its_wait_for_the_lock(self) -> None:
        """Past the wait the message is refused, never let in unattached."""
        kit = await _kit(SimpleChannel("sms1"))
        kit.register_channel(SimpleChannel("sms2"))
        holding = asyncio.Event()

        async def squatter() -> None:
            async with kit._lock_manager.locked("r1"):  # noqa: SLF001
                holding.set()
                await asyncio.sleep(3600)

        holder = asyncio.create_task(squatter())
        await asyncio.wait_for(holding.wait(), 5)
        message = InboundMessage(channel_id="sms2", sender_id="u1", content=TextContent(body="hi"))
        try:
            result = await asyncio.wait_for(
                kit.process_inbound(message, room_id="r1"), timeout=5.0
            )
        finally:
            holder.cancel()

        assert (result.blocked, result.reason) == (True, "process_timeout")
        assert await kit.store.get_binding("r1", "sms2") is None
        await kit.close()

    async def test_the_scope_read_before_a_room_s_own_message_draws_is_bounded(self) -> None:
        """A message the room's work sends into it for an organization has
        the room read before it draws its pass (RFC §17.2, §9.5.1), within
        the call's deadline: a read that hangs refuses it."""
        store = HangingRoomRead()
        kit = RoomKit(store=store, process_timeout=SHORT)
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1", organization_id="org")
        await kit.attach_channel("r1", "sms1")
        answered: list[tuple[bool, str | None]] = []
        done = asyncio.Event()

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
        async def again(event: RoomEvent, ctx: RoomContext) -> None:
            if done.is_set():
                return
            store.armed = True
            try:
                result = await kit.process_inbound(_msg(), room_id="r1", organization_id="org")
                answered.append((result.blocked, result.reason))
            finally:
                store.armed = False
                done.set()

        await kit.process_inbound(_msg(), room_id="r1")
        await asyncio.wait_for(done.wait(), 5)

        assert answered == [(True, "process_timeout")]
        await kit.close()


async def test_a_scope_read_under_a_locked_hook_keeps_close_grace() -> None:
    """A nested call already in the locked section keeps its close grace."""

    class HeldRoomRead(InMemoryStore):
        armed = False

        def __init__(self) -> None:
            super().__init__()
            self.reading = asyncio.Event()
            self.release = asyncio.Event()

        async def get_room(self, room_id: str) -> Room | None:
            if self.armed:
                self.reading.set()
                await self.release.wait()
            return await super().get_room(room_id)

    store = HeldRoomRead()
    kit = RoomKit(store=store, process_timeout=5)
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="r1", organization_id="org")
    await kit.attach_channel("r1", "sms1")
    results = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def again(event: RoomEvent, ctx: RoomContext) -> None:
        if event.content == TextContent(body="hi"):
            store.armed = True
            results.append(
                await kit.process_inbound(
                    InboundMessage(
                        channel_id="sms1", sender_id="u1", content=TextContent(body="again")
                    ),
                    room_id="r1",
                    organization_id="org",
                )
            )

    call = asyncio.create_task(kit.process_inbound(_msg(), room_id="r1"))
    await asyncio.wait_for(store.reading.wait(), 5)
    close = asyncio.create_task(kit.close())
    await asyncio.sleep(0)
    with pytest.raises(KitClosedError):
        await kit.create_room(room_id="late")
    store.armed = False
    store.release.set()
    await asyncio.wait_for(call, 5)
    await asyncio.wait_for(close, 5)
    assert len(results) == 1
    assert results[0].blocked is False
    assert "again" in [getattr(row.content, "body", None) for row in await kit.get_timeline("r1")]
