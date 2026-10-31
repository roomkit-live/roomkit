"""The kit's ``close()``, rule by rule (RMK-1030, RFC §12.10.4).

1. Admission stops at once: once ``close()`` started, every call raises
   ``KitClosedError``, whoever makes it, but the reads, which answer while
   the store is open, and the teardown calls, which return doing nothing.
2. Work not started is cut at once: a delivery queued behind another (one
   placed in an idle lane runs from its placement), a call waiting for its
   room's lock.
3. A model turn is cut at once, its stream and its tool calls: what it
   streamed is stored marked cancelled, an open tool call gets its end row.
4. Other running work (a hook, a store write, a transport send) gets one
   grace, then is cut; nothing is written or delivered after ``close()``.
5. Every call the close cuts returns one result, ``kit_closed``, with what it
   delivered and the answers already stored; never ``lane_cancelled``.
6. Kept: a close from the kit's own work does not wait for itself, concurrent
   closes share one shutdown (``tests/test_kit_close_doors.py``,
   ``tests/test_framework.py::TestOneShutdown``).
7. Bound: the grace, plus closing the channels, plus at most one unwind of
   1 s twice for cut work that ignores its cut, abandoned then.

The grace is 50 ms under test (the ``close_grace`` fixture), the unwind
30 ms (``cut_unwind``).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from roomkit import (
    ChannelBinding,
    ChannelCategory,
    ChannelType,
    ConferenceChannel,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    KitClosedError,
    MockConferenceBackend,
    RealtimeVoiceChannel,
    RoomKit,
    RoomStatus,
    TextContent,
    VideoChannel,
    VoiceChannel,
)
from roomkit.channels import _ai_steering
from roomkit.channels.agent import Agent
from roomkit.core import framework, task_utils
from roomkit.models.channel import ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.event import EventSource, RoomEvent
from roomkit.models.room import Room, RoomTimers
from roomkit.orchestration import Discussion
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    AIToolCall,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.sources.base import SourceProvider
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import SQLiteStore
from roomkit.turns import owner as turns_owner
from roomkit.turns.base import TurnTimeouts
from roomkit.turns.memory import InMemoryTurnBackend
from roomkit.video.backends.mock import MockVideoBackend
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.stt.mock import MockSTTProvider
from tests import redis_turn_mode
from tests.test_framework import AILikeChannel, SimpleChannel
from tests.test_kit_close_doors import _STORES


class _Transport(SimpleChannel):
    """A transport whose send of a body in *holds* waits: its seconds, or for
    ever when ``None``."""

    def __init__(self, channel_id: str, holds: dict[str, float | None] | None = None) -> None:
        super().__init__(channel_id)
        self.holds = holds or {}
        self.sending: dict[str, asyncio.Event] = {}
        self.sent: list[str] = []

    def entered(self, body: str) -> asyncio.Event:
        return self.sending.setdefault(body, asyncio.Event())

    async def deliver(self, event: RoomEvent, binding: Any, context: Any) -> Any:
        body = getattr(event.content, "body", "")
        self.entered(body).set()
        if body in self.holds:
            hold = self.holds[body]
            await (asyncio.Event().wait() if hold is None else asyncio.sleep(hold))
        self.sent.append(body)
        return await super().deliver(event, binding, context)


async def _kit(*targets: SimpleChannel, rooms: tuple[str, ...] = ("r",), **kw: Any) -> RoomKit:
    kit = RoomKit(**kw)
    kit.register_channel(SimpleChannel("sms"))
    for target in targets:
        kit.register_channel(target)
    for room_id in rooms:
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "sms")
        for target in targets:
            await kit.attach_channel(room_id, target.channel_id)
    return kit


def _message(body: str) -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body=body))


def _record(body: str) -> RoomEvent:
    return RoomEvent(
        room_id="r",
        source=EventSource(channel_id="sms", channel_type="sms"),
        content=TextContent(body=body),
    )


async def _bodies(kit: RoomKit, room_id: str = "r") -> list[str]:
    return [getattr(e.content, "body", "") for e in await kit.store.list_events(room_id)]


async def _a_send_held_for_ever(kit: RoomKit, transport: _Transport, room_id: str) -> None:
    """Keep the close in its grace: a send that never ends."""
    await kit.process_inbound(_message("held"), room_id=room_id, defer_delivery=True)
    await transport.entered("held").wait()


class _SlowWords(MockAIProvider):
    """Streams a long answer word by word; *started* once its first words are
    read."""

    def __init__(self, started: asyncio.Event) -> None:
        super().__init__(streaming=True)
        self.started = started

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        self.calls.append(context)
        yield StreamTextDelta(text="The ")
        await asyncio.sleep(0.01)
        self.started.set()
        for i in range(2000):
            await asyncio.sleep(0.01)
            yield StreamTextDelta(text=f"w{i} ")
        yield StreamDone(finish_reason="stop", usage={"input_tokens": 5, "output_tokens": 50})


# -- 1. Admission stops at once -------------------------------------------------
#
# Once close() started, the kit's calls fall in three classes: the reads
# answer, the teardown calls return doing nothing, every other call is
# refused. The classes are written out here, apart from the kit's own, so a
# call moved from one to another fails a test.

_READS: dict[str, tuple[Callable[[RoomKit, str], Awaitable[Any]], bool]] = {
    # The read, and whether it reads the store.
    "get_room": (lambda kit, room: kit.get_room(room), True),
    "get_timeline": (lambda kit, room: kit.get_timeline(room), True),
    "get_binding": (lambda kit, room: kit.get_binding(room, "sms"), True),
    "list_bindings": (lambda kit, room: kit.list_bindings(room), True),
    "list_members": (lambda kit, room: kit.list_members(room), True),
    "is_member": (lambda kit, room: kit.is_member(room, "u1"), True),
    "list_observations": (lambda kit, room: kit.list_observations(room), True),
    "list_read_markers": (lambda kit, room: kit.list_read_markers(room), True),
    "list_tasks": (lambda kit, room: kit.list_tasks(room), True),
    "regenerate_target": (lambda kit, room: kit.regenerate_target(room), True),
    "room_recordings": (lambda kit, room: kit.room_recordings(room), True),
    "room_strategy_is_current": (
        lambda kit, room: kit.room_strategy_is_current(room, Discussion),
        True,
    ),
    "sender_verification": (
        lambda kit, room: kit.sender_verification(
            _record("read"), RoomContext(room=Room(id=room), bindings=[])
        ),
        False,
    ),
    "source_health": (lambda kit, room: kit.source_health("sms"), False),
    "verification_of": (lambda kit, room: kit.verification_of("u1"), True),
}

_SYNC_READS: dict[str, Callable[[RoomKit], Any]] = {
    # The kit's synchronous reads: they answer during the close and after it.
    "get_channel": lambda kit: kit.get_channel("sms"),
    "hooks_for": lambda kit: kit.hooks_for(HookTrigger.BEFORE_BROADCAST),
    "identity_enabled_for": lambda kit: kit.identity_enabled_for(ChannelType.SMS),
    "list_channels": lambda kit: kit.list_channels(),
    "list_sources": lambda kit: kit.list_sources(),
    "room_strategy": lambda kit: kit.room_strategy("r"),
    "speak_queue": lambda kit: kit.speak_queue("r"),
}

_TEARDOWN: dict[str, tuple[Callable[[RoomKit], Any], Any]] = {
    # The call, and what it returns once the close started.
    "cancel_delivery": (lambda kit: kit.cancel_delivery("key"), 0),
    "cancel_task": (lambda kit: kit.cancel_task("task"), False),
    "disconnect_websocket": (lambda kit: kit.disconnect_websocket("ws", "conn"), None),
    "disconnect_voice": (lambda kit: kit.disconnect_voice(object()), None),
    "disconnect_video": (lambda kit: kit.disconnect_video(object()), None),
    "leave": (lambda kit: kit.leave(object()), None),
    "unsubscribe_room": (lambda kit: kit.unsubscribe_room("subscription"), False),
    "unsubscribe_websocket": (lambda kit: kit.unsubscribe_websocket("ws", "conn", "r"), None),
}

_REFUSED = sorted(
    name
    for name, member in inspect.getmembers(RoomKit, inspect.isfunction)
    if not name.startswith("_") and name not in {"close", *_READS, *_SYNC_READS, *_TEARDOWN}
)
"""Every other public method of the kit, a coroutine or not."""


class _Release(_Transport):
    """A transport whose send of ``held`` waits for :attr:`release`: the
    close stays in its grace until the test lets it end."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.release = asyncio.Event()

    async def deliver(self, event: RoomEvent, binding: Any, context: Any) -> Any:
        if getattr(event.content, "body", "") == "held":
            self.entered("held").set()
            await self.release.wait()
        return await super().deliver(event, binding, context)


async def _closing(kit: RoomKit) -> asyncio.Future[None]:
    """A close that started: one held in its grace by a send it waits for."""
    closing = asyncio.ensure_future(kit.close())
    await asyncio.sleep(0)  # the close starts
    return closing


@pytest.fixture(params=["memory", "sqlite", "postgres", "postgres-given-pool"])
async def kit_store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[tuple[ConversationStore, bool]]:
    """A store, and whether the kit's close closes it: a Postgres pool the
    integrator hands in stays theirs."""
    if request.param == "memory":
        yield InMemoryStore(), False
        return
    if request.param == "sqlite":
        yield SQLiteStore(tmp_path / "rooms.db"), True
        return
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        pytest.skip("POSTGRES_DSN not set")
    from roomkit.store.postgres import PostgresStore

    if request.param == "postgres":
        store = PostgresStore(dsn=dsn)
        await store.init(min_size=1, max_size=2)
        yield store, True
        return
    import asyncpg

    async def jsonb(conn: Any) -> None:  # as PostgresStore's own pool reads its rows
        await conn.set_type_codec(
            "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, init=jsonb)
    try:
        store = PostgresStore(pool=pool)
        await store.init()
        yield store, False
    finally:
        await pool.close()


@pytest.mark.parametrize("read", list(_READS))
async def test_a_read_answers_during_the_close_and_after_it_while_its_store_is_open(
    read: str, kit_store: tuple[ConversationStore, bool], close_grace: float
) -> None:
    """A read answers as before the close, during it and after it; after
    it, a store the kit closed refuses with KitClosedError, never its own
    error (RFC §12.10.4)."""
    store, closed_by_the_kit = kit_store
    call, reads_the_store = _READS[read]
    room_id, busy = f"r-{uuid4().hex[:8]}", f"busy-{uuid4().hex[:8]}"
    held = _Release("ws")
    kit = await _kit(held, rooms=(room_id, busy), store=store)
    await kit.process_inbound(_message("hello"), room_id=room_id)
    await _a_send_held_for_ever(kit, held, busy)
    before = await call(kit, room_id)
    closing = await _closing(kit)

    during = await call(kit, room_id)
    held.release.set()
    await closing

    assert during == before
    if closed_by_the_kit and reads_the_store:
        with pytest.raises(KitClosedError):
            await call(kit, room_id)
    else:
        assert await call(kit, room_id) == before


class _BeatingSource(SourceProvider):
    """A source whose health check starts a heartbeat, which posts a status
    naming room ``r`` once *release* is set."""

    def __init__(self, kit: RoomKit) -> None:
        self.kit = kit
        self.release = asyncio.Event()
        self.beats: list[asyncio.Future[None]] = []

    @property
    def name(self) -> str:
        return "beating"

    async def start(self, emit: Any) -> None:
        await asyncio.Event().wait()

    async def stop(self) -> None:
        pass

    async def healthcheck(self) -> Any:
        self.beats.append(asyncio.ensure_future(self._beat()))
        return await super().healthcheck()

    async def _beat(self) -> None:
        await self.release.wait()
        self.kit.status_bus.post("src", "heartbeat", "ok", metadata={"room_id": "r"})


async def test_work_a_read_starts_is_no_read_once_the_kit_closed(close_grace: float) -> None:
    """A task a read starts (a source's health check, run by
    ``source_health``) is no read: once the kit closed, what it sets off
    builds no room's context on the released store, and fires no hook."""
    held = _Release("ws")
    kit = await _kit(held)
    fired: list[Any] = []

    @kit.hook(HookTrigger.ON_STATUS_POSTED, execution=HookExecution.ASYNC, name="status")
    async def on_status(entry: Any, ctx: Any) -> None:
        fired.append(entry)

    source = _BeatingSource(kit)
    await kit.attach_source("sms", source)
    await _a_send_held_for_ever(kit, held, "r")
    closing = await _closing(kit)
    await kit.source_health("sms")
    held.release.set()
    await closing

    source.release.set()
    await asyncio.wait(source.beats, timeout=1)
    for _ in range(20):
        await asyncio.sleep(0)
    assert fired == []


@pytest.mark.parametrize("name", _REFUSED)
async def test_every_other_call_is_refused_once_the_close_started(name: str) -> None:
    """Any call but a read, a teardown call and close() is refused with
    KitClosedError once the close started, during it and after it, before it
    reads its arguments: a call that hands its work off (a delegation, a
    turn) as one that writes, a synchronous one (a channel's registration, a
    hook's, a WebSocket's subscription) as a coroutine (RFC §12.10.4)."""
    held = _Release("ws")
    kit = await _kit(held)
    await _a_send_held_for_ever(kit, held, "r")
    closing = await _closing(kit)

    async def call() -> Any:
        made = getattr(kit, name)()
        return await made if inspect.isawaitable(made) else made

    with pytest.raises(KitClosedError):
        await call()
    held.release.set()
    await closing
    with pytest.raises(KitClosedError):
        await call()


@pytest.mark.parametrize("name", list(_TEARDOWN))
async def test_a_teardown_call_returns_doing_nothing_once_the_close_started(name: str) -> None:
    """A teardown call returns quietly once the close started, during it and
    after it: the close tears down what it would (RFC §12.10.4)."""
    call, quiet = _TEARDOWN[name]
    held = _Release("ws")
    kit = await _kit(held)
    await _a_send_held_for_ever(kit, held, "r")
    closing = await _closing(kit)

    async def answer() -> Any:
        made = call(kit)
        return await made if inspect.isawaitable(made) else made

    during = await answer()
    held.release.set()
    await closing

    assert (during, await answer()) == (quiet, quiet)


@pytest.mark.parametrize("name", list(_SYNC_READS))
async def test_a_sync_read_answers_during_the_close_and_after_it(name: str) -> None:
    held = _Release("ws")
    kit = await _kit(held)
    await _a_send_held_for_ever(kit, held, "r")
    before = _SYNC_READS[name](kit)
    closing = await _closing(kit)

    during = _SYNC_READS[name](kit)
    held.release.set()
    await closing

    assert (during, _SYNC_READS[name](kit)) == (before, before)


async def test_a_call_from_a_hook_in_the_close_s_grace_is_refused(close_grace: float) -> None:
    """A hook still running as the kit closes calls into the kit: refused,
    while the call it runs for commits within the grace."""
    kit = await _kit(_Transport("ws"))
    entered, closing = asyncio.Event(), asyncio.Event()
    refused: list[type[BaseException]] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="late")
    async def late(event: RoomEvent, ctx: Any) -> HookResult:
        if getattr(event.content, "body", "") == "in hook":
            entered.set()
            await closing.wait()
            try:
                await kit.send_event("r", "sms", TextContent(body="from the hook"))
            except Exception as error:
                refused.append(type(error))
        return HookResult.allow()

    call = asyncio.ensure_future(kit.process_inbound(_message("in hook"), room_id="r"))
    await entered.wait()
    closed = asyncio.ensure_future(kit.close())
    closing.set()
    await closed

    assert refused == [KitClosedError]
    result = await call
    assert result.cancellation_reason == "kit_closed"
    assert "from the hook" not in await _bodies(kit)


async def test_a_call_an_observer_makes_as_the_kit_closes_is_refused(close_grace: float) -> None:
    """An observer still running as the kit closes commits into a room whose
    timer fell due: the call is refused (RMK-1030), no timer hook starts,
    and close() returns with nothing of the kit left running."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    await kit.create_room(room_id="r2", timers=RoomTimers(inactive_after_seconds=0))
    closing = asyncio.Event()
    refused: list[str] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: RoomEvent, ctx: RoomContext) -> None:
        await closing.wait()
        record = RoomEvent(
            room_id="r2",
            source=EventSource(channel_id="sms", channel_type="sms"),
            content=TextContent(body="audit record"),
        )
        try:
            await kit.commit_event("r2", record)
        except KitClosedError:
            refused.append("commit_event")

    @kit.hook(HookTrigger.ON_ROOM_PAUSED, name="paused")
    async def paused(event: RoomEvent, ctx: RoomContext) -> None:
        await asyncio.Event().wait()

    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body="hi")),
        room_id="r1",
    )
    closed = asyncio.ensure_future(kit.close())
    closing.set()
    await closed

    assert refused == ["commit_event"]
    running = [t.get_name() for t in asyncio.all_tasks() if not t.done()]
    assert [name for name in running if name.startswith("roomkit-")] == []


class _SaysByeAsItCloses(SimpleChannel):
    """A channel whose close sends a last message into its room."""

    kit: RoomKit | None = None
    refused: BaseException | None = None

    async def close(self) -> None:
        assert self.kit is not None
        try:
            await self.kit.send_event("r1", self.channel_id, TextContent(body="bye"))
        except KitClosedError as refusal:
            self.refused = refusal
        await super().close()


async def test_a_call_a_channel_s_close_makes_is_refused() -> None:
    """A channel's close sends a last message into its room: the call is
    refused, as every call once close() started (RMK-1030), and nothing is
    stored."""
    kit = RoomKit()
    channel = _SaysByeAsItCloses("sms")
    channel.kit = kit
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")

    await kit.close()

    assert isinstance(channel.refused, KitClosedError)
    assert "bye" not in [
        getattr(e.content, "body", None) for e in await kit.store.list_events("r1")
    ]


async def test_a_session_or_a_participant_arriving_once_the_close_started_is_refused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A channel's own doors refuse as the kit's do once the close started: a
    realtime session does not start, a conference arrival writes nothing,
    tells no one, and is no error in the log (RFC §12.10.4)."""
    held = _Release("ws")
    backend = MockConferenceBackend()
    conference = ConferenceChannel("conf", backend=backend, stt=MockSTTProvider())
    provider = MockRealtimeProvider()
    realtime = RealtimeVoiceChannel("rt", provider=provider, transport=MockRealtimeTransport())
    kit = await _kit(held, conference, realtime)
    joined: list[Any] = []
    kit.on("participant_joined")(joined.append)
    await _a_send_held_for_ever(kit, held, "r")
    closing = await _closing(kit)

    with pytest.raises(KitClosedError):
        await realtime.start_session("r", "p-realtime", "conn")
    with caplog.at_level(logging.WARNING, logger="roomkit"):
        await backend.simulate_participant_joined("r", "p-conference")
    members = await kit.list_members("r")
    held.release.set()
    await closing

    assert "connect" not in [c.method for c in provider.calls]
    assert (members, joined) == ([], [])
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


async def test_async_with_a_closed_kit_is_refused() -> None:
    kit = RoomKit()
    await kit.close()

    with pytest.raises(KitClosedError):
        async with kit:
            pass


async def test_the_close_returns_while_calls_keep_coming(close_grace: float) -> None:
    """Calls arriving with no gap do not hold the close open (RMK-1016):
    they are refused, and the close ends with its grace."""
    held = _Transport("ws", holds={"held": None})
    kit = await _kit(held)
    await _a_send_held_for_ever(kit, held, "r")
    stop = False
    refused: list[type[BaseException]] = []

    async def traffic() -> None:
        while not stop:
            try:
                await kit.process_inbound(_message("traffic"), room_id="r", defer_delivery=True)
            except KitClosedError as error:
                refused.append(type(error))
            await asyncio.sleep(0)

    producers = [asyncio.ensure_future(traffic()) for _ in range(4)]
    await asyncio.sleep(0)
    try:
        await asyncio.wait_for(kit.close(), 1)
    finally:
        stop = True
        await asyncio.gather(*producers)

    assert refused


# -- 2. Work not started is cut at once -----------------------------------------


@pytest.mark.parametrize("store", list(_STORES))
@pytest.mark.parametrize("lane", ["idle", "busy"])
async def test_a_delivery_placed_in_an_idle_lane_runs_one_queued_behind_is_never_sent(
    lane: str, store: str, tmp_path: Path, close_grace: float
) -> None:
    """A delivery queued behind another is dropped, never sent. One placed
    in its room's idle lane runs from its placement, taken by the lane or
    not as the close starts, whatever the store: it gets the grace, and is
    sent (RMK-1039)."""
    transport = _Transport("ws", holds={"held": close_grace / 5})
    kit = await _kit(transport, store=_STORES[store](tmp_path))
    if lane == "busy":
        await kit.process_inbound(_message("held"), room_id="r", defer_delivery=True)
        await transport.entered("held").wait()
    placed = await kit.process_inbound(_message("placed"), room_id="r", defer_delivery=True)
    assert placed.delivery is not None

    await kit.close()

    result = await placed.delivery.wait()
    sent = {"ws": "sent"} if lane == "idle" else {}
    assert result.cancellation_reason == "kit_closed"
    assert {channel: r.status for channel, r in result.delivery_results.items()} == sent
    assert transport.sent == (["placed"] if lane == "idle" else ["held"])


@pytest.fixture
def long_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """A grace no test waits out: work the close cuts at once is seen cut
    while the grace still runs, the test ending the work it holds."""
    monkeypatch.setattr(framework, "CLOSE_GRACE_S", 30.0)


_LOCKED_CALLS: dict[str, Callable[[RoomKit], Awaitable[Any]]] = {
    "process_inbound": lambda kit: kit.process_inbound(_message("waiter"), room_id="r"),
    "send_event": lambda kit: kit.send_event("r", "sms", TextContent(body="waiter")),
    "commit_event": lambda kit: kit.commit_event("r", _record("waiter")),
    "attach_channel": lambda kit: kit.attach_channel("r", "late"),
    "update_room_metadata": lambda kit: kit.update_room_metadata("r", {"waiter": True}),
    "close_room": lambda kit: kit.close_room("r"),
}


@pytest.mark.parametrize("door", list(_LOCKED_CALLS))
async def test_a_call_waiting_for_its_room_lock_is_cut_at_once(
    door: str, long_grace: None
) -> None:
    """A call into a room, or a room change, waiting for the room's lock is
    cut as the close starts, while the call holding the lock still has its
    grace: nothing of it is stored (RFC §12.10.4)."""
    transport = _Transport("ws")
    kit = await _kit(transport)
    kit.register_channel(SimpleChannel("late"))
    in_hook, release = asyncio.Event(), asyncio.Event()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="holds-the-lock")
    async def holds(event: RoomEvent, ctx: Any) -> HookResult:
        if getattr(event.content, "body", "") == "locker":
            in_hook.set()
            await release.wait()
        return HookResult.allow()

    locker = asyncio.ensure_future(kit.process_inbound(_message("locker"), room_id="r"))
    await in_hook.wait()
    waiter = asyncio.ensure_future(_LOCKED_CALLS[door](kit))
    for _ in range(10):
        await asyncio.sleep(0)
    assert not waiter.done()

    closing = asyncio.ensure_future(kit.close())
    await asyncio.wait({waiter}, timeout=1)
    assert (waiter.done(), locker.done()) == (True, False)
    release.set()
    await closing

    if door == "process_inbound":
        result = waiter.result()
        assert (result.blocked, result.reason, result.cancellation_reason) == (
            True,
            "kit_closed",
            "kit_closed",
        )
    else:
        assert isinstance(waiter.exception(), KitClosedError)
    locked = await locker
    assert (locked.event is not None, locked.cancellation_reason) == (True, "kit_closed")
    room = await kit.get_room("r")
    bindings = {b.channel_id for b in await kit.list_bindings("r")}
    assert (room.status, room.metadata.get("waiter"), "late" in bindings) == (
        RoomStatus.ACTIVE,
        None,
        False,
    )
    assert "waiter" not in await _bodies(kit)
    assert transport.sent == []  # stored within the grace, delivered to no one


class _SlowToParse(SimpleChannel):
    """A channel whose parse of ``late`` waits for :attr:`go`: its call
    reaches the room's lock only once the test lets it."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.parsing, self.go = asyncio.Event(), asyncio.Event()

    async def handle_inbound(self, message: InboundMessage, context: Any) -> RoomEvent:
        if getattr(message.content, "body", "") == "late":
            self.parsing.set()
            await self.go.wait()
        return await super().handle_inbound(message, context)


async def test_a_call_reaching_its_room_lock_once_the_close_started_is_cut(
    long_grace: None,
) -> None:
    """A call taken before the close that reaches its room's lock once the
    close started is cut there, as a call already waiting is: nothing of it
    is stored."""
    kit = RoomKit()
    slow = _SlowToParse("sms")
    kit.register_channel(slow)
    kit.register_channel(_Transport("ws"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "ws")
    call = asyncio.ensure_future(kit.process_inbound(_message("late"), room_id="r"))
    await slow.parsing.wait()
    closing = await _closing(kit)

    slow.go.set()
    result = await asyncio.wait_for(call, 1)
    await asyncio.wait_for(closing, 1)

    assert (result.blocked, result.reason, result.cancellation_reason) == (
        True,
        "kit_closed",
        "kit_closed",
    )
    assert "late" not in await _bodies(kit)


# -- 3. A model turn is cut at once ----------------------------------------------


async def test_a_streaming_turn_is_cut_at_once_its_text_stored_cancelled(
    long_grace: None,
) -> None:
    held = _Release("ws")
    started = asyncio.Event()
    kit = await _kit(held, rooms=("r", "busy"))
    kit.register_channel(Agent("agent", provider=_SlowWords(started)))
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    call = asyncio.ensure_future(kit.process_inbound(_message("Go."), room_id="r"))
    await started.wait()
    await _a_send_held_for_ever(kit, held, "busy")

    closing = asyncio.ensure_future(kit.close())
    result = await asyncio.wait_for(call, 1)
    assert not closing.done()  # the send in the other room still has its grace
    held.release.set()
    await closing

    assert result.cancellation_reason == "kit_closed"
    assert dict(result.response_metadata)["turns"]["agent"]["loop_end_reason"] == "cancelled"
    answers = [e for e in await kit.store.list_events("r") if e.source.channel_id == "agent"]
    assert [getattr(e.content, "body", "") for e in answers] == ["The "]
    assert answers[0].metadata.get("cancelled") is True


async def test_a_running_tool_call_is_cut_at_once_its_end_row_stored(long_grace: None) -> None:
    held = _Release("ws")
    entered = asyncio.Event()
    ran: list[str] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            ran.append("cut")
            raise
        return "found"

    tool = {"name": "lookup", "description": "Look up.", "parameters": {"type": "object"}}
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})],
            ),
            AIResponse(content="Found it.", finish_reason="stop"),
        ]
    )
    kit = await _kit(held, rooms=("r", "busy"))
    kit.register_channel(Agent("agent", provider=provider, tools=[tool], tool_handler=lookup))
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    call = asyncio.ensure_future(kit.process_inbound(_message("Go."), room_id="r"))
    await entered.wait()
    await _a_send_held_for_ever(kit, held, "busy")

    closing = asyncio.ensure_future(kit.close())
    result = await asyncio.wait_for(call, 1)
    assert not closing.done()  # the send in the other room still has its grace
    held.release.set()
    await closing

    assert ran == ["cut"]
    assert result.cancellation_reason == "kit_closed"
    ends = [e for e in await kit.store.list_events("r") if e.type == "tool_call_end"]
    assert [(e.content.status, e.content.error) for e in ends] == [("failed", "kit_closed")]


# -- 4. Other running work gets one grace, then is cut ----------------------------


async def test_a_send_ending_within_the_grace_is_delivered(close_grace: float) -> None:
    """Its handle, awaited while the close runs, reads the send it made."""
    transport = _Transport("ws", holds={"send": close_grace / 5})
    kit = await _kit(transport)
    call = await kit.process_inbound(_message("send"), room_id="r", defer_delivery=True)
    await transport.entered("send").wait()
    assert call.delivery is not None

    closing = asyncio.ensure_future(kit.close())
    result = await call.delivery.wait()
    await closing

    assert result.delivery_results["ws"].status == "sent"
    assert result.cancellation_reason == "kit_closed"


async def test_a_send_outlasting_the_grace_is_cut_and_reported_failed(close_grace: float) -> None:
    quick, held = _Transport("a"), _Transport("b", holds={"both": None})
    kit = await _kit(quick, held)
    call = await kit.process_inbound(_message("both"), room_id="r", defer_delivery=True)
    await held.entered("both").wait()

    await kit.close()

    assert call.delivery is not None
    result = await call.delivery.wait()
    assert result.cancellation_reason == "kit_closed"
    assert result.delivery_results["a"].status == "sent"
    cut = result.delivery_results["b"]
    assert cut.status == "failed" and cut.error is not None
    assert (cut.error.code, cut.error.retryable) == ("kit_closed", True)
    assert held.sent == []


async def test_a_hook_outlasting_the_grace_is_cut(close_grace: float) -> None:
    kit = await _kit(_Transport("ws"))
    started = asyncio.Event()
    steps: list[str] = []

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: RoomEvent, ctx: Any) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            steps.append("cut")
            raise

    # The call returns once its delivery set is done, its hook still
    # running (RFC §9.3): the close cuts the hook, not the call.
    result = await asyncio.wait_for(kit.process_inbound(_message("hi"), room_id="r"), 1)
    await started.wait()
    await kit.close()

    assert steps == ["cut"]
    assert result.cancellation_reason is None


class _NeverWrites(InMemoryStore):
    """A store whose commit of ``slow`` never ends, and says if it did."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.landed = False

    async def commit_event(self, room_id: str, event: RoomEvent, **kw: Any) -> RoomEvent:
        if getattr(event.content, "body", None) == "slow":
            self.entered.set()
            await asyncio.Event().wait()
            self.landed = True
        return await super().commit_event(room_id, event, **kw)


async def test_a_pass_the_close_cuts_starts_no_before_broadcast_observer(
    close_grace: float,
) -> None:
    """A message its BEFORE_BROADCAST chain allowed, whose commit never
    ends, is cut by the close: its ASYNC observers never start, nothing new
    starting once the close began, as in roomkit-rs (RFC §10.1 step 13,
    §12.10.4)."""
    store = _NeverWrites()
    kit = await _kit(_Transport("ws"), store=store)
    observed: list[str] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.ASYNC, name="watch")
    async def watch(event: RoomEvent, ctx: Any) -> None:
        observed.append(getattr(event.content, "body", ""))

    await kit.process_inbound(_message("fast"), room_id="r")
    call = asyncio.ensure_future(kit.process_inbound(_message("slow"), room_id="r"))
    await store.entered.wait()

    await kit.close()
    result = await call
    await kit.hook_engine.drain_observers()

    assert result.cancellation_reason == "kit_closed"
    assert observed == ["fast"]


async def test_a_write_outlasting_the_grace_is_cut_and_never_lands(close_grace: float) -> None:
    """A commit past its caller's decision (the caller cancelled) gets the
    grace, then is cut: nothing of it is written after close()."""
    store = _NeverWrites()
    kit = await _kit(_Transport("ws"), store=store)
    call = asyncio.ensure_future(kit.process_inbound(_message("slow"), room_id="r"))
    await store.entered.wait()
    call.cancel()

    await kit.close()

    await asyncio.gather(call, return_exceptions=True)
    assert not store.landed
    assert "slow" not in await _bodies(kit)


class _HeldRoomWrites(InMemoryStore):
    """A store whose room writes, once *held*, never end, and say if one
    did."""

    def __init__(self) -> None:
        super().__init__()
        self.held = False
        self.entered = asyncio.Event()
        self.landed = False

    async def update_room(self, room: Room) -> Room:
        if self.held:
            self.entered.set()
            await asyncio.Event().wait()
            self.landed = True
        return await super().update_room(room)


async def test_a_room_change_outlasting_the_grace_is_cut_and_never_lands(
    close_grace: float,
) -> None:
    store = _HeldRoomWrites()
    kit = await _kit(_Transport("ws"), store=store)
    store.held = True
    change = asyncio.ensure_future(kit.update_room_metadata("r", {"late": True}))
    await store.entered.wait()

    await kit.close()

    with pytest.raises(KitClosedError):
        await asyncio.wait_for(change, 1)
    assert not store.landed
    assert "late" not in (await kit.get_room("r")).metadata


class _AnswerNeverComes(InMemoryStore):
    """A store whose commit of ``slow`` writes its row, then never answers:
    the commit lands as the grace runs out."""

    def __init__(self) -> None:
        super().__init__()
        self.written = asyncio.Event()

    async def commit_event(self, room_id: str, event: RoomEvent, **kw: Any) -> RoomEvent:
        stored = await super().commit_event(room_id, event, **kw)
        if getattr(event.content, "body", None) == "slow":
            self.written.set()
            await asyncio.Event().wait()
        return stored


_COMMITTING_CALLS: dict[str, Callable[[RoomKit], Awaitable[Any]]] = {
    "process_inbound": lambda kit: kit.process_inbound(_message("slow"), room_id="r"),
    "send_event": lambda kit: kit.send_event("r", "sms", TextContent(body="slow")),
    "commit_event": lambda kit: kit.commit_event("r", _record("slow")),
}


@pytest.mark.parametrize("door", list(_COMMITTING_CALLS))
async def test_a_call_whose_commit_landed_as_it_was_cut_reports_it_stored(
    door: str, close_grace: float, cut_unwind: float
) -> None:
    """The close cuts a call whose commit had started: the call reads its
    event back within the unwind and reports it stored, ``kit_closed``,
    never refused; it is delivered to no one (RFC §12.10.4)."""
    store = _AnswerNeverComes()
    transport = _Transport("ws")
    kit = await _kit(transport, store=store)
    call = asyncio.ensure_future(_COMMITTING_CALLS[door](kit))
    await store.written.wait()

    await kit.close()

    result = await call
    [stored] = [e for e in await store.list_events("r") if e.content == TextContent(body="slow")]
    if door == "process_inbound":
        assert (result.event, result.blocked, result.cancellation_reason) == (
            stored,
            False,
            "kit_closed",
        )
        assert result.delivery_results == {}
    else:
        assert result == stored
    assert transport.sent == []


class _ReadBackFails(_AnswerNeverComes):
    """A store whose commit lands as the grace runs out, and that cannot
    read an event back."""

    async def get_event(self, event_id: str) -> RoomEvent | None:
        raise RuntimeError("the store is gone")


async def test_a_cut_commit_whose_read_back_fails_reports_nothing_stored_and_says_so(
    close_grace: float, cut_unwind: float, caplog: pytest.LogCaptureFixture
) -> None:
    """The limit of the read-back: one that fails leaves the call reported
    as not stored, whatever the store holds, and the log says so."""
    store = _ReadBackFails()
    kit = await _kit(_Transport("ws"), store=store)
    call = asyncio.ensure_future(kit.process_inbound(_message("slow"), room_id="r"))
    await store.written.wait()

    with caplog.at_level(logging.WARNING, logger="roomkit"):
        await kit.close()

    result = await call
    assert (result.blocked, result.reason, result.cancellation_reason) == (
        True,
        "kit_closed",
        "kit_closed",
    )
    assert [r for r in caplog.records if "reading it back failed" in r.getMessage()] != []


class _Stubborn:
    """Work that ignores its cut: it runs on until the test releases it."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.steps: list[str] = []

    async def run(self) -> None:
        self.started.set()
        while not self.release.is_set():
            with contextlib.suppress(asyncio.CancelledError):
                await self.release.wait()


class _StubbornSend(_Transport):
    def __init__(self, work: _Stubborn) -> None:
        super().__init__("ws")
        self.work = work

    async def deliver(self, event: RoomEvent, binding: Any, context: Any) -> Any:
        if getattr(event.content, "body", "") == "stubborn":
            await self.work.run()
        return await super().deliver(event, binding, context)


class _StubbornReads(InMemoryStore):
    """A store whose room reads, once *armed*, ignore their cut: an
    operation the store runs under the kit's resource lease."""

    def __init__(self, work: _Stubborn) -> None:
        super().__init__()
        self.work = work
        self.armed = False

    async def load_room_context(self, room_id: str) -> Any:
        if self.armed:
            await self.work.run()
        return await super().load_room_context(room_id)


class _StubbornSource(SourceProvider):
    def __init__(self, work: _Stubborn, *, in_stop: bool) -> None:
        self.work, self.in_stop = work, in_stop

    @property
    def name(self) -> str:
        return "stubborn"

    async def start(self, emit: Any) -> None:
        if not self.in_stop:
            await self.work.run()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.work.steps.append("receive task cut")
            raise

    async def stop(self) -> None:
        if self.in_stop:
            await self.work.run()


async def _stubborn_kit(case: str, work: _Stubborn) -> tuple[RoomKit, Any]:
    """A kit running *work* in the place *case* names, and the call it runs
    for (``None`` for the kit's own work)."""
    if case == "send":
        kit = await _kit(_StubbornSend(work))
        return kit, await kit.process_inbound(
            _message("stubborn"), room_id="r", defer_delivery=True
        )
    if case == "store-read":
        store = _StubbornReads(work)
        kit = await _kit(_Transport("ws"), store=store)
        store.armed = True
        return kit, asyncio.ensure_future(kit.process_inbound(_message("hi"), room_id="r"))
    kit = await _kit(_Transport("ws"))
    if case in ("source-start", "source-stop"):
        await kit.attach_source("sms", _StubbornSource(work, in_stop=case == "source-stop"))
        return kit, None
    if case == "tool":
        tool = {"name": "lookup", "description": "Look up.", "parameters": {"type": "object"}}
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})],
                ),
                AIResponse(content="Found it.", finish_reason="stop"),
            ]
        )

        async def lookup(name: str, arguments: dict[str, Any]) -> str:
            await work.run()
            return "late"

        kit.register_channel(Agent("agent", provider=provider, tools=[tool], tool_handler=lookup))
        await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
        return kit, asyncio.ensure_future(kit.process_inbound(_message("Go."), room_id="r"))
    if case == "on-complete":
        kit.register_channel(Agent("worker", provider=_SlowWords(asyncio.Event())))

        async def on_complete(result: Any) -> None:
            await work.run()

        await kit.delegate("r", "worker", "Go.", wait=False, on_complete=on_complete)
        return kit, None
    execution = HookExecution.ASYNC if case == "observer" else HookExecution.SYNC
    trigger = HookTrigger.AFTER_BROADCAST if case == "observer" else HookTrigger.BEFORE_BROADCAST

    @kit.hook(trigger, execution=execution, name="stubborn")
    async def stubborn(event: RoomEvent, ctx: Any) -> Any:
        await work.run()
        return HookResult.allow()

    return kit, asyncio.ensure_future(kit.process_inbound(_message("hi"), room_id="r"))


@pytest.mark.parametrize(
    "case",
    [
        "send",
        "observer",
        "hook",
        "tool",
        "store-read",
        "source-start",
        "source-stop",
        "on-complete",
    ],
)
async def test_work_ignoring_its_cut_is_abandoned_once_the_unwind_ran_out(
    case: str,
    close_grace: float,
    cut_unwind: float,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integrator code that ignores its cancellation (a send, a hook, a
    model turn's tool, a source, a delegation's on_complete) or a store
    operation stuck under the kit's resource lease does not hold the close:
    it is abandoned once the grace and the unwinds ran out, and named in the
    log once. A source whose ``stop()`` ignores its cut has its receive task
    cut all the same (RFC §12.10.4)."""
    # A channel's own close waits on a budget of its own (an agent's, for
    # its turns), and so does the turn backend across Redis (its cancel
    # bound): under test as short as the kit's unwind.
    monkeypatch.setattr(_ai_steering, "CLOSE_WAIT_S", cut_unwind)
    if redis_turn_mode.ENABLED:
        short = TurnTimeouts(cancel=cut_unwind)
        monkeypatch.setattr(
            turns_owner, "InMemoryTurnBackend", lambda: redis_turn_mode.backend(short)
        )
    work = _Stubborn()
    kit, call = await _stubborn_kit(case, work)
    if case in ("source-stop", "on-complete"):
        # The close runs the work: it stops the source, cancels the delegation.
        closing = asyncio.ensure_future(kit.close())
        await work.started.wait()
    else:
        await work.started.wait()
        closing = asyncio.ensure_future(kit.close())

    try:
        with caplog.at_level(logging.WARNING, logger="roomkit.framework"):
            await asyncio.wait_for(closing, 1)
        # Abandoned, not waited for, and named once in the log.
        logged = [r for r in caplog.records if "still running after their cut" in r.getMessage()]
        assert (len(logged), work.release.is_set()) == (1, False)
        if case == "source-stop":
            assert work.steps == ["receive task cut"]
    finally:
        work.release.set()
    if case == "send":
        cut = (await call.delivery.wait()).delivery_results["ws"]
        assert (cut.status, cut.error.code, cut.error.retryable) == ("failed", "kit_closed", True)
    if isinstance(call, asyncio.Future):
        await asyncio.gather(call, return_exceptions=True)


async def test_a_turn_left_running_logs_nothing_once_the_close_returned(
    close_grace: float,
    cut_unwind: float,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model turn whose tool ignores its cut, left running by the close,
    watches its owner no more: nothing of the kit logs after ``close()``
    returned, not an owner fallen silent (RFC §12.10.4)."""
    monkeypatch.setattr(_ai_steering, "CLOSE_WAIT_S", cut_unwind)
    backend = InMemoryTurnBackend()
    # An owner's silence noticed once the close has returned.
    backend.timeouts = TurnTimeouts(heartbeat_interval=0.01, heartbeat_timeout=0.3)
    work = _Stubborn()
    tool = {"name": "lookup", "description": "Look up.", "parameters": {"type": "object"}}
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})],
            ),
            AIResponse(content="Found it.", finish_reason="stop"),
        ]
    )

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        await work.run()
        return "late"

    kit = await _kit(_Transport("ws"), turn_backend=backend)
    kit.register_channel(Agent("agent", provider=provider, tools=[tool], tool_handler=lookup))
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    call = asyncio.ensure_future(kit.process_inbound(_message("Go."), room_id="r"))
    await work.started.wait()
    try:
        await asyncio.wait_for(kit.close(), 1)
        with caplog.at_level(logging.DEBUG, logger="roomkit"):
            caplog.clear()
            for _ in range(50):  # past the owner's heartbeat timeout
                await asyncio.sleep(0.01)
    finally:
        work.release.set()
        await asyncio.gather(call, return_exceptions=True)

    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


async def test_a_tool_awaiting_the_close_runs_on_and_logs_nothing_after_it(
    close_grace: float, cut_unwind: float, caplog: pytest.LogCaptureFixture
) -> None:
    """A tool awaits ``kit.close()`` in its own turn, then runs on: its turn,
    spared and let go by its owner, watches its owner no more, so nothing is
    logged after ``close()`` returned (RFC §12.10.4)."""
    backend = InMemoryTurnBackend()
    # An owner's silence noticed while the tool still runs after the close.
    backend.timeouts = TurnTimeouts(heartbeat_interval=0.01, heartbeat_timeout=0.05)
    closed, release = asyncio.Event(), asyncio.Event()
    tool = {"name": "quit", "description": "Quit.", "parameters": {"type": "object"}}
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="quit", arguments={})],
            ),
            AIResponse(content="after", finish_reason="stop"),
        ]
    )

    async def quit_and_run_on(name: str, arguments: dict[str, Any]) -> str:
        await kit.close()
        closed.set()
        await release.wait()
        return "closed"

    kit = await _kit(_Transport("ws"), turn_backend=backend)
    kit.register_channel(
        Agent("agent", provider=provider, tools=[tool], tool_handler=quit_and_run_on)
    )
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    call = asyncio.ensure_future(kit.process_inbound(_message("Go."), room_id="r"))
    try:
        await asyncio.wait_for(closed.wait(), 2)
        with caplog.at_level(logging.DEBUG, logger="roomkit"):
            caplog.clear()
            for _ in range(20):  # past the owner's heartbeat timeout
                await asyncio.sleep(0.01)
    finally:
        release.set()
        await asyncio.gather(call, return_exceptions=True)

    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


class _StuckVerb(Agent):
    """An agent whose ``stuck`` channel verb ignores its cut."""

    def __init__(self, channel_id: str, work: _Stubborn) -> None:
        super().__init__(channel_id, provider=MockAIProvider(responses=["hi"]))
        self.work = work

    async def channel_call(self, room_id: str, verb: str, arguments: dict[str, Any]) -> Any:
        if verb != "stuck":
            return await super().channel_call(room_id, verb, arguments)
        await self.work.run()
        return "late"


async def test_a_channel_verb_ignoring_its_cut_does_not_hold_the_close(
    close_grace: float,
    cut_unwind: float,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A channel verb (``kit.channel_call``) that ignores its cut: the close
    returns within its bound, the turn backend's close on its cancel bound
    across Redis, and a verb its backend still runs then is named once in
    the kit's line (RFC §12.10.4)."""
    if redis_turn_mode.ENABLED:
        short = TurnTimeouts(cancel=cut_unwind)
        monkeypatch.setattr(
            turns_owner, "InMemoryTurnBackend", lambda: redis_turn_mode.backend(short)
        )
    work = _Stubborn()
    kit = await _kit(_Transport("ws"))
    kit.register_channel(_StuckVerb("agent", work))
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    call = asyncio.ensure_future(kit.channel_call("agent", "r", "stuck"))
    await work.started.wait()
    try:
        with caplog.at_level(logging.WARNING, logger="roomkit"):
            await asyncio.wait_for(kit.close(), 1)
    finally:
        work.release.set()
        await asyncio.gather(call, return_exceptions=True)

    named = [r.getMessage() for r in caplog.records if "still running after" in r.getMessage()]
    if redis_turn_mode.ENABLED:
        assert len(named) == 1 and "channel-call-agent-stuck" in named[0]


class _Source(SourceProvider):
    """A source reading an open connection until cut; its ``stop()`` runs
    until *stop_release* is set, when given; with *emits*, it sends a
    message every few milliseconds."""

    def __init__(
        self, name: str, *, stop_release: asyncio.Event | None = None, emits: bool = False
    ) -> None:
        self._name, self.stop_release, self.emits = name, stop_release, emits
        self.steps: list[str] = []
        self.starts = 0
        self.reading = asyncio.Event()

    @property
    def name(self) -> str:
        return self._name

    async def start(self, emit: Any) -> None:
        self.starts += 1
        self.reading.set()
        try:
            while True:
                await asyncio.sleep(0.005)
                if self.emits:
                    await emit(_message("from the source"))
        except asyncio.CancelledError:
            self.steps.append("task cut")
            raise

    async def stop(self) -> None:
        self.steps.append("stop")
        if self.stop_release is not None:
            await self.stop_release.wait()


async def test_a_stuck_source_stop_keeps_no_other_source_running(
    close_grace: float, cut_unwind: float
) -> None:
    """The kit's own steps stop side by side within the one grace: a source
    whose ``stop()`` never ends holds no other step back, and every source's
    task is cut whatever its ``stop()`` does (RFC §12.10.4)."""
    kit = await _kit(_Transport("ws"))
    stuck, ordinary = _Source("a", stop_release=asyncio.Event()), _Source("b")
    await kit.attach_source("sms", stuck)
    await kit.attach_source("ws", ordinary)
    await asyncio.gather(stuck.reading.wait(), ordinary.reading.wait())

    try:
        await asyncio.wait_for(kit.close(), 1)
    finally:
        assert stuck.stop_release is not None
        stuck.stop_release.set()

    assert (stuck.steps, ordinary.steps) == (["stop", "task cut"], ["stop", "task cut"])


class _FailingStop(_Source):
    async def stop(self) -> None:
        raise ValueError("cannot stop")


async def test_a_stop_step_s_failure_is_logged_and_a_stuck_one_named(
    close_grace: float, cut_unwind: float, caplog: pytest.LogCaptureFixture
) -> None:
    """Each of the kit's own stop steps is told apart: one that fails is
    logged as it ends, though another outlasts the grace, and one that
    outlasts its unwind is named after what it stops."""
    kit = await _kit(_Transport("ws"))
    failing, stubborn = _FailingStop("a"), _StubbornSource(_Stubborn(), in_stop=True)
    await kit.attach_source("sms", failing)
    await kit.attach_source("ws", stubborn)
    await failing.reading.wait()

    try:
        with caplog.at_level(logging.WARNING, logger="roomkit"):
            await asyncio.wait_for(kit.close(), 1)
    finally:
        stubborn.work.release.set()

    failures = [r for r in caplog.records if r.exc_info and isinstance(r.exc_info[1], ValueError)]
    named = [r.getMessage() for r in caplog.records if "still running" in r.getMessage()]
    assert [r.getMessage() for r in failures] == [
        "Stopping the source sms failed as the kit closed"
    ]
    assert len(named) == 1 and "source-stop-ws" in named[0]


async def test_a_source_restarting_on_failure_stops_restarting_once_the_close_started(
    close_grace: float, caplog: pytest.LogCaptureFixture
) -> None:
    """A source that restarts on failure fails on a closing kit (its message
    refused) while its ``stop()`` runs: it is not restarted, and its end is
    no error in the log."""
    kit = await _kit(_Transport("ws"))
    source = _Source("restarting", stop_release=asyncio.Event(), emits=True)
    await kit.attach_source("sms", source, auto_restart=True, restart_delay=0.001)
    await source.reading.wait()

    try:
        with caplog.at_level(logging.WARNING, logger="roomkit"):
            await asyncio.wait_for(kit.close(), 1)
    finally:
        assert source.stop_release is not None
        source.stop_release.set()

    assert source.starts == 1
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []


async def _a_session_ending(kind: str) -> tuple[RoomKit, Callable[[], Any]]:
    """A kit with a voice or a video session, and the call that ends it."""
    if kind == "voice":
        voice = MockVoiceBackend()
        kit = RoomKit(voice=voice)
        kit.register_channel(VoiceChannel("voice", stt=MockSTTProvider(), backend=voice))
        await kit.create_room(room_id="r")
        await kit.attach_channel("r", "voice")
        session = await kit.join("r", "voice", participant_id="u")
        return kit, lambda: voice.simulate_client_disconnected(session)
    backend = MockVideoBackend()
    kit = RoomKit()
    video = VideoChannel("video", backend=backend)
    kit.register_channel(video)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "video")
    stream = await backend.connect("r", "u", "video")
    binding = ChannelBinding(room_id="r", channel_id="video", channel_type=ChannelType.VIDEO)
    video.bind_session(stream, "r", binding)

    async def unbind() -> None:
        video.unbind_session(stream)

    return kit, unbind


@pytest.mark.parametrize("kind", ["voice", "video"])
async def test_a_session_s_end_handler_ending_within_the_grace_ends(
    kind: str, close_grace: float
) -> None:
    """A channel runs the kit's work for a session on a task of its own (its
    ``*_session_ended`` handler): the close gives it the grace, as any
    running hook, rather than cut it as the channel closes."""
    kit, end_session = await _a_session_ending(kind)
    entered, steps = asyncio.Event(), []

    @kit.on(f"{kind}_session_ended")
    async def ended(event: Any) -> None:
        entered.set()
        try:
            await asyncio.sleep(close_grace / 5)  # an audit write, say
            steps.append("ended")
        except asyncio.CancelledError:
            steps.append("cut")
            raise

    await end_session()
    await entered.wait()
    await kit.close()

    assert steps == ["ended"]


class _ClosesAfterItsSend(_Transport):
    """A transport that says, as it closes, whether its send was cut first."""

    def __init__(self) -> None:
        super().__init__("ws", holds={"held": None})
        self.steps: list[str] = []

    async def deliver(self, event: RoomEvent, binding: Any, context: Any) -> Any:
        try:
            return await super().deliver(event, binding, context)
        except asyncio.CancelledError:
            self.steps.append("send cut")
            raise

    async def close(self) -> None:
        self.steps.append("closed")
        await super().close()


async def test_the_work_outlasting_the_grace_is_cut_before_the_channels_close(
    close_grace: float,
) -> None:
    transport = _ClosesAfterItsSend()
    kit = await _kit(transport)
    await _a_send_held_for_ever(kit, transport, "r")

    await kit.close()

    assert transport.steps == ["send cut", "closed"]


def _postgres_dsn() -> str:
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        pytest.skip("POSTGRES_DSN not set")
    return dsn


async def test_a_connection_held_past_the_unwind_does_not_hold_the_close_of_the_kit_s_pool(
    close_grace: float, cut_unwind: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store operation that holds a connection of the pool the store opened
    and ignores its cut: the close gives the pool's connections a moment to
    come back, then terminates them, and returns (RFC §12.10.4)."""
    from roomkit.store.postgres import PostgresStore

    monkeypatch.setattr(PostgresStore, "_close_timeout", cut_unwind)
    room = f"stuck-{uuid4().hex[:8]}"
    work = _Stubborn()

    class Stuck(PostgresStore):
        async def load_room_context(self, room_id: str) -> Any:
            if room_id == room and work.steps == ["armed"]:
                async with self._ensure_pool().acquire() as conn:
                    await conn.fetchval("SELECT 1")
                    await work.run()
            return await super().load_room_context(room_id)

    store = Stuck(dsn=_postgres_dsn())
    await store.init(min_size=1, max_size=3)
    kit = await _kit(_Transport("ws"), rooms=(room,), store=store)
    work.steps.append("armed")
    call = asyncio.ensure_future(kit.process_inbound(_message("hi"), room_id=room))
    await work.started.wait()

    try:
        await asyncio.wait_for(kit.close(), 2)
    finally:
        work.release.set()
        await asyncio.gather(call, return_exceptions=True)


async def test_a_room_lock_held_past_the_unwind_does_not_hold_the_close_of_its_pool(
    close_grace: float, cut_unwind: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hook that holds its room's advisory lock and ignores its cut: the
    lock manager's own pool is closed within its moment, its connection
    terminated, and the close returns (RFC §12.10.4)."""
    from roomkit.store.postgres_lock import PostgresAdvisoryLockManager

    monkeypatch.setattr(PostgresAdvisoryLockManager, "_close_timeout", cut_unwind)
    locks = PostgresAdvisoryLockManager(_postgres_dsn(), min_size=1, max_size=3)
    await locks.init()
    room = f"locked-{uuid4().hex[:8]}"
    kit = await _kit(_Transport("ws"), rooms=(room,), lock_manager=locks)
    work = _Stubborn()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="stubborn")
    async def stubborn(event: RoomEvent, ctx: Any) -> HookResult:
        await work.run()
        return HookResult.allow()

    call = asyncio.ensure_future(kit.process_inbound(_message("hi"), room_id=room))
    await work.started.wait()

    try:
        await asyncio.wait_for(kit.close(), 2)
    finally:
        work.release.set()
        await asyncio.gather(call, return_exceptions=True)


async def test_a_store_closed_by_its_owner_waits_for_its_query_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the kit's close bounds a store's pool: ``PostgresStore.close()``
    called by its owner waits for the query in flight, as it always did."""
    from roomkit.store.postgres import PostgresStore

    monkeypatch.setattr(PostgresStore, "_close_timeout", 0.01)
    store = PostgresStore(dsn=_postgres_dsn())
    await store.init(min_size=1, max_size=2)
    started = asyncio.Event()

    async def query() -> Any:
        async with store._ensure_pool().acquire() as conn:
            started.set()
            return await conn.fetchval("SELECT 'done' FROM pg_sleep(0.2)")

    running = asyncio.ensure_future(query())
    await started.wait()
    await asyncio.wait_for(store.close(), 2)

    assert await running == "done"


async def test_a_lock_manager_closed_by_its_owner_keeps_its_room_locked_until_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the kit's close bounds a lock manager's pool:
    ``PostgresAdvisoryLockManager.close()`` called by its owner waits for
    the holder of a room's lock, so no other process takes the room
    meanwhile."""
    from roomkit.store.postgres_lock import PostgresAdvisoryLockManager

    monkeypatch.setattr(PostgresAdvisoryLockManager, "_close_timeout", 0.01)
    first, second = (PostgresAdvisoryLockManager(_postgres_dsn()) for _ in range(2))
    await first.init()
    await second.init()
    room = f"held-{uuid4().hex[:8]}"
    holding, release = asyncio.Event(), asyncio.Event()

    async def hold() -> None:
        async with first.locked(room):
            holding.set()
            await release.wait()

    async def take() -> None:
        async with second.locked(room):
            pass

    holder = asyncio.ensure_future(hold())
    await holding.wait()
    closing = asyncio.ensure_future(first.close())
    taking = asyncio.ensure_future(take())
    try:
        done, _ = await asyncio.wait({closing, taking}, timeout=0.2)
        assert done == set()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(holder, closing, taking), 2)
        await second.close()


# -- 5. One result per cut call, kit_closed, with what it delivered ---------------


class _Batch(_Transport):
    """``m0`` is sent at once, ``m1``'s send never ends."""

    def __init__(self) -> None:
        super().__init__("ws", holds={"m1": None})


async def test_each_call_the_close_cuts_reads_kit_closed_with_what_it_delivered(
    close_grace: float,
) -> None:
    transport = _Batch()
    kit = await _kit(transport)
    handles = []
    for n in range(3):
        result = await kit.process_inbound(_message(f"m{n}"), room_id="r", defer_delivery=True)
        assert result.delivery is not None
        handles.append(result.delivery)
    await transport.entered("m1").wait()

    await kit.close()

    results = [await handle.wait() for handle in handles]
    assert [r.cancellation_reason for r in results] == ["kit_closed"] * 3
    assert [{k: v.status for k, v in r.delivery_results.items()} for r in results] == [
        {"ws": "sent"},
        {"ws": "failed"},
        {},
    ]


class _StuckOnAnswers(AILikeChannel):
    """An agent whose turn on another agent's answer never ends."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.started = asyncio.Event()

    async def on_event(self, event: RoomEvent, binding: Any, context: Any) -> Any:
        if event.source.channel_id == "ai":
            self.started.set()
            await asyncio.Event().wait()
        return ChannelOutput.empty()


async def test_a_cut_call_carries_the_answers_already_stored(close_grace: float) -> None:
    kit = await _kit(_Transport("ws"))
    stuck = _StuckOnAnswers("stuck")
    kit.register_channel(AILikeChannel("ai", response="hello"))
    kit.register_channel(stuck)
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("r", "stuck", category=ChannelCategory.INTELLIGENCE)
    call = asyncio.ensure_future(kit.process_inbound(_message("hi"), room_id="r"))
    await stuck.started.wait()

    await kit.close()

    result = await call
    assert result.cancellation_reason == "kit_closed"
    assert [getattr(e.content, "body", None) for e in result.response_events] == ["hello"]


# -- 7. Bound ------------------------------------------------------------------


def test_the_close_gives_five_seconds_of_grace_and_one_second_of_unwind() -> None:
    """The production bound, pinned once: every other test runs on the
    ``close_grace`` and ``cut_unwind`` fixtures."""
    assert (task_utils.CLOSE_GRACE_S, task_utils.CUT_UNWIND_S) == (5.0, 1.0)
    assert (framework.CLOSE_GRACE_S, framework.CUT_UNWIND_S) == (5.0, 1.0)


@pytest.mark.wallclock
async def test_the_close_takes_its_grace_with_every_kind_of_work_stuck(
    close_grace: float,
) -> None:
    """A send, a decided write, an observer and a hook holding its room's
    lock, none ever ending: the close cuts them all once its grace ran out."""
    held = _Transport("ws", holds={"held": None})
    store = _NeverWrites()
    rooms = ("send", "write", "observer", "lock")
    kit = await _kit(held, rooms=rooms, store=store)
    observing, locking = asyncio.Event(), asyncio.Event()

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="observer")
    async def observer(event: RoomEvent, ctx: Any) -> None:
        if getattr(event.content, "body", "") == "observed":
            observing.set()
            await asyncio.Event().wait()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="locker")
    async def locker(event: RoomEvent, ctx: Any) -> HookResult:
        if getattr(event.content, "body", "") == "locked":
            locking.set()
            await asyncio.Event().wait()
        return HookResult.allow()

    await _a_send_held_for_ever(kit, held, "send")
    calls = [
        asyncio.ensure_future(kit.process_inbound(_message("observed"), room_id="observer")),
        asyncio.ensure_future(kit.process_inbound(_message("locked"), room_id="lock")),
    ]
    write = asyncio.ensure_future(kit.process_inbound(_message("slow"), room_id="write"))
    await asyncio.gather(observing.wait(), locking.wait(), store.entered.wait())
    write.cancel()
    loop = asyncio.get_running_loop()
    started = loop.time()

    await asyncio.wait_for(kit.close(), 1)

    assert loop.time() - started < close_grace + 0.5
    await asyncio.gather(*calls, write, return_exceptions=True)
