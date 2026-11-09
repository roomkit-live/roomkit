"""A room's timers (RFC §5.1).

An ACTIVE room pauses once ``inactive_after_seconds`` passed since its last
activity, or since its creation before any, and ``paused_at`` records the
instant; a PAUSED room closes once ``closed_after_seconds`` passed since
``paused_at``; a delivered commit resumes a PAUSED room, and what is no
activity resumes nothing. Every locked section that reads the room's status
applies the transitions due at its instant first, each at the instant it fell
due, each emitting its framework event as soon as stored, before anything the
section decides next, its hook once the room lock is released.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from roomkit import (
    FrameworkEvent,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomContext,
    RoomEvent,
    RoomKit,
    RoomStatus,
    RoomTimers,
)
from roomkit.core._timer_transitions import TimerTransitions
from roomkit.core.locks import InMemoryLockManager
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import SystemContent, TextContent
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter
from roomkit.store.memory import InMemoryStore
from tests.conftest import make_event
from tests.test_framework import AILikeChannel, SimpleChannel

_HOOKS = (HookTrigger.ON_ROOM_PAUSED, HookTrigger.ON_ROOM_CLOSED)


class _Heard:
    """What the room's transitions and refusals report, in order: their
    framework events (``room_paused:timer``) and their hooks
    (``on_room_paused``)."""

    def __init__(self, kit: RoomKit) -> None:
        self.reports: list[str] = []
        self._hook = asyncio.Event()
        for name in ("room_paused", "room_closed", "room_refused_event"):
            kit.on(name)(self._on_event)
        for trigger in _HOOKS:
            kit.hook(trigger, execution=HookExecution.ASYNC)(self._on_hook)

    async def _on_event(self, event: FrameworkEvent) -> None:
        reason = event.data.get("reason", event.data.get("operation"))
        self.reports.append(f"{event.type}:{reason}")

    async def _on_hook(self, event: RoomEvent, ctx: RoomContext) -> None:
        assert isinstance(event.content, SystemContent)
        self.reports.append(event.content.code.removesuffix("_by_timer"))
        self._hook.set()

    @property
    def hooks(self) -> list[str]:
        return [report for report in self.reports if ":" not in report]

    async def hooks_fired(self, count: int) -> None:
        async def enough() -> None:
            while len(self.hooks) < count:
                self._hook.clear()
                await self._hook.wait()

        await asyncio.wait_for(enough(), timeout=1.0)


async def _kit(store: InMemoryStore | None = None, **room: Any) -> tuple[RoomKit, _Heard]:
    """A kit holding room ``r1``, its channel ``sms1`` bound, the room then
    written with *room*'s fields."""
    kit = RoomKit(store=store)
    kit.register_channel(SimpleChannel("sms1"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    stored = await kit.get_room("r1")
    await kit.store.update_room(stored.model_copy(update=room))
    return kit, _Heard(kit)


def _ago(seconds: float) -> datetime:
    return datetime.now(UTC) - timedelta(seconds=seconds)


def _message(body: str = "hi") -> InboundMessage:
    return InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body=body))


async def test_inactivity_counts_from_creation_before_any_activity() -> None:
    kit = RoomKit()
    created = await kit.create_room(room_id="r1", timers=RoomTimers(inactive_after_seconds=60))
    assert created.timers.last_activity_at is None
    created_at = _ago(120)
    await kit.store.update_room(created.model_copy(update={"created_at": created_at}))

    room = await kit.check_room_timers("r1")

    assert (room.status, room.paused_at) == (
        RoomStatus.PAUSED,
        created_at + timedelta(seconds=60),
    )
    await kit.close()


async def test_a_late_check_pauses_then_closes_each_at_its_instant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One write: the pause at its instant, the close at its own, each
    reported, its hook fired once the check has released the lock."""
    last = _ago(3600)
    kit, heard = await _kit(
        timers=RoomTimers(
            inactive_after_seconds=60, closed_after_seconds=120, last_activity_at=last
        )
    )
    writes: list[Room] = []
    update_room = kit.store.update_room

    async def counted(room: Room) -> Room:
        writes.append(room)
        return await update_room(room)

    monkeypatch.setattr(kit.store, "update_room", counted)

    room = await kit.check_room_timers("r1")

    assert (room.status, room.paused_at) == (RoomStatus.CLOSED, None)
    assert room.closed_at == last + timedelta(seconds=180)
    assert len(writes) == 1
    assert heard.reports == [
        "room_paused:timer",
        "room_closed:timer",
        "room_paused",
        "room_closed",
    ]
    await kit.close()


@pytest.mark.parametrize(("paused_for", "status"), [(30, RoomStatus.PAUSED), (90, None)])
async def test_the_close_timer_counts_from_paused_at(
    paused_for: int, status: RoomStatus | None
) -> None:
    paused_at = _ago(paused_for)
    kit, _heard = await _kit(
        status=RoomStatus.PAUSED,
        paused_at=paused_at,
        timers=RoomTimers(closed_after_seconds=60, last_activity_at=_ago(3600)),
    )

    room = await kit.check_room_timers("r1")

    if status is RoomStatus.PAUSED:
        assert (room.status, room.paused_at) == (RoomStatus.PAUSED, paused_at)
    else:
        assert (room.status, room.closed_at) == (
            RoomStatus.CLOSED,
            paused_at + timedelta(seconds=60),
        )
    await kit.close()


async def test_a_delivered_message_resumes_a_paused_room_and_reports_nothing() -> None:
    last = _ago(3600)
    kit, heard = await _kit(
        status=RoomStatus.PAUSED,
        paused_at=_ago(30),
        timers=RoomTimers(closed_after_seconds=3600, last_activity_at=last),
    )

    result = await kit.process_inbound(_message(), room_id="r1")

    room = await kit.get_room("r1")
    assert not result.blocked
    assert (room.status, room.paused_at) == (RoomStatus.ACTIVE, None)
    assert room.timers.last_activity_at is not None and room.timers.last_activity_at > last
    assert heard.reports == []
    await kit.close()


async def _blocked(kit: RoomKit) -> None:
    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def refuse(event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult.block("refused")

    assert (await kit.process_inbound(_message(), room_id="r1")).blocked


async def _record(kit: RoomKit) -> None:
    await kit.commit_event("r1", make_event(room_id="r1", channel_id="tracer", body="trace"))


async def _mute(kit: RoomKit) -> None:
    await kit.mute("r1", "sms1")


@pytest.mark.parametrize("operation", [_blocked, _record, _mute])
async def test_what_is_no_activity_resumes_nothing(operation: Any) -> None:
    paused_at = _ago(30)
    kit, _heard = await _kit(
        status=RoomStatus.PAUSED,
        paused_at=paused_at,
        timers=RoomTimers(closed_after_seconds=3600),
    )

    await operation(kit)

    room = await kit.get_room("r1")
    assert (room.status, room.paused_at) == (RoomStatus.PAUSED, paused_at)
    await kit.close()


async def test_a_message_after_its_room_fell_due_is_refused_as_if_on_time() -> None:
    """The gate of §10.1 step 6 pauses and closes the room first, reports
    both before the refusal, and fires their hooks once it released the lock."""
    last = _ago(3600)
    kit, heard = await _kit(
        timers=RoomTimers(
            inactive_after_seconds=60, closed_after_seconds=60, last_activity_at=last
        )
    )
    count = (await kit.get_room("r1")).event_count

    result = await kit.process_inbound(_message("too late"), room_id="r1")
    await heard.hooks_fired(2)

    room = await kit.get_room("r1")
    assert (result.blocked, result.reason) == (True, "room_closed")
    assert (room.status, room.closed_at) == (RoomStatus.CLOSED, last + timedelta(seconds=120))
    assert heard.reports == [
        "room_paused:timer",
        "room_closed:timer",
        "room_refused_event:inbound",
        "room_paused",
        "room_closed",
    ]
    stored = await kit.store.list_events("r1", after_index=count - 1)
    assert all(getattr(e.content, "body", None) != "too late" for e in stored)
    await kit.close()


class _SlowTransitionAnswer(InMemoryStore):
    """A transition's write whose answer travels back once armed: the room is
    written and the call has not returned yet, as over a round trip."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = False
        self.written = asyncio.Event()
        self.answer = asyncio.Event()

    async def update_room_if(self, room: Room, **expected: Any) -> Room | None:
        written = await super().update_room_if(room, **expected)
        if self.armed and written is not None:
            self.armed = False
            self.written.set()
            await self.answer.wait()
        return written


@pytest.mark.parametrize("entry", ["inbound", "check", "attach", "record"])
async def test_a_transition_its_caller_left_is_reported_whole(entry: str) -> None:
    """The message's caller is cancelled once the transitions' write is
    stored: each still takes its record, its framework event and its hook
    (RFC §5.1, §13.6), since no later section finds them due."""
    store = _SlowTransitionAnswer()
    kit, heard = await _kit(
        store,
        timers=RoomTimers(
            inactive_after_seconds=60, closed_after_seconds=60, last_activity_at=_ago(3600)
        ),
    )
    store.armed = True
    try:
        calls = {
            "inbound": lambda: kit.process_inbound(_message("too late"), room_id="r1"),
            "check": lambda: kit.check_room_timers("r1"),
            "attach": lambda: _attach(kit),
            "record": lambda: _record(kit),
        }
        caller = asyncio.create_task(calls[entry]())
        await asyncio.wait_for(store.written.wait(), 1)
        caller.cancel()
        store.answer.set()
        with pytest.raises(asyncio.CancelledError):
            await caller
        await heard.hooks_fired(2)

        records = await kit.store.list_events("r1", event_filter=EventFilter(include_blocked=True))
        codes = [e.content.code for e in records if isinstance(e.content, SystemContent)]
        assert codes[-2:] == ["room_paused_by_timer", "room_closed_by_timer"]
        assert await kit.store.get_binding("r1", "sms2") is None
        assert heard.reports == [
            "room_paused:timer",
            "room_closed:timer",
            "room_paused",
            "room_closed",
        ]
    finally:
        await kit.close()


async def _attach(kit: RoomKit) -> None:
    kit.register_channel(SimpleChannel("sms2"))
    await kit.attach_channel("r1", "sms2")


async def _update(kit: RoomKit) -> None:
    [record] = await kit.store.list_events("r1", limit=1)
    await kit.update_event("r1", record.id, metadata={"seen": True})


async def _delete(kit: RoomKit) -> None:
    [record] = await kit.store.list_events("r1", limit=1)
    await kit.delete_event("r1", record.id)


async def _send(kit: RoomKit) -> None:
    await kit.send_event("r1", "sms1", TextContent(body="hello"))


async def _close(kit: RoomKit) -> None:
    await kit.close_room("r1")


@pytest.mark.parametrize(
    "operation", [_attach, _mute, _record, _update, _delete, _send, _close, _blocked]
)
async def test_every_locked_section_applies_the_due_transitions_first(operation: Any) -> None:
    kit, heard = await _kit(
        timers=RoomTimers(inactive_after_seconds=60, last_activity_at=_ago(3600))
    )

    await operation(kit)
    await heard.hooks_fired(1)

    assert heard.reports[0] == "room_paused:timer"
    assert "room_paused" in heard.hooks
    await kit.close()


async def test_a_response_pass_applies_the_due_transitions_first() -> None:
    """A response's commit is a locked section of its own (§10.1 step 14):
    with no inactivity allowed, the message's gate pauses the room before
    the message resumes it, and the answer's pass does the same."""
    kit, heard = await _kit(timers=RoomTimers(inactive_after_seconds=0))
    kit.register_channel(AILikeChannel("ai"))
    await kit.attach_channel("r1", "ai")

    result = await kit.process_inbound(_message(), room_id="r1")
    await heard.hooks_fired(2)

    events = await kit.store.list_events("r1")
    assert not result.blocked
    assert [e.content.body for e in events if e.source.channel_id == "ai"] == ["AI reply"]
    assert heard.reports.count("room_paused:timer") == 2
    assert (await kit.get_room("r1")).status is RoomStatus.ACTIVE
    await kit.close()


async def test_the_component_applies_a_due_pause_with_what_it_was_given() -> None:
    """``TimerTransitions`` on its own dependencies: nothing due costs
    nothing and returns the room read; a due pause is written, recorded,
    reported, then its hook fired once the lock is free."""
    store = InMemoryStore()
    locks = InMemoryLockManager()
    reports: list[str] = []

    async def read_room(room_id: str, *, organization_id: str | None = None) -> Room:
        room = await store.get_room(room_id)
        assert room is not None
        return room

    async def record(room_id: str, event_type: Any, **kwargs: Any) -> None:
        reports.append(kwargs["code"])

    async def emit(name: str, **kwargs: Any) -> None:
        reports.append(f"{name}:{kwargs['data']['reason']}")

    async def fire_hook(room_id: str, trigger: HookTrigger, *args: Any, **kwargs: Any) -> None:
        reports.append(str(trigger))

    async def context(room_id: str) -> RoomContext:
        return RoomContext(room=await read_room(room_id))

    async def stop(room_id: str) -> None:
        reports.append("stopped")

    timers = TimerTransitions(
        store=store,
        lock_manager=locks,
        read_room=read_room,
        count_closed=lambda: None,
        stop_recordings=stop,
        write_record=record,
        emit=emit,
        fire_hook=fire_hook,
        build_context=context,
        background=set(),
        writes=set(),
    )
    idle = await store.create_room(
        Room(id="idle", timers=RoomTimers(inactive_after_seconds=60, last_activity_at=_ago(30)))
    )
    due = await store.create_room(
        Room(id="due", timers=RoomTimers(inactive_after_seconds=60, last_activity_at=_ago(90)))
    )
    hooks: list[asyncio.Task[None]] = []

    assert await timers.apply_due(idle, hooks=hooks) is idle
    async with locks.locked("due"):
        paused = await timers.apply_due(due, hooks=hooks)
        assert reports == ["room_paused_by_timer", "room_paused:timer"]
    await timers.settle("due", hooks)

    assert paused.status is RoomStatus.PAUSED
    assert (await read_room("due")).paused_at == paused.paused_at
    assert reports[-1] == str(HookTrigger.ON_ROOM_PAUSED)


async def test_close_room_on_a_room_its_timer_closes_returns_it_as_stored() -> None:
    """The transitions' records are counted in the room it returns."""
    kit, _heard = await _kit(
        timers=RoomTimers(
            inactive_after_seconds=60, closed_after_seconds=60, last_activity_at=_ago(3600)
        )
    )

    closed = await kit.close_room("r1")

    assert closed.status is RoomStatus.CLOSED
    assert closed == await kit.get_room("r1")
    await kit.close()


@pytest.mark.parametrize(
    "room",
    [
        {"timers": RoomTimers(inactive_after_seconds=60, last_activity_at=_ago(3600))},
        {
            "status": RoomStatus.PAUSED,
            "paused_at": _ago(3600),
            "timers": RoomTimers(closed_after_seconds=60),
        },
    ],
    ids=["pause", "close"],
)
async def test_a_commit_landing_before_the_transition_write_is_not_overwritten(
    room: dict[str, Any],
) -> None:
    """A streamed row or a greeting commits without the room lock: one that
    lands between the read a transition is computed from and its write
    leaves the room ACTIVE, its activity as the commit wrote it."""
    kit, heard = await _kit(**room)
    stale = await kit.get_room("r1")
    await kit.store.commit_event("r1", make_event(room_id="r1", channel_id="sms1", body="hi"))
    fresh = await kit.get_room("r1")

    await kit._room_timers.apply_due(stale)

    after = await kit.get_room("r1")
    assert (after.status, after.paused_at) == (RoomStatus.ACTIVE, None)
    assert after.timers.last_activity_at == fresh.timers.last_activity_at
    assert heard.reports == []
    await kit.close()


async def test_timer_hooks_read_activity_committed_after_the_transition_write() -> None:
    """A greeting off the room lock can land after the timer's status write."""
    store = _SlowTransitionAnswer()
    kit, heard = await _kit(
        store, timers=RoomTimers(inactive_after_seconds=60, last_activity_at=_ago(3600))
    )
    contexts: list[RoomContext] = []

    @kit.hook(HookTrigger.ON_ROOM_PAUSED, execution=HookExecution.ASYNC)
    async def observe(event: RoomEvent, context: RoomContext) -> None:
        contexts.append(context)

    store.armed = True
    checking = asyncio.create_task(kit.check_room_timers("r1"))
    try:
        await asyncio.wait_for(store.written.wait(), 1)
        await store.commit_event("r1", make_event(room_id="r1", channel_id="voice", body="hello"))
        store.answer.set()
        checked = await asyncio.wait_for(checking, 1)
        await heard.hooks_fired(1)
        assert checked == await kit.get_room("r1")
        assert checked.status is RoomStatus.ACTIVE
        assert len(contexts) == 1
        assert contexts[0].room == checked
        assert any(
            getattr(event.content, "body", None) == "hello" for event in contexts[0].recent_events
        )
    finally:
        store.answer.set()
        await kit.close()
