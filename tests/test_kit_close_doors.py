"""``kit.close()`` and the kit's own work (RMK-526, RMK-1015, RMK-1019,
RFC §12.10.4, §23.3).

A turn someone awaits ends ``cancelled`` as on the neighbouring door: an
inline delegation ends ``cancelled``, its ON_TASK_COMPLETED fired before the
store is sealed, as a background one does; an awaited room turn returns its
``cancelled`` turn, never a ``CancelledError`` its caller did not ask for. A
``close()`` called from inside work the kit holds (a tool, a delivery, a
session's end, an announcement) does not wait for that work, and one that
joins a running close is spared by its own markers. What the kit holds as it
closes (a background run, a hook task a channel's close starts, a timer
check) ends with the close.

The close's rules, one by one, are in ``tests/test_kit_close.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from types import FrameType
from typing import Any
from uuid import uuid4

import pytest

from roomkit import (
    ChannelBinding,
    ChannelType,
    ConferenceChannel,
    HookExecution,
    HookTrigger,
    HostChannel,
    KitClosedError,
    MockConferenceBackend,
    RealtimeVoiceChannel,
    RoomKit,
    RoomStatus,
    SpeakQueueChange,
    SpeakQueueEvent,
    VideoChannel,
    VoiceChannel,
)
from roomkit.channels.agent import Agent
from roomkit.core.task_utils import hold_task, run_to_its_end
from roomkit.models.channel import ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.room import RoomTimers
from roomkit.models.trace import ProtocolTrace
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    AIToolCall,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import SQLiteStore
from roomkit.tasks.posting import send_posted
from roomkit.video.backends.mock import MockVideoBackend
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.stt.mock import MockSTTProvider
from tests.orchestration.test_strategy_discussion import _provider, _say
from tests.orchestration.test_strategy_discussion import _room as _discussion
from tests.tasks.test_cancel_task import _Endless
from tests.test_framework import AILikeChannel, SimpleChannel


class _SlowWords(MockAIProvider):
    """Streams a long answer word by word, and says when it started; with
    *answers_first*, its first generation answers at once."""

    def __init__(self, started: asyncio.Event, *, answers_first: bool = False) -> None:
        super().__init__(streaming=True)
        self.started = started
        self.answers_first = answers_first

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        self.calls.append(context)
        if self.answers_first and len(self.calls) == 1:
            yield StreamTextDelta(text="Quick.")
            yield StreamDone(finish_reason="stop", usage={"input_tokens": 1, "output_tokens": 1})
            return
        yield StreamTextDelta(text="The ")
        # The reader reads what came before the test hears it (RFC §10.6.4).
        await asyncio.sleep(0.05)
        self.started.set()
        for i in range(2000):
            await asyncio.sleep(0.01)
            yield StreamTextDelta(text=f"w{i} ")
        yield StreamDone(finish_reason="stop", usage={"input_tokens": 5, "output_tokens": 50})


# Longer than any test here takes, its own deadlines included.
_LOOP_DEADLINE_S = 10.0


class _LoopNeverYieldedError(Exception):
    """A test ran past :data:`_LOOP_DEADLINE_S`: interrupted where it was."""


@pytest.fixture
def loop_keeps_turning() -> Iterator[None]:
    """Fail, not hang, when the close loops without ever yielding to the
    event loop (RMK-1038): no ``asyncio`` deadline fires then, so an alarm
    interrupts the code where it is, and the test fails. No guard where
    there is no alarm (Windows). A timer set before is set again after,
    less the time the test took."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    interrupted: list[str] = []

    def interrupt(signum: int, frame: FrameType | None) -> None:
        interrupted.append("".join(traceback.format_stack(frame, limit=4)))
        raise _LoopNeverYieldedError(interrupted[0])

    previous = signal.signal(signal.SIGALRM, interrupt)
    started = time.monotonic()
    delay, interval = signal.setitimer(signal.ITIMER_REAL, _LOOP_DEADLINE_S)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if delay:
            left = max(delay - (time.monotonic() - started), 1e-6)
            signal.setitimer(signal.ITIMER_REAL, left, interval)
    assert not interrupted, f"interrupted after {_LOOP_DEADLINE_S} s at:\n{interrupted[0]}"


def _ended_tasks(kit: RoomKit) -> list[tuple[str, Any]]:
    ended: list[tuple[str, Any]] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_task(event: Any, context: Any) -> None:
        ended.append((str(event.metadata.get("task_status")), event.metadata.get("error")))

    return ended


# -- a delegation under kit.close() --------------------------------------------


@pytest.mark.parametrize("shared", [False, True], ids=["alone", "shared-channel"])
@pytest.mark.parametrize("wait", [True, False], ids=["inline", "background"])
async def test_a_delegation_ends_cancelled_with_its_hook(
    wait: bool, shared: bool, caplog: pytest.LogCaptureFixture
) -> None:
    started = asyncio.Event()
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(Agent("worker", provider=_SlowWords(started)))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    ended = _ended_tasks(kit)
    share = ["sms"] if shared else None

    delegation = asyncio.ensure_future(
        kit.delegate("r", "worker", "Go.", wait=wait, share_channels=share)
    )
    if not wait:
        await delegation
    await started.wait()
    with caplog.at_level(logging.WARNING, logger="roomkit"):
        await kit.close()
        task = await asyncio.wait_for(delegation, timeout=5.0)

    assert task.result is not None
    assert (task.result.status, task.result.error, task.result.output) == (
        "cancelled",
        "cancelled",
        None,
    )
    assert ended == [("cancelled", "cancelled")]
    assert [r.message for r in caplog.records if r.levelno >= logging.WARNING] == []


async def test_the_close_waits_for_the_end_of_a_task_being_cancelled(
    loop_keeps_turning: None,
) -> None:
    """``cancel_task`` ends its task once the task's work has ended, its
    on_complete included: a close meanwhile waits for that end, within its
    grace, and returns once it is over (RMK-1038)."""
    kit = RoomKit()
    worker = _Endless()
    kit.register_channel(Agent("worker", provider=worker))
    await kit.create_room(room_id="r")
    completing, release = asyncio.Event(), asyncio.Event()
    completed: list[str] = []

    async def on_complete(result: Any) -> None:
        completing.set()
        await release.wait()
        completed.append(str(result.status))

    task = await kit.delegate("r", "worker", "Go.", on_complete=on_complete)
    await asyncio.wait_for(worker.started.wait(), 1)
    cancel = asyncio.ensure_future(kit.cancel_task(task.id))
    await asyncio.wait_for(completing.wait(), 1)
    closing = asyncio.ensure_future(kit.close())
    # Long enough for a close waiting for no end to return (7 turns).
    for _ in range(20):
        await asyncio.sleep(0)
    assert not closing.done()
    release.set()

    await asyncio.wait_for(closing, 1)
    assert await cancel is True
    assert completed == ["cancelled"]


async def test_a_close_from_a_held_run_does_not_wait_for_it() -> None:
    kit = RoomKit()
    closed = asyncio.Event()

    async def run() -> None:
        await kit.close()
        closed.set()

    hold_task(kit, run())
    await asyncio.wait_for(closed.wait(), timeout=5.0)


@pytest.mark.in_process(
    "the kit closes from a tool in its own turn: an executor elsewhere holds no kit"
)
async def test_a_close_from_an_inline_workers_tool_does_not_wait_for_it() -> None:
    kit = RoomKit()
    closed = asyncio.Event()

    async def shutdown(name: str, arguments: dict[str, Any]) -> str:
        await kit.close()
        closed.set()
        return "closed"

    tool = {"name": "shutdown", "description": "Shut down.", "parameters": {"type": "object"}}
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="shutdown", arguments={})],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ]
    )
    kit.register_channel(Agent("worker", provider=provider, tools=[tool], tool_handler=shutdown))
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")

    delegation = asyncio.ensure_future(kit.delegate("r", "worker", "Go.", wait=True))
    await asyncio.wait_for(closed.wait(), timeout=5.0)
    await asyncio.wait_for(asyncio.gather(delegation, return_exceptions=True), timeout=5.0)


# -- a room turn under kit.close() ----------------------------------------------


@pytest.mark.parametrize(
    ("door", "entry"),
    [
        ("awaited", "process_inbound"),
        ("deferred", "process_inbound"),
        ("awaited", "send_event"),
        ("awaited", "regenerate_response"),
    ],
)
async def test_a_room_turn_returns_cancelled_to_its_caller(
    door: str, entry: str, caplog: pytest.LogCaptureFixture
) -> None:
    started = asyncio.Event()
    kit = RoomKit()
    kit.register_channel(SimpleChannel("tx"))
    provider = _SlowWords(started, answers_first=entry == "regenerate_response")
    kit.register_channel(Agent("agent", provider=provider))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    responses: list[str] = []
    message = InboundMessage(channel_id="tx", sender_id="u", content=TextContent(body="Go."))
    if entry == "regenerate_response":
        await kit.process_inbound(message)

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def on_response(event: Any, context: Any) -> None:
        responses.append(str(event.loop_end_reason))

    if entry == "send_event":
        call: Any = asyncio.ensure_future(kit.send_event("r", "tx", TextContent(body="Go.")))
    elif entry == "regenerate_response":
        call = asyncio.ensure_future(kit.regenerate_response("r"))
    else:
        call = asyncio.ensure_future(
            kit.process_inbound(message, defer_delivery=door == "deferred")
        )
    await started.wait()
    with caplog.at_level(logging.WARNING, logger="roomkit"):
        await kit.close()
        result = await asyncio.wait_for(call, timeout=5.0)
        if door == "deferred":
            await asyncio.wait_for(result.delivery.wait(), timeout=5.0)

    if entry != "send_event":
        turns = dict(result.response_metadata).get("turns", {})
        assert turns["agent"]["loop_end_reason"] == "cancelled"
        assert result.error is None
    assert responses == []
    assert [r.message for r in caplog.records if r.levelno >= logging.WARNING] == []


# -- a close from inside work the kit holds -------------------------------------


def _closing_agent(kit: RoomKit, closed: asyncio.Event) -> Agent:
    """An agent whose one tool closes the kit, then answers."""

    async def shutdown(name: str, arguments: dict[str, Any]) -> str:
        await kit.close()
        closed.set()
        return "closed"

    tool = {"name": "shutdown", "description": "Shut down.", "parameters": {"type": "object"}}
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="shutdown", arguments={})],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ]
    )
    return Agent("agent", provider=provider, tools=[tool], tool_handler=shutdown)


@pytest.mark.in_process(
    "the kit closes from a tool in its own turn: an executor elsewhere holds no kit"
)
@pytest.mark.parametrize(
    "door", ["room-turn", "inline", "inline-shared", "background"], ids=lambda d: d
)
async def test_a_close_from_a_tool_returns_and_seals_the_kit(door: str) -> None:
    """A tool that closes the kit while its turn is awaited, or runs as a
    delegated task, does not wait for itself (RMK-526)."""
    kit = RoomKit()
    closed = asyncio.Event()
    kit.register_channel(_closing_agent(kit, closed))
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")

    if door == "room-turn":
        await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
        message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Go."))
        call: Any = asyncio.ensure_future(kit.process_inbound(message))
    else:
        share = ["sms"] if door == "inline-shared" else None
        call = asyncio.ensure_future(
            kit.delegate("r", "agent", "Go.", wait=door != "background", share_channels=share)
        )
    await asyncio.wait_for(closed.wait(), timeout=5.0)
    await asyncio.wait_for(asyncio.gather(call, return_exceptions=True), timeout=5.0)

    assert kit._closed


class _SlowCursor(InMemoryStore):
    """A store whose delivery cursor read takes a moment, as a networked
    store's does: a turn's rows still wait in the room's lane as its tool
    runs."""

    async def get_delivered_index(self, room_id: str) -> int:
        await asyncio.sleep(0.01)
        return await super().get_delivered_index(room_id)


_STORES: dict[str, Callable[[Any], ConversationStore]] = {
    # A kit's store by name, given the test's tmp_path.
    "memory": lambda tmp_path: InMemoryStore(),
    "slow-cursor": lambda tmp_path: _SlowCursor(),
    "sqlite": lambda tmp_path: SQLiteStore(tmp_path / "r.db"),
}


def _recording(store: type[ConversationStore]) -> type[ConversationStore]:
    """*store*, recording each row it commits, and whether the kit had
    released it then (its ``close()`` called: nothing is written after)."""

    class Recording(store):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any) -> None:
            super().__init__(*args)
            self.released = False
            self.commits: list[tuple[EventType, bool, Any]] = []

        async def commit_event(self, room_id: str, event: RoomEvent, **kw: Any) -> RoomEvent:
            stored = await super().commit_event(room_id, event, **kw)
            error = getattr(event.content, "error", None)
            self.commits.append((event.type, self.released, error))
            return stored

        async def close(self) -> None:
            self.released = True
            await super().close()

    return Recording


class _HoldsForEver(SimpleChannel):
    """A transport whose send of ``hold`` never ends: the room's lane busy."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.holding = asyncio.Event()

    async def deliver(self, event: RoomEvent, binding: Any, context: Any) -> Any:
        if getattr(event.content, "body", "") == "hold":
            self.holding.set()
            await asyncio.Event().wait()
        return await super().deliver(event, binding, context)


@pytest.mark.in_process(
    "the kit closes from a tool in its own turn: an executor elsewhere holds no kit"
)
@pytest.mark.parametrize("how", ["itself", "from-a-task"])
@pytest.mark.parametrize("lane", ["idle", "busy"])
@pytest.mark.parametrize("store", ["memory", "slow-cursor", "sqlite"])
async def test_a_tool_closing_the_kit_gets_its_end_row_before_the_close_returns(
    store: str,
    lane: str,
    how: str,
    tmp_path: Any,
    close_grace: float,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tool closes the kit from its own turn, or from a task it starts and
    awaits, its room's lane idle or busy: the close cuts that turn as any
    other, the open call's end row is stored before the kit releases its
    store, nothing is written after, no error is logged, and the call reads
    its cut, ``kit_closed``, never that cut as its error (RMK-1030)."""
    if store == "sqlite":
        recording: Any = _recording(SQLiteStore)(tmp_path / "r.db")
    else:
        recording = _recording(_SlowCursor if store == "slow-cursor" else InMemoryStore)()
    kit = RoomKit(store=recording)
    holds = _HoldsForEver("ws")
    ended = asyncio.Event()
    steps: list[str] = []
    held: list[asyncio.Future[Any]] = []

    async def shutdown(name: str, arguments: dict[str, Any]) -> str:
        if lane == "busy":
            message = InboundMessage(
                channel_id="sms", sender_id="u", content=TextContent(body="hold")
            )
            held.append(asyncio.ensure_future(kit.process_inbound(message, room_id="r")))
            await holds.holding.wait()
        try:
            await (kit.close() if how == "itself" else asyncio.create_task(kit.close()))
            steps.append("closed")
        except BaseException as error:
            steps.append(type(error).__name__)
            raise
        finally:
            ended.set()
        return "closed"

    tool = {"name": "shutdown", "description": "Shut down.", "parameters": {"type": "object"}}
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="shutdown", arguments={})],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ]
    )
    kit.register_channel(Agent("agent", provider=provider, tools=[tool], tool_handler=shutdown))
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(holds)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "ws")
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Go."))

    with caplog.at_level(logging.WARNING, logger="roomkit"):
        result = await asyncio.wait_for(kit.process_inbound(message, room_id="r"), 5)
        await asyncio.wait_for(ended.wait(), 1)
        await asyncio.wait_for(asyncio.gather(*held, return_exceptions=True), 1)
        for _ in range(20):  # what the tool's turn does once it resumed
            await asyncio.sleep(0)

    # The tool's own close returns; a tool whose task the close cut (one
    # awaiting a close it started) sees its cut.
    assert steps == ["closed"] or (how == "from-a-task" and steps == ["CancelledError"])
    assert (result.cancellation_reason, result.error) == ("kit_closed", None)
    ends = [row for row in recording.commits if row[0] == EventType.TOOL_CALL_END]
    assert ends == [(EventType.TOOL_CALL_END, False, "kit_closed")]
    assert [kind for kind, after, _ in recording.commits if after] == []
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []


async def _a_store(name: str, tmp_path: Any) -> ConversationStore:
    """A kit's store by name, as :data:`_STORES` makes it, or ``postgres``,
    which needs ``POSTGRES_DSN``: the test skips without it."""
    if name != "postgres":
        return _STORES[name](tmp_path)
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        pytest.skip("POSTGRES_DSN not set")
    from roomkit.store.postgres import PostgresStore  # the postgres extra

    store = PostgresStore(dsn=dsn)
    await store.init(min_size=1, max_size=2)
    return store


async def _a_room(kit: RoomKit, *targets: SimpleChannel) -> str:
    """A room of its own (a Postgres store is shared), its messages coming
    from ``sms`` and delivered to *targets*."""
    room_id = f"r-{uuid4().hex[:8]}"
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id=room_id)
    await kit.attach_channel(room_id, "sms")
    for target in targets:
        kit.register_channel(target)
        await kit.attach_channel(room_id, target.channel_id)
    return room_id


def _said(body: str) -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body=body))


@pytest.mark.parametrize("lane", ["idle", "busy"])
@pytest.mark.parametrize("store", ["memory", "slow-cursor", "sqlite", "postgres"])
async def test_an_observer_closing_the_kit_finds_a_set_placed_in_an_idle_lane_running(
    store: str, lane: str, tmp_path: Any, close_grace: float
) -> None:
    """An ASYNC BEFORE_BROADCAST observer closes the kit: it starts once its
    message's set has its place in the room's lane, whatever the store's
    commit takes, and placed in the idle lane, the set runs from its
    placement, untaken or not, and is delivered (RMK-1039); queued behind a
    send still under way, it is dropped, never sent. The call reads
    ``kit_closed`` either way."""
    kit = RoomKit(store=await _a_store(store, tmp_path))
    holds = _HoldsForEver("ws")
    room_id = await _a_room(kit, holds)

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.ASYNC, name="quit")
    async def quit(event: RoomEvent, ctx: RoomContext) -> None:
        if getattr(event.content, "body", "") == "Bye.":
            await kit.close()

    held: list[asyncio.Future[Any]] = []
    if lane == "busy":
        held.append(asyncio.ensure_future(kit.process_inbound(_said("hold"), room_id=room_id)))
        await holds.holding.wait()
    result = await asyncio.wait_for(kit.process_inbound(_said("Bye."), room_id=room_id), 1)
    await asyncio.wait_for(asyncio.gather(*held), 1)

    sent = {"ws": "sent"} if lane == "idle" else {}
    assert result.cancellation_reason == "kit_closed"
    assert {channel: r.status for channel, r in result.delivery_results.items()} == sent
    delivered = [event.content.body for event in holds.delivered]
    assert delivered == (["Bye."] if lane == "idle" else [])


@pytest.mark.parametrize("door", ["response", "after-broadcast"])
@pytest.mark.parametrize("store", ["memory", "sqlite", "postgres"])
async def test_an_answer_s_observer_closing_the_kit_finds_the_answer_running(
    store: str, door: str, tmp_path: Any, close_grace: float
) -> None:
    """An answer placed while the message it answers is past its sends,
    its AFTER_BROADCAST hooks and its responses' passes still running: a
    channel's response, or a ``send_event`` from an AFTER_BROADCAST hook.
    The lane counts as idle once the sends ahead have answered, so the
    answer's set runs from its placement, and an ASYNC BEFORE_BROADCAST
    observer of the answer that closes the kit finds it delivered, never
    dropped (RFC §12.10.4, RMK-1039)."""
    kit = RoomKit(store=await _a_store(store, tmp_path))
    ws = SimpleChannel("ws")
    room_id = await _a_room(kit, ws)
    if door == "response":
        kit.register_channel(AILikeChannel("ai", response="Hello."))
        await kit.attach_channel(room_id, "ai", category=ChannelCategory.INTELLIGENCE)
    else:

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="reply")
        async def reply(event: RoomEvent, ctx: RoomContext) -> None:
            if event.content.body == "Hi.":
                await kit.send_event(room_id, "sms", TextContent(body="Hello."))

    closed = asyncio.Event()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.ASYNC, name="quit")
    async def quit(event: RoomEvent, ctx: RoomContext) -> None:
        if event.content.body == "Hello.":
            await kit.close()
            closed.set()

    await asyncio.wait_for(kit.process_inbound(_said("Hi."), room_id=room_id), 1)
    await asyncio.wait_for(closed.wait(), 1)

    assert [event.content.body for event in ws.delivered] == ["Hi.", "Hello."]


@pytest.mark.parametrize("store", ["memory", "sqlite"])
async def test_a_close_from_the_lane_s_own_work_drops_what_it_kept_and_the_lane_ends(
    store: str, tmp_path: Any, close_grace: float, caplog: pytest.LogCaptureFixture
) -> None:
    """An AFTER_BROADCAST hook, which the room's lane runs as the
    aftermath of its message, sends an answer, placed in the idle lane,
    then closes the kit: the close waits for none of the lane's work, so
    it drops the answer it kept, and the lane ends with its round instead
    of running on over a released store (RFC §12.10.4, RMK-1039)."""
    kit = RoomKit(store=await _a_store(store, tmp_path))
    ws = SimpleChannel("ws")
    room_id = await _a_room(kit, ws)

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="reply")
    async def reply(event: RoomEvent, ctx: RoomContext) -> None:
        if event.content.body == "Hi.":
            await kit.send_event(room_id, "sms", TextContent(body="Hello."))
            await kit.close()

    with caplog.at_level(logging.WARNING, logger="roomkit"):
        await asyncio.wait_for(kit.process_inbound(_said("Hi."), room_id=room_id), 1)
        await asyncio.wait_for(kit.close(), 1)
        await asyncio.sleep(0.05)

    assert [event.content.body for event in ws.delivered] == ["Hi."]
    assert not [r for r in caplog.records if "failed a round" in r.getMessage()]


def _noting_commits(store: type[ConversationStore]) -> type[ConversationStore]:
    """*store*, its commit suspending, as a networked store's does, and
    noting each message it committed in ``steps``."""

    class Noting(store):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any) -> None:
            super().__init__(*args)
            self.steps: list[str] = []

        async def commit_event(self, room_id: str, event: RoomEvent, **kw: Any) -> RoomEvent:
            await asyncio.sleep(0.005)
            stored = await super().commit_event(room_id, event, **kw)
            if isinstance(event.content, TextContent):
                self.steps.append(f"committed {event.content.body}")
            return stored

    return Noting


async def _a_door(kit: RoomKit, room_id: str, door: str) -> None:
    """Send ``Hi.`` into *room_id* through *door*: a message, a direct send,
    a host's post, or a message whose answer or injected event takes a pass
    of its own (an agent's row, streamed or not, a regenerated answer, a
    channel's response, a hook's injection)."""
    if door == "send":
        await kit.send_event(room_id, "sms", TextContent(body="Hi."))
        return
    if door == "post":
        kit.register_channel(HostChannel("host"))
        await kit.attach_channel(room_id, "host")
        await kit.post(room_id, "Hi.")
        return
    if door in ("agent", "agent-stream", "regenerate", "response"):
        answers = (
            AILikeChannel("ai", response="Hello.")
            if door == "response"
            else Agent(
                "ai",
                provider=MockAIProvider(
                    responses=["Hello.", "Again."], streaming=door == "agent-stream"
                ),
            )
        )
        kit.register_channel(answers)
        await kit.attach_channel(room_id, "ai", category=ChannelCategory.INTELLIGENCE)
    if door == "injected":
        note = RoomEvent(
            room_id=room_id,
            source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
            content=TextContent(body="Note."),
        )

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="inject")
        async def inject(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.content.body != "Hi. (labelled)":
                return HookResult.allow()
            return HookResult.allow(
                injected=[InjectedEvent(event=note, target_channel_ids=["ws"])]
            )

    await kit.process_inbound(_said("Hi."), room_id=room_id)
    if door == "regenerate":
        await kit.regenerate_response(room_id)


@pytest.mark.parametrize(
    ("door", "store", "observed"),
    [
        ("inbound", "memory", ["Hi."]),
        ("inbound", "sqlite", ["Hi."]),
        ("send", "memory", ["Hi."]),
        ("post", "memory", ["Hi."]),
        ("agent", "memory", ["Hi.", "Hello."]),
        ("agent-stream", "memory", ["Hi.", "Hello."]),
        ("regenerate", "memory", ["Hi.", "Hello.", "Again."]),
        ("response", "memory", ["Hi.", "Hello."]),
        ("injected", "memory", ["Hi.", "Note."]),
    ],
    ids=[
        "inbound",
        "inbound-sqlite",
        "send",
        "post",
        "agent",
        "agent-stream",
        "regenerate",
        "response",
        "injected",
    ],
)
async def test_a_before_broadcast_observer_starts_once_its_pass_is_over(
    door: str, store: str, observed: list[str], tmp_path: Any
) -> None:
    """The ASYNC BEFORE_BROADCAST observers of a pass start once it is over,
    its event committed and its set placed in the room's lane, though the
    store's commit suspends, never during it, on every pass: a message, a
    direct send, a host's post, an agent's row, streamed or not, a
    regenerated answer, a channel's response, a hook's injected event. They
    observe the event as the SYNC hooks left it (RMK-1039)."""
    noting = _noting_commits(InMemoryStore if store == "memory" else SQLiteStore)
    recording: Any = noting() if store == "memory" else noting(tmp_path / "r.db")
    kit = RoomKit(store=recording)
    room_id = await _a_room(kit, SimpleChannel("ws"))

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="label")
    async def label(event: RoomEvent, ctx: RoomContext) -> HookResult:
        content = TextContent(body=f"{event.content.body} (labelled)")
        return HookResult.modify(event.model_copy(update={"content": content}))

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: RoomEvent, ctx: RoomContext) -> None:
        recording.steps.append(f"observed {event.content.body}")

    await _a_door(kit, room_id, door)
    await kit.close()

    steps = recording.steps
    seen = [step.removeprefix("observed ") for step in steps if step.startswith("observed ")]
    assert seen == [f"{body} (labelled)" for body in observed]
    for body in seen:
        assert steps.index(f"committed {body}") < steps.index(f"observed {body}"), steps


# -- a close from a delivery, a session's end, an announcement -----------------


class _Witness(SimpleChannel):
    """A channel that says whether it was closed."""

    closed = False

    async def close(self) -> None:
        self.closed = True


class _Closer:
    """The kit's own work closing the kit, once let go."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.go = asyncio.Event()
        self.closed = asyncio.Event()
        self.door: asyncio.Future[Any] | None = None
        """What opened the door, when it waits for the work: awaited last."""

    async def __call__(self, kit: RoomKit) -> None:
        if self.entered.is_set():
            return
        self.entered.set()
        await self.go.wait()
        await kit.close()
        self.closed.set()


async def _delivery_door(witness: _Witness, work: _Closer) -> RoomKit:
    """A delivery's ASYNC AFTER_BROADCAST hook closes the kit: the aftermath
    of the delivery, its close spares the room's lane."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(witness)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", witness.channel_id)

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="quit")
    async def quit(event: RoomEvent, ctx: RoomContext) -> None:
        await work(kit)

    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Bye."))
    # Its delivery waits for the hook, which may be held.
    work.door = asyncio.ensure_future(kit.process_inbound(message, room_id="r"))
    entered = asyncio.ensure_future(work.entered.wait())
    await asyncio.wait({work.door, entered}, return_when=asyncio.FIRST_COMPLETED)
    entered.cancel()
    return kit


async def _realtime_session_end_door(witness: _Witness, work: _Closer) -> RoomKit:
    """The client hangs up a realtime session; the ``voice_session_ended``
    handler, run in the session's teardown, closes the kit."""
    kit = RoomKit()
    transport = MockRealtimeTransport()
    channel = RealtimeVoiceChannel("rt", provider=MockRealtimeProvider(), transport=transport)
    kit.register_channel(channel)
    kit.register_channel(witness)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "rt")
    session = await channel.start_session("r", "p", "fake-ws")

    @kit.on("voice_session_ended")
    async def ended(event: Any) -> None:
        await work(kit)

    await transport.simulate_client_disconnect(session)
    return kit


async def _voice_session_end_door(witness: _Witness, work: _Closer) -> RoomKit:
    """The client hangs up a voice session; the ``voice_session_ended``
    handler, run in one of the voice channel's tasks, closes the kit."""
    backend = MockVoiceBackend()
    kit = RoomKit(voice=backend)
    kit.register_channel(VoiceChannel("voice", stt=MockSTTProvider(), backend=backend))
    kit.register_channel(witness)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "voice")
    session = await kit.join("r", "voice", participant_id="u")

    @kit.on("voice_session_ended")
    async def ended(event: Any) -> None:
        await work(kit)

    await backend.simulate_client_disconnected(session)
    return kit


async def _video_session_end_door(witness: _Witness, work: _Closer) -> RoomKit:
    """A video session is unbound; the ``video_session_ended`` handler, run
    in one of the video channel's tasks, closes the kit."""
    backend = MockVideoBackend()
    kit = RoomKit()
    video = VideoChannel("video", backend=backend)
    kit.register_channel(video)
    kit.register_channel(witness)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "video")
    session = await backend.connect("r", "u", "video")
    binding = ChannelBinding(room_id="r", channel_id="video", channel_type=ChannelType.VIDEO)
    video.bind_session(session, "r", binding)

    @kit.on("video_session_ended")
    async def ended(event: Any) -> None:
        await work(kit)

    video.unbind_session(session)
    return kit


async def _conference_started_door(witness: _Witness, work: _Closer) -> RoomKit:
    """The bot joins a conference as its first participant arrives; the
    ``conference_started`` handler, run inside that arrival's callback,
    closes the kit (RMK-1019)."""
    backend = MockConferenceBackend()
    kit = RoomKit()
    kit.register_channel(ConferenceChannel("conf", backend=backend, stt=MockSTTProvider()))
    kit.register_channel(witness)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "conf")

    @kit.on("conference_started")
    async def started(event: Any) -> None:
        await work(kit)

    # The arrival's callback waits for the handler, which may be held.
    work.door = asyncio.ensure_future(backend.simulate_participant_joined("r", "p-alice"))
    entered = asyncio.ensure_future(work.entered.wait())
    await asyncio.wait({work.door, entered}, return_when=asyncio.FIRST_COMPLETED)
    entered.cancel()
    return kit


async def _delegation_end_door(witness: _Witness, work: _Closer) -> RoomKit:
    """A background delegation ends; its ``on_complete``, run in the task's
    end, closes the kit."""
    kit = RoomKit()
    kit.register_channel(witness)
    kit.register_channel(Agent("worker", provider=MockAIProvider(responses=["Done."])))
    await kit.create_room(room_id="r")

    async def on_complete(result: Any) -> None:
        await work(kit)

    await kit.delegate("r", "worker", "Go.", on_complete=on_complete)
    return kit


async def _cancelled_delegation_end_door(witness: _Witness, work: _Closer) -> RoomKit:
    """A background delegation is cancelled; its ``on_complete``, run in the
    end the cancel took, closes the kit (RMK-1038)."""
    kit = RoomKit()
    kit.register_channel(witness)
    worker = _Endless()
    kit.register_channel(Agent("worker", provider=worker))
    await kit.create_room(room_id="r")

    async def on_complete(result: Any) -> None:
        await work(kit)

    task = await kit.delegate("r", "worker", "Go.", on_complete=on_complete)
    await worker.started.wait()
    # The cancel waits for the task's end, which may be held.
    work.door = asyncio.ensure_future(kit.cancel_task(task.id))
    entered = asyncio.ensure_future(work.entered.wait())
    await asyncio.wait({work.door, entered}, return_when=asyncio.FIRST_COMPLETED)
    entered.cancel()
    return kit


async def _held_write_door(witness: _Witness, work: _Closer) -> RoomKit:
    """A decided write a background run waits for (a strategy's or a
    delegation's, which its caller never cuts) closes the kit, from a hook
    it runs."""
    kit = RoomKit()
    kit.register_channel(witness)

    async def run() -> None:
        await run_to_its_end(work(kit), held=kit._decided_writes, name="write")

    hold_task(kit, run())
    return kit


async def _speak_queue_door(witness: _Witness, work: _Closer) -> RoomKit:
    """A discussion's turn ends; an ASYNC ON_SPEAK_QUEUE hook, run under the
    announcement of that change, closes the kit."""
    kit = RoomKit()
    kit.register_channel(witness)

    @kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def quit(event: SpeakQueueEvent, ctx: RoomContext) -> None:
        if event.change == SpeakQueueChange.TURN_ENDED:
            await work(kit)

    await _discussion(kit, {"a": _provider("Done.")})
    await _say(kit, "@a go")
    return kit


@pytest.mark.parametrize(
    "door",
    [
        _delivery_door,
        _realtime_session_end_door,
        _voice_session_end_door,
        _video_session_end_door,
        _speak_queue_door,
        _conference_started_door,
        _delegation_end_door,
        _cancelled_delegation_end_door,
        _held_write_door,
    ],
    ids=[
        "delivery-hook",
        "realtime-session-end",
        "voice-session-end",
        "video-session-end",
        "speak-queue-hook",
        "conference-started",
        "delegation-end",
        "cancelled-delegation-end",
        "held-write",
    ],
)
async def test_a_close_from_the_kit_s_own_work_returns_and_closes_the_kit(
    door: Callable[[_Witness, _Closer], Awaitable[RoomKit]], loop_keeps_turning: None
) -> None:
    """The kit's own work closes the kit: its close neither waits for nor
    cuts the work it runs in, and returns once every channel is closed
    (RMK-1015)."""
    witness = _Witness("witness")
    work = _Closer()
    work.go.set()

    # A realtime channel's attach compiles its marks' copies, ~1 s on its own.
    kit = await asyncio.wait_for(door(witness, work), 5)
    await asyncio.wait_for(work.closed.wait(), 1)

    assert witness.closed
    await kit.close()
    if work.door is not None:
        await asyncio.wait_for(work.door, 1)


@pytest.mark.parametrize(
    "door",
    [
        _delivery_door,
        _realtime_session_end_door,
        _speak_queue_door,
        _conference_started_door,
        _delegation_end_door,
        _cancelled_delegation_end_door,
        _held_write_door,
    ],
    ids=[
        "delivery-hook",
        "realtime-session-end",
        "speak-queue-hook",
        "conference-started",
        "delegation-end",
        "cancelled-delegation-end",
        "held-write",
    ],
)
async def test_a_close_from_the_kit_s_own_work_joining_another_spares_that_work(
    door: Callable[[_Witness, _Closer], Awaitable[RoomKit]], loop_keeps_turning: None
) -> None:
    """Another close() already runs, and waits for the work (the lane's
    drain, the session's teardown) when that work closes the kit too: from
    then on it is spared, and both calls return once the kit is closed
    (RMK-1015)."""
    witness = _Witness("witness")
    work = _Closer()
    kit = await asyncio.wait_for(door(witness, work), 5)
    await asyncio.wait_for(work.entered.wait(), 1)

    first = asyncio.ensure_future(kit.close())
    # The other close's shutdown reaches the work, and waits for it, first.
    for _ in range(5):
        await asyncio.sleep(0)
    work.go.set()

    await asyncio.wait_for(work.closed.wait(), 1)
    await asyncio.wait_for(first, 1)
    assert witness.closed
    if work.door is not None:
        await asyncio.wait_for(work.door, 1)


async def test_two_sessions_ending_at_once_each_close_the_kit() -> None:
    """Two realtime sessions' clients hang up together, and each one's
    ``voice_session_ended`` handler closes the kit: the second joins the
    shutdown the first started, which spares both (RMK-1015)."""
    kit = RoomKit()
    transport = MockRealtimeTransport()
    channel = RealtimeVoiceChannel("rt", provider=MockRealtimeProvider(), transport=transport)
    witness = _Witness("witness")
    kit.register_channel(channel)
    kit.register_channel(witness)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "rt")
    sessions = [await channel.start_session("r", f"p{n}", f"ws-{n}") for n in (1, 2)]
    returned: list[str] = []
    both = asyncio.Event()

    @kit.on("voice_session_ended")
    async def ended(event: Any) -> None:
        await kit.close()
        returned.append(event.room_id)
        if len(returned) == 2:
            both.set()

    await asyncio.wait_for(
        asyncio.gather(*(transport.simulate_client_disconnect(s) for s in sessions)), 1
    )
    await asyncio.wait_for(both.wait(), 1)
    assert witness.closed


async def _close_by(kit: RoomKit, how: str) -> None:
    """Close *kit* from the current task, or from a task it starts."""
    if how == "itself":
        await kit.close()
    else:
        await asyncio.create_task(kit.close())


async def _a_cut_tool_door(kit: RoomKit, handler: Any) -> Any:
    """An agent's turn whose tool runs *handler*: the call that runs it."""
    tool = {"name": "shutdown", "description": "Shut down.", "parameters": {"type": "object"}}
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="shutdown", arguments={})],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ]
    )
    kit.register_channel(Agent("agent", provider=provider, tools=[tool], tool_handler=handler))
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Go."))
    return asyncio.ensure_future(kit.process_inbound(message, room_id="r"))


async def _a_cut_realtime_door(kit: RoomKit, handler: Any) -> Any:
    """A realtime model's tool call running *handler*: the call that runs it."""
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt", provider=provider, transport=MockRealtimeTransport(), tool_handler=handler
    )
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "rt")
    session = await channel.start_session("r", "p", "fake-ws")
    return asyncio.ensure_future(provider.simulate_tool_call(session, "c1", "shutdown", {}))


@pytest.mark.in_process(
    "the kit closes from a tool in its own turn: an executor elsewhere holds no kit"
)
@pytest.mark.parametrize(
    ("door", "how"),
    [("agent-tool", "itself"), ("realtime-tool", "itself"), ("realtime-tool", "from-a-task")],
)
async def test_a_close_from_a_cut_tool_joins_the_close_that_cut_it(door: str, how: str) -> None:
    """A close() cuts a tool call (an agent's turn, a realtime model's call)
    while it waits, and the tool's cleanup closes the kit too, from its own
    task or from a task it starts: that close joins the running one, which
    neither waits for the tool nor cuts it again, and both return once the
    kit is closed (RMK-1019)."""
    kit = RoomKit()
    entered, ended = asyncio.Event(), asyncio.Event()
    steps: list[str] = []

    async def handler(*args: Any, **kwargs: Any) -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            steps.append("cut")
        try:
            await _close_by(kit, how)
            steps.append("closed")
        except asyncio.CancelledError:
            steps.append("its close cut")
            raise
        finally:
            ended.set()
        return "closed"

    witness = _Witness("witness")
    kit.register_channel(witness)
    opens = _a_cut_tool_door if door == "agent-tool" else _a_cut_realtime_door
    call = await opens(kit, handler)
    await asyncio.wait_for(entered.wait(), 5)

    await asyncio.wait_for(kit.close(), 1)
    await asyncio.wait_for(ended.wait(), 1)
    await asyncio.wait_for(asyncio.gather(call, return_exceptions=True), 1)

    assert steps == ["cut", "closed"]
    assert witness.closed


class _ClosingSend(SimpleChannel):
    """A transport whose send is the kit's own work closing the kit: it runs
    in the room's delivery lane."""

    def __init__(self, channel_id: str, work: _Closer) -> None:
        super().__init__(channel_id)
        self.kit: RoomKit | None = None
        self.work = work

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if self.kit is not None:
            await self.work(self.kit)
        return await super().deliver(event, binding, context)


async def test_the_deliveries_queued_behind_a_lane_a_close_runs_in_are_dropped() -> None:
    """A delivery's send closes the kit while the room's next message waits
    behind it in the lane: the lane is spared, not drained, so that message
    is never delivered, and its caller is not held (RMK-1015)."""
    witness = _Witness("witness")
    work = _Closer()
    sender = _ClosingSend("sender", work)
    kit = RoomKit()
    for channel in (SimpleChannel("sms"), witness, sender):
        kit.register_channel(channel)
    await kit.create_room(room_id="r")
    for channel_id in ("sms", "witness", "sender"):
        await kit.attach_channel("r", channel_id)
    sender.kit = kit
    bye = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Bye."))
    work.door = asyncio.ensure_future(kit.process_inbound(bye, room_id="r"))
    await asyncio.wait_for(work.entered.wait(), 1)
    message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Again."))
    queued = asyncio.ensure_future(kit.process_inbound(message, room_id="r"))
    # Queued behind the delivery whose hook runs.
    for _ in range(20):
        await asyncio.sleep(0)
    work.go.set()

    await asyncio.wait_for(work.closed.wait(), 1)
    await asyncio.wait_for(asyncio.gather(*(f for f in (work.door, queued) if f)), 1)
    assert [event.content.body for event in witness.delivered] == ["Bye."]


async def test_a_held_inline_run_starts_before_a_close_can_cut_it() -> None:
    """An inline task's run is never cancelled unstarted, its end never run:
    it starts as it is held."""
    kit = RoomKit()
    steps: list[str] = []

    async def work() -> None:
        steps.append("started")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            steps.append("ended")
            raise

    run = hold_task(kit, work(), eager=True)
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)

    assert steps == ["started", "ended"]


@pytest.mark.parametrize("wait", [True, False], ids=["inline", "background"])
async def test_a_task_announced_as_the_kit_closes_ends_cancelled(
    wait: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kit's close lands between the task's announcement and its run: it
    ends cancelled on both doors, its worker never runs (RFC §23.3)."""
    started = asyncio.Event()
    kit = RoomKit()
    provider = _SlowWords(started)
    kit.register_channel(Agent("worker", provider=provider))
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    announce = kit._announce_task

    async def announce_then_close(handle: Any) -> None:
        await announce(handle)
        await kit.close()

    monkeypatch.setattr(kit, "_announce_task", announce_then_close)
    task = await asyncio.wait_for(kit.delegate("r", "worker", "Go.", wait=wait), timeout=5.0)

    assert task.result is not None and task.result.status == "cancelled"
    assert provider.calls == []


# -- what the kit holds as it closes --------------------------------------------


async def test_a_background_run_started_in_the_close_s_grace_ends_with_its_close() -> None:
    """An observer still running as the kit closes starts work the kit holds
    (a hand-off greeting's turn) in the close's grace: it is cut with the
    kit's other background work, never left running past close()."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    closing = asyncio.Event()
    runs: list[asyncio.Task[None]] = []

    async def greeting() -> None:
        await asyncio.Event().wait()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.ASYNC, name="greet")
    async def greet(event: RoomEvent, ctx: RoomContext) -> None:
        await closing.wait()
        runs.append(hold_task(kit, greeting()))

    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body="hi")),
        room_id="r1",
    )
    closed = asyncio.ensure_future(kit.close())
    closing.set()
    await closed

    assert [run.done() for run in runs] == [True]


class _HoldsAsItCloses(SimpleChannel):
    """A channel whose close starts work the kit holds (a hand-back its
    session's end starts)."""

    def __init__(self, channel_id: str, hold: Callable[[], None]) -> None:
        super().__init__(channel_id)
        self._hold = hold

    async def close(self) -> None:
        self._hold()
        await super().close()


@pytest.mark.parametrize(
    "when", ["before-its-first-step", "as-a-channel-closes", "after-the-close"]
)
async def test_a_run_held_around_the_close_is_cut_with_its_end_run(when: str) -> None:
    """Work the kit holds (a strategy's background run, a hand-back, a
    greeting's turn) started however late in the kit's close, or after it,
    is cut as close cuts the runs it holds: once it started, so its own end
    runs (a run's terminal entry, a task's completion), and never left
    running past close() (RFC §19.7.3, §23.3)."""
    kit = RoomKit()
    steps: list[str] = []
    runs: list[asyncio.Task[None]] = []

    async def work() -> None:
        steps.append("started")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            steps.append("ended")
            raise

    def hold() -> None:
        runs.append(hold_task(kit, work()))

    if when == "as-a-channel-closes":
        kit.register_channel(_HoldsAsItCloses("sms", hold))
    if when == "before-its-first-step":
        hold()
    await kit.close()
    if when == "after-the-close":
        hold()
        await asyncio.wait(runs, timeout=1.0)

    assert [run.cancelled() for run in runs] == [True]
    assert steps == ["started", "ended"]


class _TracesAsItCloses(SimpleChannel):
    """A channel whose close emits a protocol trace (a SIP BYE), and returns
    once the trace's hook has started."""

    def __init__(self, channel_id: str, hook_started: asyncio.Event) -> None:
        super().__init__(channel_id)
        self._hook_started = hook_started

    async def close(self) -> None:
        self.emit_trace(
            ProtocolTrace(
                channel_id=self.channel_id,
                direction="outbound",
                protocol="sip",
                summary="BYE",
                room_id="r1",
            )
        )
        await asyncio.wait_for(self._hook_started.wait(), 1.0)
        await super().close()


class _SlowRecords(InMemoryStore):
    """A store whose commit of a ``record`` takes a few steps, and that says
    how many were still in flight when it closed."""

    def __init__(self) -> None:
        super().__init__()
        self.in_flight = 0
        self.in_flight_at_close: list[int] = []

    async def commit_event(self, room_id: str, event: RoomEvent, **kw: Any) -> RoomEvent:
        if getattr(event.content, "body", None) != "record":
            return await super().commit_event(room_id, event, **kw)
        self.in_flight += 1
        try:
            for _ in range(5):
                await asyncio.sleep(0)
            return await super().commit_event(room_id, event, **kw)
        finally:
            self.in_flight -= 1

    async def close(self) -> None:
        self.in_flight_at_close.append(self.in_flight)
        await super().close()


def _record() -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="sip", channel_type="sms"),
        content=TextContent(body="record"),
    )


@pytest.mark.parametrize("its_end", ["ends", "holds-a-run", "commits-unawaited"])
async def test_a_hook_task_started_as_a_channel_closes_ends_with_the_close(its_end: str) -> None:
    """A channel's close starts a hook of the kit's (its last trace's
    ON_PROTOCOL_TRACE): the hook is cut before close() returns, its end run,
    and what its end starts (a run the kit holds, a record it commits
    without awaiting it) ends before the store is released."""
    store = _SlowRecords()
    kit = RoomKit(store=store)
    hook_started = asyncio.Event()
    kit.register_channel(_TracesAsItCloses("sip", hook_started))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sip")
    steps: list[str] = []

    async def late_run() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)  # its end: a terminal entry it posts
            steps.append("run ended")
            raise

    @kit.hook(HookTrigger.ON_PROTOCOL_TRACE, execution=HookExecution.ASYNC, name="audit")
    async def audit(trace: ProtocolTrace, ctx: RoomContext) -> None:
        steps.append("started")
        hook_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            steps.append("ended")
            if its_end == "holds-a-run":
                hold_task(kit, late_run())
            if its_end == "commits-unawaited":
                asyncio.get_running_loop().create_task(kit.commit_event("r1", _record()))
            raise

    await kit.close()

    assert steps == ["started", "ended", *(["run ended"] if its_end == "holds-a-run" else [])]
    assert store.in_flight_at_close == [0]


async def test_a_caller_waiting_for_its_injected_events_returns_when_the_close_cuts_them(
    close_grace: float,
) -> None:
    """process_inbound committed its event and waits for the pass of what
    its hook injected when the kit closes and cuts that pass: the caller
    returns its committed event, never a CancelledError it did not ask for."""
    kit = RoomKit()
    for channel_id in ("sms1", "sms2"):
        kit.register_channel(SimpleChannel(channel_id))
    await kit.create_room(room_id="r1")
    for channel_id in ("sms1", "sms2"):
        await kit.attach_channel("r1", channel_id)
    in_pass = asyncio.Event()

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="gate")
    async def gate(event: RoomEvent, ctx: RoomContext) -> HookResult:
        body = getattr(event.content, "body", "")
        if body == "injected":
            in_pass.set()
            await asyncio.Event().wait()
        if body != "hello":
            return HookResult.allow()
        injected = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="sms2", channel_type="sms"),
            content=TextContent(body="injected"),
        )
        return HookResult(
            action="allow",
            injected_events=[InjectedEvent(event=injected, target_channel_ids=["sms2"])],
        )

    message = InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="hello"))
    caller = asyncio.create_task(kit.process_inbound(message, room_id="r1"))
    await asyncio.wait_for(in_pass.wait(), 1)
    await kit.close()

    result = await asyncio.wait_for(caller, 1)
    assert result.event is not None and result.event.content == TextContent(body="hello")


async def test_a_timer_check_whose_hooks_outlast_the_grace_is_cut(close_grace: float) -> None:
    """check_room_timers() paused the room and waits for its ON_ROOM_PAUSED
    hook when the kit closes: a room change is running work, so once the
    grace ran out the check is cut, KitClosedError, never a CancelledError
    its caller did not ask for; the room stays as stored (RMK-1030)."""
    kit = RoomKit()
    await kit.create_room(room_id="r1", timers=RoomTimers(inactive_after_seconds=0))
    in_hook = asyncio.Event()

    @kit.hook(HookTrigger.ON_ROOM_PAUSED, name="slow")
    async def slow(event: RoomEvent, ctx: RoomContext) -> None:
        in_hook.set()
        await asyncio.Event().wait()

    check = asyncio.create_task(kit.check_room_timers("r1"))
    await asyncio.wait_for(in_hook.wait(), 1)
    await kit.close()

    with pytest.raises(KitClosedError):
        await asyncio.wait_for(check, 1)
    room = await kit.get_room("r1")
    assert room.status is RoomStatus.PAUSED


async def test_a_result_post_held_just_before_the_close_posts_nothing() -> None:
    """A task's result posted as its agent's message, in work the kit holds
    (a discussion's post turn), is held just before the kit closes: cut once
    started, it posts nothing on the closing kit (RFC §19.7.5 rule 19)."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(Agent("ai1", provider=MockAIProvider(responses=["hello"])))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    result = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="ai1", channel_type="ai"),
        content=TextContent(body="result"),
    )
    hold_task(kit, send_posted(kit, result))
    await kit.close()

    assert "result" not in [
        getattr(e.content, "body", None) for e in await kit.store.list_events("r1")
    ]
