"""What a turn's stream costs and how its cleanup is accounted (RFC §10.6.4,
§10.6.8): a cancelled cascade reports cleanup done only once each turn it read
ended on its executor, and no unread remote stream is left to the garbage
collector: a delivery that stops between its turns and their reader closes
them itself, whatever still holds its result."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import os
import time
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

import pytest

from roomkit import HookResult, HookTrigger, RoomKit
from roomkit.core._room_turns import QueuedTurn, RoomTurns
from roomkit.core.event_router import StreamingResponse
from roomkit.core.exceptions import TurnLinkError
from roomkit.core.lanes import DeliveryCascade, ExecEntry, RoomDeliveryLane
from roomkit.core.task_utils import cancel_reason
from roomkit.models.channel import ChannelOutput
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import TextContent
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.turns import InMemoryTurnBackend, TurnClient, TurnExecutor, TurnTimeouts
from roomkit.turns._executor_turn import ExecutorTurn
from roomkit.turns.base import TurnBackend
from roomkit.turns.owner import registry_turn_client
from roomkit.turns.redis import RedisTurnBackend
from roomkit.turns.wire import Cancel, Close
from tests.redis_keys import delete_keys, key_prefix
from tests.test_framework import AILikeChannel, SimpleChannel
from tests.test_turns_backend import ScriptedChannel, _binding, _context, _event, _run

QUICK = TurnTimeouts(cancel=0.05)


def _client(channel: ScriptedChannel, timeouts: TurnTimeouts) -> TurnClient:
    channels = {channel.channel_id: channel}
    return TurnClient(
        InMemoryTurnBackend(),
        executor=TurnExecutor(channels.get, hosts=channels.__contains__, timeouts=timeouts),
        timeouts=timeouts,
    )


def _response(stream: Any) -> StreamingResponse:
    return StreamingResponse(
        stream=stream,
        source_channel_id="agent",
        source_channel_type=ChannelType.AI,
        trigger_event=_event(),
    )


@pytest.mark.parametrize("waited", [False, True], ids=["side by side", "waited its place"])
async def test_a_cancelled_cascade_reports_its_cleanup_incomplete_while_an_end_is_owed(
    waited: bool,
) -> None:
    """The executor's stream hangs in its cleanup, so the turn's end never
    comes: the owner's drain gives up at its bound, and the cascade, which
    read that turn, says its cleanup is incomplete rather than done, a turn
    that waited its place in the room included (RFC §6.4)."""
    let_go = asyncio.Event()

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        try:
            yield "one"
            await asyncio.Event().wait()
        finally:
            await let_go.wait()

    client = _client(ScriptedChannel(script), QUICK)
    if waited:
        place = RoomTurns(_run, None, None).join("room", "agent")
        stream: Any = QueuedTurn(place, partial(_run, client), ResponseMetadata())
    else:
        stream = (await _run(client)).response_stream
    assert await anext(stream) == "one"
    cascade = DeliveryCascade("room", reentry_budget=10)
    cascade.add_streams([_response(stream)])

    async def consume() -> None:
        async for _ in stream:
            pass

    cascade.track(asyncio.ensure_future(consume()), turn_work=True)
    await asyncio.sleep(0.05)

    with pytest.raises(TimeoutError, match="did not end"):
        await cascade.cancel_and_wait("caller_cancelled", timeout=2.0)
    let_go.set()
    await client.close()


async def _unending(channel: ScriptedChannel) -> AsyncIterator[Any]:
    yield "one"
    await asyncio.Event().wait()


def _deaf_to_its_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the executor deaf to a turn's close and cancel, as one stuck in a
    blocking call: the turn's end never comes."""
    apply_control = ExecutorTurn._apply_control

    async def deaf(turn: ExecutorTurn, message: Any) -> None:
        if not isinstance(message, Close | Cancel):
            await apply_control(turn, message)

    monkeypatch.setattr(ExecutorTurn, "_apply_control", deaf)


@pytest.mark.parametrize("stuck", [True, False], ids=["executor stuck", "its executor ends it"])
async def test_a_cancelled_cascade_waits_for_the_end_of_a_stream_it_never_read(
    stuck: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn answered with a stream nobody read: the cancelled cascade closes
    it and waits for its end, with no bound of its own (RFC §10.1, §10.6.8).
    An executor that never reads the close, stuck in a blocking call, leaves
    the cleanup incomplete at the waiter's timeout; one that ends the turn
    lets it return."""
    if stuck:
        _deaf_to_its_end(monkeypatch)
    client = _client(ScriptedChannel(_unending), QUICK)
    stream = (await _run(client)).response_stream
    cascade = DeliveryCascade("room", reentry_budget=10)
    cascade.add_streams([_response(stream)])

    if stuck:
        with pytest.raises(TimeoutError, match="Delivery cleanup incomplete"):
            await cascade.cancel_and_wait("caller_cancelled", timeout=0.05)
    else:
        await cascade.cancel_and_wait("caller_cancelled", timeout=2.0)
    await client.close()


async def test_a_cancelled_cascade_waits_for_no_stream_it_is_handed_once_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream a delivery set hands the cascade once it is cancelled is
    dropped unread: the cleanup closes it without waiting for its end (RFC
    §10.1), so an executor that never ends the turn holds no waiter."""
    _deaf_to_its_end(monkeypatch)
    client = _client(ScriptedChannel(_unending), QUICK)
    stream = (await _run(client)).response_stream
    cascade = DeliveryCascade("room", reentry_budget=10)
    cascade.cancel("caller_cancelled")
    cascade.add_streams([_response(stream)])

    await cascade.cancel_and_wait("caller_cancelled", timeout=0.05)
    await client.close()


async def test_a_cancelled_cascade_whose_turns_ended_reports_its_cleanup_done() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "one"
        await asyncio.Event().wait()

    client = _client(ScriptedChannel(script), QUICK)
    output = await _run(client)
    stream = output.response_stream
    assert await anext(stream) == "one"
    cascade = DeliveryCascade("room", reentry_budget=10)
    cascade.add_streams([_response(stream)])

    async def consume() -> None:
        async for _ in stream:
            pass

    cascade.track(asyncio.ensure_future(consume()), turn_work=True)
    await asyncio.sleep(0.05)

    await cascade.cancel_and_wait("caller_cancelled", timeout=2.0)
    await client.close()


async def _consumed(stream: Any) -> None:
    with contextlib.suppress(TurnLinkError):
        async for _ in stream:
            pass


async def test_a_cancelled_cascade_nobody_waits_for_leaves_no_unretrieved_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cancelled without a waiter (a dropped lane entry), its cleanup that
    finds an end owed is no task error left for the garbage collector."""
    let_go = asyncio.Event()

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        try:
            yield "one"
            await asyncio.Event().wait()
        finally:
            await let_go.wait()

    client = _client(ScriptedChannel(script), QUICK)
    stream = (await _run(client)).response_stream
    assert await anext(stream) == "one"
    cascade = DeliveryCascade("room", reentry_budget=10)
    cascade.add_streams([_response(stream)])
    cascade.track(asyncio.ensure_future(_consumed(stream)), turn_work=True)
    await asyncio.sleep(0.05)

    with caplog.at_level(logging.ERROR, logger="asyncio"):
        cascade.cancel("kit_closed")
        await asyncio.sleep(3 * QUICK.cancel)  # past the cancel bound
        del cascade, stream
        gc.collect()
        await asyncio.sleep(0.01)
    let_go.set()
    await client.close()

    assert [r.getMessage() for r in caplog.records if r.name == "asyncio"] == []


@pytest.fixture(params=["memory", "redis"])
async def owner_and_host(
    request: pytest.FixtureRequest,
) -> AsyncIterator[tuple[TurnBackend, TurnBackend]]:
    """The owner's turn backend and the one its turns' executor is served on:
    one in memory; across Redis, two, as two processes, so the owner closing
    is the executor's owner going away and never its own backend closing."""
    if request.param == "memory":
        backend = InMemoryTurnBackend()
        yield backend, backend
        return
    if not os.environ.get("REDIS_URL"):
        pytest.skip("REDIS_URL not set")
    pytest.importorskip("redis.asyncio")
    prefix = key_prefix()
    owner, host = (
        RedisTurnBackend(os.environ["REDIS_URL"], key_prefix=prefix, timeouts=QUICK)
        for _ in range(2)
    )
    yield owner, host
    await owner.close()
    await host.close()
    await delete_keys(prefix)


async def test_a_cascade_whose_turn_its_owner_gave_up_reports_its_cleanup_done(
    owner_and_host: tuple[TurnBackend, TurnBackend],
) -> None:
    """The owner closing gives up the turns it holds, waiting for nothing of
    them (a close from inside a turn spares that turn): no end is owed, and
    the executor cuts the turn as one whose owner was lost (RFC §10.6.8)."""
    let_go = asyncio.Event()
    cut: list[str | None] = []

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        try:
            yield "one"
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancelled:
            cut.append(cancel_reason(cancelled))
            raise
        finally:
            await let_go.wait()

    channels = {"agent": ScriptedChannel(script)}
    owner, host = owner_and_host
    # Declared as a collection: a host of another process advertises it.
    executor = TurnExecutor(channels.get, hosts=set(channels), timeouts=QUICK)
    await host.serve(executor)  # a host elsewhere beats at once
    client = TurnClient(owner, timeouts=QUICK)
    stream = (await _run(client)).response_stream
    assert await anext(stream) == "one"
    cascade = DeliveryCascade("room", reentry_budget=10)
    cascade.add_streams([_response(stream)])
    cascade.track(asyncio.ensure_future(_consumed(stream)), turn_work=True)

    await client.close()
    await cascade.cancel_and_wait("kit_closed", timeout=2.0)
    async with asyncio.timeout(5.0):
        while not cut:
            await asyncio.sleep(0.02)
    let_go.set()
    await executor.close()

    assert cut == ["owner_lost"]


async def test_a_turn_given_up_as_its_owner_closes_fails_backend_closed(
    turn_backend: TurnBackend,
) -> None:
    """A turn still open when its owner closes is given up: its reader fails
    with ``backend_closed``, the owner's backend having closed, never with
    ``executor_lost``, its executor being alive (RFC §10.6, §10.6.8; RMK-878)."""
    let_go = asyncio.Event()

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        try:
            yield "one"
            await asyncio.Event().wait()
        finally:
            await let_go.wait()

    channels = {"agent": ScriptedChannel(script)}
    executor = TurnExecutor(channels.get, hosts=channels.__contains__)
    await turn_backend.serve(executor)
    client = TurnClient(turn_backend)
    stream = (await _run(client)).response_stream
    assert await anext(stream) == "one"

    await client.close()

    with pytest.raises(TurnLinkError) as raised:
        await anext(stream)
    let_go.set()
    await executor.close()
    assert raised.value.reason == "backend_closed"


async def test_a_turn_whose_end_is_owed_stays_owed_when_its_owner_closes() -> None:
    """The owner closing gives up the turns still open, never one already
    over: a turn whose drain passed ``cancel_timeout`` keeps its end owed, so
    a cascade still cleaning up reports it incomplete (RFC §10.1, §10.6.8)."""
    let_go = asyncio.Event()

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        try:
            yield "one"
            await asyncio.Event().wait()
        finally:
            await let_go.wait()

    async def consume(stream: Any) -> None:
        with contextlib.suppress(TurnLinkError):
            async for _ in stream:
                pass

    async def slow_to_stop() -> None:
        # More of the cascade's turn work, slow to honour the cut: its
        # cleanup reads the turn's end after the owner closed.
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(3 * QUICK.cancel)  # past the owner's close below
            raise

    channels = {"agent": ScriptedChannel(script)}
    backend = InMemoryTurnBackend()
    executor = TurnExecutor(channels.get, hosts=channels.__contains__, timeouts=QUICK)
    await backend.serve(executor)
    client = TurnClient(backend, timeouts=QUICK)
    stream = (await _run(client)).response_stream
    assert await anext(stream) == "one"
    cascade = DeliveryCascade("room", reentry_budget=10)
    cascade.add_streams([_response(stream)])
    cascade.track(asyncio.ensure_future(consume(stream)), turn_work=True)
    cascade.track(asyncio.ensure_future(slow_to_stop()), turn_work=True)
    await asyncio.sleep(0.05)
    cleanup = asyncio.ensure_future(cascade.cancel_and_wait("caller_cancelled", timeout=3.0))
    await asyncio.sleep(2 * QUICK.cancel)  # past cancel_timeout: the turn is over, its end owed

    await client.close()

    with pytest.raises(TimeoutError):
        await cleanup
    let_go.set()
    await executor.close()


@pytest.mark.in_process("measures the in-memory link")
@pytest.mark.wallclock
async def test_two_thousand_deltas_cost_the_owner_s_loop_little() -> None:
    """A turn's text deltas cross the in-memory backend at a cost close to
    reading them directly (RFC §10.6.4): a pull grants many of them.

    Measured against a direct read on the same machine, so a slower one
    reads the same: about 12 times a direct read with a pull's window,
    about 240 times when each delta crossed on its own."""

    async def deltas() -> AsyncIterator[str]:
        for i in range(2000):
            yield f"tok{i} "

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        async for delta in deltas():
            yield delta

    client = registry_turn_client({"agent": ScriptedChannel(script)})
    expected = "".join(f"tok{i} " for i in range(2000))
    costs, direct = [], []
    for _ in range(5):
        output = await client.run(_event(), _binding(), _context())
        started = time.process_time()
        assert "".join([item async for item in output.response_stream]) == expected
        costs.append(time.process_time() - started)
        started = time.process_time()
        assert "".join([delta async for delta in deltas()]) == expected
        direct.append(time.process_time() - started)
    await client.close()

    assert min(costs) < 50 * min(direct), (costs, direct)


# -- a delivery that stops between its turns and their reader -------------------


async def _endless(channel: ScriptedChannel) -> AsyncIterator[Any]:
    yield "never read"
    await asyncio.Event().wait()


def _turn_tasks() -> list[asyncio.Task[Any]]:
    return [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and t.get_name().startswith("roomkit-turn-")
    ]


async def _turns_ended() -> bool:
    for _ in range(100):
        if not _turn_tasks():
            return True
        await asyncio.sleep(0.01)
    return False


async def _room_kit() -> RoomKit:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("tx"))
    kit.register_channel(ScriptedChannel(_endless))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    return kit


def _message() -> InboundMessage:
    return InboundMessage(channel_id="tx", sender_id="u", content=TextContent(body="Go."))


async def test_a_delivery_cancelled_once_its_plan_ran_closes_the_turns_it_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = await _room_kit()
    held: list[Any] = []
    execute = RoomDeliveryLane._execute

    async def cancelled_after(lane: RoomDeliveryLane, entry: ExecEntry) -> Any:
        result = await execute(lane, entry)
        held.append(result)  # something still holds the result
        # The caller cancels once the plan ran, before its effects.
        asyncio.get_running_loop().call_soon(entry.cascade.cancel, "caller_cancelled")
        return result

    monkeypatch.setattr(RoomDeliveryLane, "_execute", cancelled_after)
    result = await kit.process_inbound(_message(), defer_delivery=True)
    assert result.delivery is not None
    await asyncio.wait_for(result.delivery.wait(), timeout=5.0)

    assert held and held[0].streaming_responses
    assert await _turns_ended()
    await kit.close()


async def test_a_root_pass_that_fails_closes_the_turns_its_caller_never_gets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = await _room_kit()

    async def fails(*args: Any) -> None:
        raise RuntimeError("reporting failed")

    monkeypatch.setattr(kit, "_report_root_pass", fails)
    # The failure is kept with its traceback, which holds the result.
    result = await kit.process_inbound(_message())

    assert isinstance(result.error, RuntimeError)
    assert await _turns_ended()
    await kit.close()


async def test_a_delegated_turn_whose_commit_fails_closes_the_turns_it_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = await _room_kit()

    async def fails(*args: Any) -> None:
        raise RuntimeError("commit failed")

    monkeypatch.setattr(kit, "_commit_blocked_events", fails)
    task = await kit.delegate("r", "agent", "Go.", wait=True)

    assert task.result is not None
    assert await _turns_ended()
    await kit.close()


async def test_a_shared_delegated_turn_whose_commit_fails_closes_the_turns_it_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a transport shared, the answer takes the room's path: a commit
    that fails there leaves no stream to the garbage collector either."""
    kit = await _room_kit()

    async def fails(*args: Any) -> None:
        raise RuntimeError("commit failed")

    monkeypatch.setattr(kit, "_commit_responses", fails)
    task = await kit.delegate("r", "agent", "Go.", wait=True, share_channels=["tx"])

    assert task.result is not None
    assert await _turns_ended()
    await kit.close()


# -- A cut's unread streams: who waits for their close (RFC §10.1) -----------


async def _ending(channel: ScriptedChannel) -> AsyncIterator[Any]:
    yield "Hi"


class _Agent(ScriptedChannel):
    """An agent whose stream ends; it answers no agent's answer."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(_ending)
        self.channel_id = channel_id

    async def on_event(self, event: Any, binding: Any, context: Any) -> Any:
        if event.source.channel_type == ChannelType.AI:
            return ChannelOutput.empty()
        return await super().on_event(event, binding, context)


class _FlushingTarget(SimpleChannel):
    """A streaming target that reads its stream to its end, then works on
    past it."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.past_the_end = asyncio.Event()

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self, stream: AsyncIterator[Any], event: Any, binding: Any, context: Any
    ) -> Any:
        async for _ in stream:
            pass
        self.past_the_end.set()
        await asyncio.sleep(0.1)
        return ChannelOutput.empty()


async def _two_answers() -> tuple[RoomKit, _FlushingTarget]:
    """A room whose two agents answer with a stream that ends, one of them
    read by a target that flushes past its end, the other not read yet."""
    kit = RoomKit()
    target = _FlushingTarget("ws")
    kit.register_channel(SimpleChannel("tx"))
    kit.register_channel(target)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    await kit.attach_channel("r", "ws")
    for channel_id in ("agent", "agent2"):
        kit.register_channel(_Agent(channel_id))
        await kit.attach_channel("r", channel_id, category=ChannelCategory.INTELLIGENCE)
    return kit, target


def _keyed() -> InboundMessage:
    return InboundMessage(
        channel_id="tx", sender_id="u", content=TextContent(body="Go."), delivery_key="k"
    )


async def _cut_by_its_key(kit: RoomKit, call: asyncio.Task[Any], stuck: bool) -> None:
    """Cut *call* by its key, then close *kit*: stuck, the call has not
    returned a while after the cut, and the close ends it; else it returns,
    cancelled."""
    assert await kit.cancel_delivery("k") == 1
    done, _ = await asyncio.wait({call}, timeout=0.05 if stuck else 2.0)
    await kit.close()
    if stuck:
        assert not done
        with contextlib.suppress(Exception):
            await asyncio.wait_for(call, timeout=2.0)
    else:
        assert call.result().cancellation_reason == "caller_cancelled"


@pytest.mark.usefixtures("close_grace", "cut_unwind")
@pytest.mark.parametrize("stuck", [True, False], ids=["executor stuck", "its executor ends it"])
async def test_an_awaited_call_cut_waits_for_the_end_of_a_stream_it_never_read(
    stuck: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An awaited call reads its streams in a task whose ``finally`` closes
    every stream of its cascade, each waiting for its end: cut by its key
    while its target flushes past one stream, the other never read, it
    waits for that close (RFC §10.1)."""
    if stuck:
        _deaf_to_its_end(monkeypatch)
    kit, target = await _two_answers()
    call = asyncio.create_task(kit.process_inbound(_keyed()))
    await asyncio.wait_for(target.past_the_end.wait(), 2.0)

    await _cut_by_its_key(kit, call, stuck)


class _HoldingAnswers(SimpleChannel):
    """A transport whose delivery of the answering agent's answer never ends."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.holding = asyncio.Event()

    async def deliver(self, event: Any, binding: Any, context: Any) -> Any:
        if event.source.channel_id == "answering":
            self.holding.set()
            await asyncio.Event().wait()
        return await super().deliver(event, binding, context)


@pytest.mark.usefixtures("close_grace", "cut_unwind")
@pytest.mark.parametrize("stuck", [True, False], ids=["executor stuck", "its executor ends it"])
async def test_an_awaited_call_cut_while_an_answer_s_set_runs_waits_for_its_unread_stream(
    stuck: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cut before its streams are read, an agent's answer still delivered and
    a streaming agent's stream handed to its cascade: the awaited call joins
    the cleanup, which closes that stream and waits for its end (RFC §10.1)."""
    if stuck:
        _deaf_to_its_end(monkeypatch)
    kit = RoomKit()
    holding = _HoldingAnswers("tx")
    kit.register_channel(holding)
    kit.register_channel(ScriptedChannel(_unending))
    kit.register_channel(AILikeChannel("answering", "Done."))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("r", "answering", category=ChannelCategory.INTELLIGENCE)
    call = asyncio.create_task(kit.process_inbound(_keyed()))
    await asyncio.wait_for(holding.holding.wait(), 2.0)

    await _cut_by_its_key(kit, call, stuck)


@pytest.mark.usefixtures("close_grace", "cut_unwind")
async def test_a_handle_whose_unread_stream_is_closing_is_not_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past a cancel that timed out, its cleanup still closing a stream the
    call never read: the handle is not done, a second cancel times out at
    its own timeout, and wait() waits on (RFC §10.1, §10.6.8)."""
    _deaf_to_its_end(monkeypatch)
    kit, target = await _two_answers()
    result = await kit.process_inbound(
        InboundMessage(channel_id="tx", sender_id="u", content=TextContent(body="Go.")),
        defer_delivery=True,
    )
    handle = result.delivery
    assert handle is not None
    await asyncio.wait_for(target.past_the_end.wait(), 2.0)
    with pytest.raises(TimeoutError):
        await handle.cancel(timeout=0.05)
    loop = asyncio.get_running_loop()

    done = handle.done
    start = loop.time()
    with pytest.raises(TimeoutError):
        await handle.cancel(timeout=0.05)
    second = loop.time() - start
    waited = asyncio.ensure_future(handle.wait())
    finished, _ = await asyncio.wait({waited}, timeout=0.05)
    waited.cancel()
    await kit.close()

    assert not done
    assert 0.04 < second < 0.5
    assert not finished


@pytest.mark.usefixtures("close_grace", "cut_unwind")
@pytest.mark.parametrize("stuck", [True, False], ids=["executor stuck", "its executor ends it"])
async def test_an_adopted_call_s_handle_waits_for_the_end_of_a_stream_it_never_read(
    stuck: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deferred call an off-lock hook's body makes is adopted beside the
    event with a cascade of its own: cut by its handle while its target
    flushes past one stream, the other never read, its cancel closes that
    stream and waits for its end, as any deferred call's does (RFC §9.5.1,
    §10.1). Its executor deaf to the close, the cancel times out."""
    if stuck:
        _deaf_to_its_end(monkeypatch)
    kit, target = await _two_answers()
    cancelled: list[str | None] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, needs_lock=False, name="adopts")
    async def adopts(event: Any, ctx: Any) -> HookResult:
        if event.content != TextContent(body="Go."):
            return HookResult.allow()
        side = await kit.process_inbound(
            InboundMessage(channel_id="tx", sender_id="u", content=TextContent(body="side")),
            room_id="r",
            defer_delivery=True,
        )
        assert side.delivery is not None
        await asyncio.wait_for(target.past_the_end.wait(), 2.0)
        try:
            # Stuck, any bound times out: the shortest keeps the test quick.
            ended = await side.delivery.cancel(timeout=0.01 if stuck else 2.0)
            cancelled.append(ended.cancellation_reason)
        except TimeoutError:
            cancelled.append("timed out")
        # Nothing more to deliver: the test is the side call's.
        return HookResult.block("adopted")

    await asyncio.wait_for(kit.process_inbound(_message(), room_id="r"), 5.0)
    await kit.close()

    assert cancelled == ["timed out" if stuck else "caller_cancelled"]


class _ReadingTarget(SimpleChannel):
    """A streaming target that reads its stream, told at its first item."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.reading = asyncio.Event()

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self, stream: AsyncIterator[Any], event: Any, binding: Any, context: Any
    ) -> Any:
        async for _ in stream:
            self.reading.set()
        return ChannelOutput.empty()


@pytest.mark.usefixtures("close_grace", "cut_unwind")
@pytest.mark.parametrize("stuck", [True, False], ids=["executor stuck", "its executor ends it"])
async def test_an_adopted_call_whose_read_turn_s_end_is_owed_times_out(
    stuck: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deferred call an off-lock hook's body makes is adopted beside the
    event with a cascade of its own: cut by its handle while its target reads
    its agent's stream, its executor deaf to the cut, the turn is over for its
    owner at its cancel bound without its executor's end, and the cancel
    raises TimeoutError well within its own timeout, as any deferred call's
    does (RFC §9.5.1, §10.1, §10.6.8). Its executor ending the turn, the
    cancel returns."""
    if stuck:
        _deaf_to_its_end(monkeypatch)
    kit, target = await _reading_room(monkeypatch)
    handles = _keeping_its_read(kit, target, cancel=True)

    await asyncio.wait_for(kit.process_inbound(_message(), room_id="r"), 5.0)
    handle, cancelled = handles[0]
    waits = [await _outcome(handle.wait()) for _ in range(2)]
    await kit.close()

    owed = "Delivery cleanup incomplete for room r: the turns of ['agent'] did not end"
    ended = f"{owed} on their executor" if stuck else "caller_cancelled"
    assert (cancelled, waits) == (ended, [ended if stuck else "caller_cancelled"] * 2)


async def _reading_room(monkeypatch: pytest.MonkeyPatch) -> tuple[RoomKit, _ReadingTarget]:
    """A room whose agent's stream never ends, read by a streaming target,
    its owner giving a cut turn's end up at its 50 ms cancel bound."""
    kit = RoomKit()
    monkeypatch.setattr(kit._turns, "_timeouts", QUICK)
    target = _ReadingTarget("ws")
    kit.register_channel(SimpleChannel("tx"))
    kit.register_channel(target)
    kit.register_channel(ScriptedChannel(_unending))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    await kit.attach_channel("r", "ws")
    await kit.attach_channel("r", "agent", category=ChannelCategory.INTELLIGENCE)
    return kit, target


async def _outcome(waiting: Any) -> str | None:
    """A wait's or a cancel's end: its cancellation reason, or its timeout's
    words."""
    try:
        return (await waiting).cancellation_reason
    except TimeoutError as error:
        return str(error)


def _keeping_its_read(
    kit: RoomKit, target: _ReadingTarget, *, cancel: bool
) -> list[tuple[Any, str | None]]:
    """An off-lock hook that, on "Go.", makes a deferred call adopted beside
    it, keeps its handle once *target* reads its stream, cancels it when
    *cancel* (what the cancel came to kept beside it), and blocks "Go.":
    nothing more is delivered."""
    kept: list[tuple[Any, str | None]] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, needs_lock=False, name="adopts")
    async def adopts(event: Any, ctx: Any) -> HookResult:
        if event.content != TextContent(body="Go."):
            return HookResult.allow()
        side = await kit.process_inbound(
            InboundMessage(channel_id="tx", sender_id="u", content=TextContent(body="side")),
            room_id="r",
            defer_delivery=True,
        )
        assert side.delivery is not None
        await asyncio.wait_for(target.reading.wait(), 2.0)
        cancelled = await _outcome(side.delivery.cancel(timeout=2.0)) if cancel else None
        kept.append((side.delivery, cancelled))
        return HookResult.block("adopted")

    return kept


@pytest.mark.usefixtures("close_grace", "cut_unwind")
async def test_a_wait_past_the_kit_s_close_on_an_adopted_call_whose_turn_s_end_is_owed_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kit's close cancels every cascade in flight, an adopted call's own
    among them (core/_closing.py): a turn its target read when the close cut
    it, its end owed, fails every wait on its handle with TimeoutError, as a
    deferred call's (RFC §10.6.8, §12.10.4)."""
    _deaf_to_its_end(monkeypatch)
    kit, target = await _reading_room(monkeypatch)
    handles = _keeping_its_read(kit, target, cancel=False)

    await asyncio.wait_for(kit.process_inbound(_message(), room_id="r"), 5.0)
    await kit.close()
    handle, _ = handles[0]
    waits = [await _outcome(handle.wait()) for _ in range(2)]

    owed = "Delivery cleanup incomplete for room r: the turns of ['agent'] did not end"
    assert waits == [f"{owed} on their executor"] * 2
