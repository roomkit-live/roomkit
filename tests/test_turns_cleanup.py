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

from roomkit import RoomKit
from roomkit.core._room_turns import QueuedTurn, RoomTurns
from roomkit.core.event_router import StreamingResponse
from roomkit.core.exceptions import TurnLinkError
from roomkit.core.lanes import DeliveryCascade, ExecEntry, RoomDeliveryLane
from roomkit.core.task_utils import cancel_reason
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
from tests.test_framework import SimpleChannel
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


@pytest.mark.parametrize("stuck", [True, False], ids=["executor stuck", "its executor ends it"])
async def test_a_cancelled_cascade_waits_for_the_end_of_a_stream_it_never_read(
    stuck: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn answered with a stream nobody read: the cancelled cascade closes
    it and waits for its end, with no bound of its own (RFC §10.1, §10.6.8).
    An executor that never reads the close, stuck in a blocking call, leaves
    the cleanup incomplete at the waiter's timeout; one that ends the turn
    lets it return."""

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "one"
        await asyncio.Event().wait()

    if stuck:
        apply_control = ExecutorTurn._apply_control

        async def deaf_to_its_end(turn: ExecutorTurn, message: Any) -> None:
            if not isinstance(message, Close | Cancel):
                await apply_control(turn, message)

        monkeypatch.setattr(ExecutorTurn, "_apply_control", deaf_to_its_end)
    client = _client(ScriptedChannel(script), QUICK)
    stream = (await _run(client)).response_stream
    cascade = DeliveryCascade("room", reentry_budget=10)
    cascade.add_streams([_response(stream)])

    if stuck:
        with pytest.raises(TimeoutError, match="Delivery cleanup incomplete"):
            await cascade.cancel_and_wait("caller_cancelled", timeout=0.05)
    else:
        await cascade.cancel_and_wait("caller_cancelled", timeout=2.0)
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
