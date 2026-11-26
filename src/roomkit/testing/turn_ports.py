"""What a broker and a registry give a turn backend across processes,
checked on any implementation of the two ports: the library's own, in this
process, and a service's, by its own tests.

Each check raises ``AssertionError``, saying what, where the contract is
not kept. It writes under names of its own (the nodes ``n1``, ``n2`` and
``waiting``, the agent ``agent``): give it a broker or a registry nothing
else writes to, as a namespace of the test's own on a shared server.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from roomkit.turns.broker import Beat, BrokerMessage, InboxEntry, TurnBroker, TurnRegistry

_BLOCK = 0.02
# How far a registry's clock may stray from this machine's: a server's
# clock is another machine's.
_CLOCK_SKEW_MS = 3_600_000


async def check_turn_broker(broker: TurnBroker) -> None:
    """Checks that *broker* keeps the contract of
    :class:`~roomkit.turns.broker.TurnBroker`: an inbox is read in the order
    it was posted in, the messages of one post in theirs, after any entry,
    until acknowledged; a node reads its own inbox alone; a read that waits
    takes a message posted meanwhile, and waits no longer than its block.

    Raises:
        AssertionError: Where *broker* does not keep the contract, saying what.
    """
    await _an_inbox_is_read_in_the_order_posted_until_acknowledged(broker)
    await _a_waiting_read_takes_a_message_posted_meanwhile(broker)


async def check_turn_registry(registry: TurnRegistry) -> None:
    """Checks that *registry* keeps the contract of
    :class:`~roomkit.turns.broker.TurnRegistry`: a beat is written at the
    registry's time, the time of day, and viewed until withdrawn or
    forgotten, a beat older
    than its node's last one changes nothing; a room has one holder until
    forgotten; a presence is claimed once a change.

    Raises:
        AssertionError: Where *registry* does not keep the contract, saying what.
    """
    await _a_beat_is_viewed_until_withdrawn(registry)
    await _a_room_has_one_holder_until_forgotten(registry)
    await _a_presence_is_claimed_once_a_change(registry)


def _same(read: Any, expected: Any, what: str) -> None:
    if read != expected:
        raise AssertionError(f"{what}: read {read!r}, expected {expected!r}")


def _frame(body: str) -> BrokerMessage:
    return BrokerMessage("frame", "sender", body, "turn", "")


def _bodies(entries: list[InboxEntry]) -> list[str | None]:
    return [entry.message.body if entry.message is not None else None for entry in entries]


async def _an_inbox_is_read_in_the_order_posted_until_acknowledged(broker: TurnBroker) -> None:
    await broker.post("n1", [_frame("one"), _frame("two")])
    await broker.post("n2", [_frame("another node's")])
    inbox = await broker.inbox("n1", _BLOCK)
    entries = await inbox.read("")
    _same(_bodies(entries), ["one", "two"], "an inbox read from its start, in the order posted")
    ids = [entry.id for entry in entries]

    # After the last entry nothing is read, and the read waits no longer
    # than its block; after the first, the second again.
    _same(await inbox.read(ids[1]), [], "a read after the last entry")
    _same(_bodies(await inbox.read(ids[0])), ["two"], "a read after the first entry")

    await broker.post("n1", [_frame("three")])
    _same(_bodies(await inbox.read(ids[1])), ["three"], "a read after the last entry read")
    await inbox.close()

    # A reader that starts over reads what was not acknowledged.
    await broker.acknowledge("n1", ids)
    again = await broker.inbox("n1", _BLOCK)
    _same(_bodies(await again.read("")), ["three"], "a reader that starts over")
    await again.close()


async def _a_waiting_read_takes_a_message_posted_meanwhile(broker: TurnBroker) -> None:
    inbox = await broker.inbox("waiting", 5.0)
    reading = asyncio.ensure_future(inbox.read(""))
    try:
        await asyncio.sleep(_BLOCK)
        await broker.post("waiting", [_frame("late")])
        try:
            entries = await asyncio.wait_for(reading, 2.0)
        except TimeoutError:
            raise AssertionError("the read waited its whole block") from None
    finally:
        reading.cancel()
        await asyncio.gather(reading, return_exceptions=True)
        await inbox.close()
    _same(_bodies(entries), ["late"], "a read that waited")


def _beat(generation: int, **fields: Any) -> Beat:
    return Beat(node="n1", generation=generation, **fields)


async def _a_beat_is_viewed_until_withdrawn(registry: TurnRegistry) -> None:
    await registry.connect()
    _same(registry.connected(), True, "a registry reached")
    await registry.beat(
        _beat(
            2,
            declared={"n1:a": "its declaration"},
            undeclared=["n1:b"],
            running=["turn-1"],
            owner=True,
        )
    )
    view = await registry.view(60_000)
    written, declared = view.executors.get("n1:a", (None, None))
    _same(declared, "its declaration", "the declaration of an executor declared")
    _same(view.executors.get("n1:b", (None, "none"))[1], None, "an executor that declares nothing")
    if written is None or view.now < written:
        raise AssertionError("a beat is written at the registry's time")
    if abs(view.now - time.time() * 1000) >= _CLOCK_SKEW_MS:
        raise AssertionError(
            "the registry's time is the time of day, in milliseconds since 1970: "
            f"it reads {view.now}"
        )
    _same("n1" in view.owners, True, "a room owner that beats")

    # A beat older than the node's last one changes nothing.
    await registry.beat(_beat(1, gone=["n1:a"]))
    view = await registry.view(60_000)
    _same("n1:a" in view.executors, True, "an executor after an older beat left it gone")

    # A node's last beat and the turns it runs are read together.
    running = [("n1", ["turn-1", "turn-2"])]
    beats = await registry.beats(["n1:a", "n1:unknown"], running)
    _same(beats.last, [written, None], "the last beats asked")
    _same(beats.running, [[True, False]], "the turns a node runs")

    # The turns a node runs are read with no executor's beat asked, and a
    # beat with no turn.
    beats = await registry.beats([], running)
    _same((beats.last, beats.running), ([], [[True, False]]), "beats with no member asked")
    beats = await registry.beats(["n1:a"], [("n1", [])])
    _same((beats.last, beats.running), ([written], [[]]), "beats with no turn asked")

    # The heartbeat's own writer writes a beat as the registry does.
    beater = await registry.beater()
    await beater.beat(_beat(3, undeclared=["n1:b"], gone=["n1:a"], owner=True))
    await beater.close()
    view = await registry.view(60_000)
    _same(sorted(view.executors), ["n1:b"], "the executors after one is gone")
    beats = await registry.beats(["n1:b"], running)
    _same(beats.running, [[False, False]], "the turns of a node's last beat")

    await registry.forget_owner("n1")
    _same((await registry.view(60_000)).owners, {}, "the owners once one is forgotten")

    # A view forgets what last beat too long ago, for the next ones too.
    _same((await registry.view(0)).executors, {}, "a view that forgets every beat")
    _same((await registry.view(60_000)).executors, {}, "the view after one that forgot")


async def _a_room_has_one_holder_until_forgotten(registry: TurnRegistry) -> None:
    _same(await registry.holder("agent", "room"), None, "a room nobody holds")
    await registry.hold("agent", "room", "n1:a")
    await registry.hold("agent", "room", "n2:b")
    _same(await registry.holder("agent", "room"), "n2:b", "a room's last holder")

    # A channel and a room are told apart whatever their ids hold.
    await registry.hold("a:b", "c", "n1:a")
    _same(await registry.holder("a", "b:c"), None, "a room whose ids read as another's")

    await registry.forget_holder("agent", "room")
    _same(await registry.holder("agent", "room"), None, "a holder forgotten")


async def _a_presence_is_claimed_once_a_change(registry: TurnRegistry) -> None:
    async def claim(online: bool | None) -> bool:
        return await registry.claim_presence("agent", "room", online)

    _same(await claim(True), True, "a first presence")
    _same(await claim(True), False, "a presence already so")
    _same(await claim(False), True, "a presence that changes")
    _same(await claim(None), False, "forgetting a presence is no change")
    _same(await claim(False), True, "a change again, once forgotten")
    _same(
        await registry.claim_presence("agent", "another room", False),
        True,
        "the presence in another room",
    )
