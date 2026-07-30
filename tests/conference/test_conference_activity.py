"""The drain that keeps a conference teardown behind the work it contradicts.

Three properties, and the third is the one that makes the other two safe to
have: a drain waits for in-flight work, a drain gives up rather than hanging on
work that never finishes, and a drain never waits for work it is running inside.
"""

from __future__ import annotations

import asyncio
import time

from roomkit.channels._conference_activity import RoomActivity

ROOM = "room-1"


class TestDraining:
    async def test_a_drain_waits_for_work_in_flight(self) -> None:
        activity = RoomActivity()
        order: list[str] = []
        gate = asyncio.Event()

        async def work() -> None:
            async with activity.track(ROOM):
                await gate.wait()
                order.append("work")

        working = asyncio.create_task(work())
        await asyncio.sleep(0)

        async def teardown() -> None:
            await activity.drain(ROOM)
            order.append("teardown")

        tearing = asyncio.create_task(teardown())
        await asyncio.sleep(0.01)
        assert order == []

        gate.set()
        await asyncio.gather(working, tearing)

        assert order == ["work", "teardown"]

    async def test_a_drain_with_nothing_in_flight_returns(self) -> None:
        activity = RoomActivity()

        await asyncio.wait_for(activity.drain(ROOM), timeout=1.0)

    async def test_a_drain_ignores_another_room(self) -> None:
        activity = RoomActivity()
        gate = asyncio.Event()

        async def work() -> None:
            async with activity.track("room-2"):
                await gate.wait()

        working = asyncio.create_task(work())
        await asyncio.sleep(0)

        await asyncio.wait_for(activity.drain(ROOM), timeout=1.0)

        gate.set()
        await working

    async def test_a_drain_gives_up_on_work_that_never_finishes(self) -> None:
        """Teardown that waits forever on a wedged backend is a worse failure
        than teardown that overtakes it.
        """
        activity = RoomActivity()
        gate = asyncio.Event()

        async def wedged() -> None:
            async with activity.track(ROOM):
                await gate.wait()

        working = asyncio.create_task(wedged())
        await asyncio.sleep(0)

        await asyncio.wait_for(activity.drain(ROOM, timeout=0.05), timeout=1.0)

        gate.set()
        await working

    async def test_the_timeout_is_a_budget_for_the_whole_drain(self) -> None:
        """Per item, seven wedged activities at five seconds each is
        thirty-five — past the thirty the hook engine cancels a lifecycle hook
        at, so the drain would be the reason that ceiling is reached.
        """
        activity = RoomActivity()
        gate = asyncio.Event()

        async def wedged() -> None:
            async with activity.track(ROOM):
                await gate.wait()

        working = [asyncio.create_task(wedged()) for _ in range(5)]
        await asyncio.sleep(0)

        started = time.monotonic()
        await activity.drain(ROOM, timeout=0.05)
        elapsed = time.monotonic() - started

        assert elapsed < 0.2, f"five wedged activities took {elapsed:.3f}s on a 0.05s budget"

        gate.set()
        await asyncio.gather(*working)

    async def test_drain_all_shares_one_budget_across_rooms(self) -> None:
        activity = RoomActivity()
        gate = asyncio.Event()

        async def wedged(room_id: str) -> None:
            async with activity.track(room_id):
                await gate.wait()

        working = [asyncio.create_task(wedged(f"room-{i}")) for i in range(5)]
        await asyncio.sleep(0)

        started = time.monotonic()
        await activity.drain_all(timeout=0.05)
        elapsed = time.monotonic() - started

        assert elapsed < 0.2, f"five wedged rooms took {elapsed:.3f}s on a 0.05s budget"

        gate.set()
        await asyncio.gather(*working)


class TestDeferring:
    """What a re-entrant teardown does instead of waiting."""

    async def test_enclosing_names_the_work_the_caller_is_inside(self) -> None:
        activity = RoomActivity()
        found: list[int] = []

        async def work() -> None:
            async with activity.track(ROOM):
                found.append(len(activity.enclosing(ROOM)))

        await work()

        assert found == [1]

    async def test_enclosing_is_empty_outside_the_work(self) -> None:
        activity = RoomActivity()
        gate = asyncio.Event()

        async def work() -> None:
            async with activity.track(ROOM):
                await gate.wait()

        working = asyncio.create_task(work())
        await asyncio.sleep(0)

        assert activity.enclosing(ROOM) == []

        gate.set()
        await working

    async def test_waiting_on_the_enclosing_signals_resolves_when_it_ends(self) -> None:
        """The deferred teardown's whole mechanism: leave the block, then wait
        on the signals it handed out.
        """
        activity = RoomActivity()
        order: list[str] = []
        signals: list[asyncio.Event] = []

        async def work() -> None:
            async with activity.track(ROOM):
                signals.extend(activity.enclosing(ROOM))
            order.append("work-done")

        async def deferred() -> None:
            while not signals:
                await asyncio.sleep(0)
            await activity.wait_for(signals, timeout=1.0)
            order.append("teardown")

        await asyncio.gather(work(), deferred())

        assert order == ["work-done", "teardown"]


class TestReentrancy:
    """A drain must not wait for work it is running inside.

    An integrator whose ``conference_started`` handler detaches the channel is
    writing ordinary code; a drain that waited for the announcement carrying
    that handler would deadlock on it.
    """

    async def test_a_drain_inside_the_work_does_not_wait_for_it(self) -> None:
        activity = RoomActivity()

        async def work() -> None:
            async with activity.track(ROOM):
                await activity.drain(ROOM)

        await asyncio.wait_for(work(), timeout=1.0)

    async def test_the_marker_reaches_a_task_spawned_from_the_work(self) -> None:
        """The hook engine dispatches lifecycle hooks onto tasks of their own,
        so task identity alone would not recognise the re-entrant case — the
        marker has to follow the context a task inherits.
        """
        activity = RoomActivity()

        async def teardown() -> None:
            await activity.drain(ROOM)

        async def work() -> None:
            async with activity.track(ROOM):
                await asyncio.create_task(teardown())

        await asyncio.wait_for(work(), timeout=1.0)

    async def test_a_drain_outside_the_work_still_waits(self) -> None:
        """The escape is for nesting, not a licence to skip the wait."""
        activity = RoomActivity()
        released: list[str] = []
        gate = asyncio.Event()

        async def work() -> None:
            async with activity.track(ROOM):
                await gate.wait()
                released.append("work")

        working = asyncio.create_task(work())
        await asyncio.sleep(0)

        async def teardown() -> None:
            await activity.drain(ROOM, timeout=1.0)
            released.append("teardown")

        tearing = asyncio.create_task(teardown())
        await asyncio.sleep(0.01)
        gate.set()
        await asyncio.gather(working, tearing)

        assert released == ["work", "teardown"]

    async def test_nested_work_is_released_in_order(self) -> None:
        """Leaving an inner block must not drop the outer block's marker."""
        activity = RoomActivity()

        async def work() -> None:
            async with activity.track(ROOM):
                async with activity.track(ROOM):
                    pass
                # Still inside the outer block: a drain here must not wait for it.
                await activity.drain(ROOM)

        await asyncio.wait_for(work(), timeout=1.0)
