"""Discussion regeneration keeps its selection across close and uninstall."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.core import framework
from roomkit.core.locks import InMemoryLockManager
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundResult
from roomkit.models.event import RoomEvent
from roomkit.models.room import Room
from roomkit.orchestration.strategies.discussion import SpeakQueueEvent
from tests.orchestration.test_discussion_edges import _room, _say
from tests.orchestration.test_strategy_discussion import _provider, _settle


@pytest.fixture
async def kit(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[RoomKit]:
    monkeypatch.setattr(framework, "CLOSE_GRACE_S", 0.02)
    kit = RoomKit()
    try:
        async with asyncio.timeout(5):
            await _room(kit, {"a": _provider("first", "second")})
            await _say(kit, "@a hello")
            await _settle(kit)
        yield kit
    finally:
        await asyncio.wait_for(kit.close(), 5)


def _assert_selected(result: InboundResult | None, trigger: RoomEvent | None) -> None:
    assert trigger is not None
    assert result is not None and result.event is not None
    assert result.event.id == trigger.id
    assert not result.blocked
    assert result.reason is None and result.error is None
    assert result.cancellation_reason is None
    assert result.response_events == []


async def _cancel_if_running(task: asyncio.Task[InboundResult | None]) -> None:
    if not task.done():
        task.cancel()
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)


async def test_close_cuts_a_discussion_regenerations_initial_read(
    kit: RoomKit, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before selection, a closing kit cuts the call and its held store read."""
    entered, cancelled = asyncio.Event(), asyncio.Event()
    original = kit.store.get_room
    regeneration: asyncio.Task[InboundResult | None]

    async def held(room_id: str) -> Room | None:
        if asyncio.current_task() is regeneration:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return await original(room_id)

    monkeypatch.setattr(kit.store, "get_room", held)
    regeneration = asyncio.create_task(kit.regenerate_response("r1"))
    try:
        async with asyncio.timeout(5):
            await entered.wait()
            await kit.close()
            result = await regeneration
        assert cancelled.is_set()
        assert result is not None and result.blocked
        assert result.reason == "kit_closed" and result.event is None
        assert result.cancellation_reason == "kit_closed"
    finally:
        await _cancel_if_running(regeneration)


async def test_close_spares_a_discussion_regeneration_after_selection(
    kit: RoomKit, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent close ends without cutting the selected call's fresh read."""
    trigger = await kit.regenerate_target("r1")
    entered, opened, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = kit._stored_discussion_over

    async def held(room_id: str) -> bool:
        entered.set()
        try:
            await opened.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return await original(room_id)

    monkeypatch.setattr(kit, "_stored_discussion_over", held)
    regeneration = asyncio.create_task(kit.regenerate_response("r1"))
    try:
        async with asyncio.timeout(5):
            await entered.wait()
            await kit.close()
            assert not regeneration.done() and not cancelled.is_set()
            opened.set()
            _assert_selected(await regeneration, trigger)
    finally:
        opened.set()
        await _cancel_if_running(regeneration)


async def test_uninstall_after_regeneration_selection_keeps_the_selected_trigger(
    kit: RoomKit, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selected Discussion can leave the registry without failing the call."""
    trigger = await kit.regenerate_target("r1")
    original = kit._stored_discussion_over

    async def uninstall(room_id: str) -> bool:
        await kit.uninstall_strategy(room_id)
        assert room_id not in kit._discussions
        return await original(room_id)

    monkeypatch.setattr(kit, "_stored_discussion_over", uninstall)
    async with asyncio.timeout(5):
        _assert_selected(await kit.regenerate_response("r1"), trigger)


async def test_an_uninstall_in_another_kit_announces_no_regenerated_turn(
    kit: RoomKit, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing a queue removed elsewhere announces no turn, even with a live owner."""
    trigger = await kit.regenerate_target("r1")
    other = RoomKit(store=kit.store, lock_manager=kit._lock_manager)
    original = kit._stored_discussion_over
    changes: list[SpeakQueueEvent] = []

    @kit.hook(HookTrigger.ON_SPEAK_QUEUE, execution=HookExecution.ASYNC)
    async def observe(event: SpeakQueueEvent, context: RoomContext) -> None:
        changes.append(event)

    async def uninstall(room_id: str) -> bool:
        over = await original(room_id)
        await other.uninstall_strategy(room_id)
        assert not kit._discussions[room_id].closed
        return over

    monkeypatch.setattr(kit, "_stored_discussion_over", uninstall)
    try:
        async with asyncio.timeout(5):
            _assert_selected(await kit.regenerate_response("r1"), trigger)
            await kit.hook_engine.drain_observers()
        assert changes == []
    finally:
        await asyncio.wait_for(other.close(), 5)


class _FailingReleases(InMemoryLockManager):
    """Fail selected releases of the caller, after actually releasing its lock."""

    def __init__(self, failures: tuple[int, ...]) -> None:
        super().__init__()
        self.errors = {number: RuntimeError(f"release {number} failed") for number in failures}
        self.caller: asyncio.Task[object] | None = None
        self.releases = 0

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncIterator[None]:
        async with super().locked(room_id):
            yield
        if self.caller is not None and asyncio.current_task() is self.caller:
            self.releases += 1
            if error := self.errors.get(self.releases):
                raise error


@pytest.mark.parametrize("failures", [(1,), (2,), (1, 2)], ids=["selection", "queue", "both"])
async def test_regeneration_reports_the_first_failed_release_without_losing_its_turn(
    failures: tuple[int, ...],
) -> None:
    """Either lock may fail to release; the first failure wins and the answer still runs."""
    locks = _FailingReleases(failures)
    kit = RoomKit(lock_manager=locks)
    provider = _provider("first", "second")
    try:
        async with asyncio.timeout(5):
            await _room(kit, {"a": provider})
            await _say(kit, "@a hello")
            await _settle(kit)
            trigger = await kit.regenerate_target("r1")
            locks.caller = asyncio.current_task()
            result = await kit.regenerate_response("r1")
            locks.caller = None
            await _settle(kit)
        assert trigger is not None and result is not None and result.event is not None
        assert result.event.id == trigger.id and not result.blocked
        assert result.error is locks.errors[min(failures)]
        assert locks.releases == 2
        assert len(provider.calls) == 2
    finally:
        locks.caller = None
        await asyncio.wait_for(kit.close(), 5)
