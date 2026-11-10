from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.core import framework
from roomkit.models import Task
from roomkit.store.memory import InMemoryStore
from tests.test_kit_close import _closing, _kit, _message, _Transport


async def test_nonempty_reads_remain_available_after_the_memory_store_closes() -> None:
    """The memory store stays usable; owned SQLite stores instead refuse reads."""
    kit = await _kit()
    task = Task(id="kept", room_id="r", title="Follow up")
    await kit.store.add_task(task)
    await kit.submit_feedback("r", 0.75, comment="Kept")
    observations = await kit.list_observations("r")
    assert len(observations) == 1
    assert observations[0].content == "[overall] 0.75: Kept"
    assert await kit.list_tasks("r") == [task]
    await kit.close()
    assert await kit.list_tasks("r") == [task]
    assert await kit.list_observations("r") == observations


class HeldCursor(InMemoryStore):
    """Hold the cursor read itself, before the active place sends anything."""

    def __init__(self) -> None:
        super().__init__()
        self.reading = asyncio.Event()
        self.release = asyncio.Event()

    async def get_delivered_index(self, room_id: str) -> int:
        self.reading.set()
        await self.release.wait()
        return await super().get_delivered_index(room_id)


@pytest.mark.parametrize("busy", [False, True])
async def test_a_place_held_on_its_cursor_gets_grace_but_its_neighbor_does_not(
    busy: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RFC §12.10.4: placement, before a slow cursor read, owns the grace."""
    monkeypatch.setattr(framework, "CLOSE_GRACE_S", 30.0)
    release = asyncio.Event()
    target = _Transport("ws", holds={"held": release})
    store = HeldCursor()
    kit = await _kit(target, store=store)
    skipped: list[Any] = []
    kit.on("delivery_skipped")(lambda event: skipped.append(event))
    first = await asyncio.wait_for(
        kit.process_inbound(
            _message("held" if busy else "placed"), room_id="r", defer_delivery=True
        ),
        1,
    )
    assert first.delivery is not None
    await asyncio.wait_for(store.reading.wait(), 1)
    assert target.sent == []
    queued = (
        await kit.process_inbound(_message("placed"), room_id="r", defer_delivery=True)
        if busy
        else None
    )
    close = await _closing(kit)
    if queued is not None:
        assert queued.delivery is not None
        result = await asyncio.wait_for(queued.delivery.wait(), 1)
        assert (result.cancellation_reason, result.delivery_results) == ("kit_closed", {})
    store.release.set()
    if busy:
        await asyncio.wait_for(target.entered("held").wait(), 1)
        release.set()
    await asyncio.wait_for(close, 1)
    result = await first.delivery.wait()
    assert result.cancellation_reason == "kit_closed"
    assert {key: value.status for key, value in result.delivery_results.items()} == {"ws": "sent"}
    assert target.sent == (["held"] if busy else ["placed"])
    assert skipped == []
    assert first.event is not None
    assert await store.get_delivered_index("r") == first.event.index


async def test_three_unfinished_calls_keep_their_sent_failed_and_queued_results(
    close_grace: float,
) -> None:
    """Queue the batch before releasing its first send, so its round is live."""
    release = asyncio.Event()
    target = _Transport("ws", holds={"m0": release, "m1": None})
    kit = await _kit(target)
    handles = []
    for body in ["m0", "m1", "m2"]:
        result = await kit.process_inbound(_message(body), room_id="r", defer_delivery=True)
        assert result.delivery is not None
        handles.append(result.delivery)
    await asyncio.wait_for(target.entered("m0").wait(), 1)
    release.set()
    await asyncio.wait_for(target.entered("m1").wait(), 1)
    assert target.sent == ["m0"]
    first_wait = asyncio.create_task(handles[0].wait())
    await asyncio.sleep(0)
    assert not first_wait.done()
    await asyncio.wait_for(kit.close(), 1)
    results = [await first_wait, await handles[1].wait(), await handles[2].wait()]
    assert [result.cancellation_reason for result in results] == ["kit_closed"] * 3
    assert [
        {key: value.status for key, value in result.delivery_results.items()} for result in results
    ] == [{"ws": "sent"}, {"ws": "failed"}, {}]
    assert target.sent == ["m0"]
