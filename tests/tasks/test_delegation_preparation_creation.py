"""An in-flight creation settles before delegation cleans its child (RFC §23.3)."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from roomkit import Agent, RoomKit
from roomkit.models.room import Room
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import SQLiteStore


class _RetainedWriteStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.reached, self.release = asyncio.Event(), asyncio.Event()
        self.writing: asyncio.Task[Room] | None = None

    async def create_room(self, room: Room) -> Room:
        if room.id == "p":
            return await super().create_room(room)

        async def write() -> Room:
            self.reached.set()
            await self.release.wait()
            return await super(_RetainedWriteStore, self).create_room(room)

        self.writing = asyncio.create_task(write())
        return await asyncio.shield(self.writing)


async def _kit(store: InMemoryStore | SQLiteStore) -> RoomKit:
    kit = RoomKit(store=store)
    kit.register_channel(Agent("worker", provider=MockAIProvider(responses=["unused"])))
    await kit.create_room(room_id="p")
    return kit


async def test_cleanup_waits_for_a_custom_stores_retained_create() -> None:
    store = _RetainedWriteStore()
    kit = await _kit(store)
    call = asyncio.create_task(kit.delegate("p", "worker", "go"))
    try:
        await asyncio.wait_for(store.reached.wait(), 5)
        call.cancel()
        # Give the cancelled caller a turn before allowing the decided write.
        for _ in range(10):
            await asyncio.sleep(0)
        early = call.done()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, 5)
        assert store.writing is not None
        await asyncio.wait_for(store.writing, 5)
        children = [room for room in await store.list_rooms() if room.id != "p"]
        assert [(room.metadata["task_status"], early) for room in children] == [
            ("cancelled", False)
        ]
    finally:
        store.release.set()
        await asyncio.gather(
            call, *([store.writing] if store.writing else []), return_exceptions=True
        )
        await kit.close()


async def test_sqlite_cleanup_waits_for_the_late_create_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteStore(str(tmp_path / "preparation.sqlite"))
    kit = await _kit(store)
    reached, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    create = store._x_create_room

    def write(room: Room) -> Room:
        loop.call_soon_threadsafe(reached.set)
        if not release.wait(5):
            raise RuntimeError("creation gate never released")
        return create(room)

    monkeypatch.setattr(store, "_x_create_room", write)
    call = asyncio.create_task(kit.delegate("p", "worker", "go"))
    try:
        await asyncio.wait_for(reached.wait(), 5)
        call.cancel()
        for _ in range(10):
            await asyncio.sleep(0)
        early = call.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, 5)
        children = [room for room in await store.list_rooms() if room.id != "p"]
        assert [(room.metadata["task_status"], early) for room in children] == [
            ("cancelled", False)
        ]
    finally:
        release.set()
        await asyncio.gather(call, return_exceptions=True)
        await kit.close()
