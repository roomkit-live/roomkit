"""InMemoryStore deep-copy isolation on read, and store lifecycle close()."""

from __future__ import annotations

from roomkit.models.room import Room
from roomkit.store.memory import InMemoryStore
from tests.conftest import make_event


async def test_read_room_is_deep_copy() -> None:
    store = InMemoryStore()
    await store.create_room(Room(id="r1", metadata={"nested": {"x": 1}}))
    got = await store.get_room("r1")
    assert got is not None
    got.metadata["nested"]["x"] = 999  # mutate the read copy
    again = await store.get_room("r1")
    assert again is not None
    assert again.metadata["nested"]["x"] == 1  # stored object untouched


async def test_read_event_is_deep_copy() -> None:
    store = InMemoryStore()
    await store.create_room(Room(id="r1"))
    await store.add_event(make_event(room_id="r1", id="e1", metadata={"nested": {"a": 1}}))
    got = await store.get_event("e1")
    assert got is not None
    got.metadata["nested"]["a"] = 999
    again = await store.get_event("e1")
    assert again is not None
    assert again.metadata["nested"]["a"] == 1


async def test_store_close_is_idempotent_noop() -> None:
    store = InMemoryStore()
    await store.close()
    await store.close()  # idempotent — must not raise


async def test_roomkit_close_closes_store() -> None:
    from roomkit import RoomKit

    closed: list[bool] = []

    class SpyStore(InMemoryStore):
        async def close(self) -> None:
            closed.append(True)
            await super().close()

    kit = RoomKit(store=SpyStore())
    await kit.close()
    assert closed == [True]
