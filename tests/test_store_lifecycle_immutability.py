"""InMemoryStore object-ownership contract (RFC §14.4), and store close().

Rooms are caller-owned copies on read. Events flipped direction: the
store copies ON WRITE and shares the immutable snapshot on read — a
caller's mutation of the object it *wrote* must not reach the log, and a
read object is treated as frozen (its isolation is not promised).
"""

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


async def test_written_event_is_copied_in() -> None:
    """The store owns its stored representation from the moment the write
    returns (RFC §14.4): the writer's retained reference is not the log's."""
    store = InMemoryStore()
    await store.create_room(Room(id="r1"))
    event = make_event(room_id="r1", id="e1", metadata={"nested": {"a": 1}})
    await store.add_event(event)
    event.metadata["nested"]["a"] = 999  # the caller mutates ITS object
    again = await store.get_event("e1")
    assert again is not None
    assert again.metadata["nested"]["a"] == 1  # the log never saw it


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
