"""Keep a delegated child's creation write ahead of its cleanup (RFC §23.3)."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from roomkit.core.task_utils import run_to_its_end

if TYPE_CHECKING:
    from roomkit.models.room import Room
    from roomkit.store.base import ConversationStore

_PREPARING: ContextVar[tuple[ConversationStore, str] | None] = ContextVar(
    "roomkit_preparing_child", default=None
)


@contextmanager
def preparing_child(store: ConversationStore, room_id: str) -> Generator[None]:
    """Protect this child's write only, not hooks or other rooms it creates."""
    token = _PREPARING.set((store, room_id))
    try:
        yield
    finally:
        _PREPARING.reset(token)


async def write_created_room(
    store: ConversationStore, room: Room, held: set[asyncio.Task[Any]]
) -> Room:
    preparing = _PREPARING.get()
    write = store.create_room(room)
    if preparing is None or preparing[0] is not store or preparing[1] != room.id:
        return await write
    # A store may keep writing after its waiter is cancelled. Wait for the
    # decided creation before the delegation reads its child for cleanup.
    return await run_to_its_end(write, held=held, name=f"roomkit-create-child-{room.id}")
