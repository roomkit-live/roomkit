"""The kit's delivery backend a scenario's ``setup.kit`` names (RFC §22.2),
and the calls waiting for a room's lock.

roomkit-specs ``conformance/README.md`` ("Setup"): the in-memory backend,
started with the kit; the runner's idle wait waits for its queue, the items
waiting out a retry and those its worker holds. ("Steps"): the
``wait_for_lock`` step waits until calls wait for a room's lock.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from roomkit import RoomKit
from roomkit.delivery.memory import InMemoryDeliveryBackend
from tests.conformance._calls import read

_BOUNDS = {
    "retry_delay_ms": "retry_delay",
    "max_retry_delay_ms": "max_retry_delay",
}
_SIZES = ("max_queue_size", "max_dead_letter_size")
_LOCK_BOUND_S = 5.0


def delivery_options(kit: Mapping[str, Any]) -> dict[str, Any]:
    """The kit's ``delivery_backend`` keyword a scenario's ``setup.kit`` names."""
    if "delivery_backend" not in kit:
        return {}
    spec = kit["delivery_backend"]
    kwargs: dict[str, Any] = {key: spec[key] for key in _SIZES if key in spec}
    kwargs |= {name: spec[key] / 1000 for key, name in _BOUNDS.items() if key in spec}
    return {"delivery_backend": read(lambda: InMemoryDeliveryBackend(**kwargs))}


async def start_backend(kit: RoomKit) -> None:
    """Starts the kit's delivery backend, as entering the kit does."""
    backend = kit.delivery_backend
    if backend is not None:
        await backend.start(kit)


async def backend_busy(kit: RoomKit) -> bool:
    """Whether the kit's in-memory backend still holds work: a queued item,
    one waiting out its retry delay, or one its worker claimed."""
    backend = kit.delivery_backend
    if not isinstance(backend, InMemoryDeliveryBackend):
        return False
    return bool(await backend.get_queue_depth() or backend._in_flight)


def lock_waiters(kit: RoomKit, room_id: str) -> int:
    """How many calls wait for *room_id*'s lock, read off the kit's
    in-memory lock manager's ``asyncio.Lock``: the core offers no public
    count."""
    lock = kit._given_lock_manager._locks.get(room_id)  # type: ignore[attr-defined]
    waiters = getattr(lock, "_waiters", None) or ()
    return sum(1 for waiter in waiters if not waiter.cancelled())


async def wait_for_lock(kit: RoomKit, room_id: str, waiters: int) -> None:
    """Waits until *waiters* calls wait for *room_id*'s lock.

    Raises:
        TimeoutError: fewer wait after five seconds.
    """
    async with asyncio.timeout(_LOCK_BOUND_S):
        while lock_waiters(kit, room_id) < waiters:
            await asyncio.sleep(0.001)
