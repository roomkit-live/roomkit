"""The kit's delivery backend a scenario's ``setup.kit`` names (RFC §22.2).

roomkit-specs ``conformance/README.md`` ("Setup"): the in-memory backend,
started with the kit; the runner's idle wait waits for its queue, the items
waiting out a retry and those its worker holds.
"""

from __future__ import annotations

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
