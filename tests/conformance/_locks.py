"""The kit's lock manager and delivery gap a scenario's ``setup.kit`` names
(RFC §13.5).

roomkit-specs ``conformance/README.md`` ("Setup"): ``lock_manager``, the
in-memory lock manager whose release fails with ``release_error`` while a
``lock_releases`` step has set it failing, and ``delivery_gap_timeout_ms``,
how long a delivery lane waits on a cursor hole before it skips it.
("Steps"): ``{"lock_releases": "fail" | "succeed"}``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from roomkit import InMemoryLockManager, RoomKit

# The framework event of a delivery lane's gap skip (§8.2, §13.5).
LOCK_EVENTS = ("delivery_skipped",)


class LockReleaseFailedError(Exception):
    """What a scripted lock manager's failing release raises."""


class ScriptedLockManager(InMemoryLockManager):
    """The in-memory lock manager, whose release fails while ``failing`` is
    set, as a distributed lock that finds its lease expired at release."""

    def __init__(self, release_error: str) -> None:
        super().__init__()
        self.release_error = release_error
        self.failing = False

    @asynccontextmanager
    async def locked(self, room_id: str) -> AsyncGenerator[None, None]:
        async with super().locked(room_id):
            yield
        if self.failing:
            raise LockReleaseFailedError(self.release_error)


def lock_options(kit: Mapping[str, Any]) -> dict[str, Any]:
    """The kit's ``lock_manager`` and ``delivery_gap_timeout`` keywords a
    scenario's ``setup.kit`` names."""
    kwargs: dict[str, Any] = {}
    if "lock_manager" in kit:
        kwargs["lock_manager"] = ScriptedLockManager(kit["lock_manager"]["release_error"])
    if "delivery_gap_timeout_ms" in kit:
        kwargs["delivery_gap_timeout"] = kit["delivery_gap_timeout_ms"] / 1000
    return kwargs


def set_lock_releases(kit: RoomKit, releases: str) -> None:
    """Sets the releases of the kit's scripted lock manager failing or
    succeeding, from the next one on.

    Raises:
        ValueError: the scenario's kit has no scripted lock manager.
    """
    manager = kit.lock_manager
    if not isinstance(manager, ScriptedLockManager):
        raise ValueError("a lock_releases step needs setup.kit.lock_manager")
    manager.failing = releases == "fail"
