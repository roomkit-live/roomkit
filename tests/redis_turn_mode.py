"""The suite with every kit's turns across a real Redis (RFC §10.6).

``ROOMKIT_TEST_TURN_BACKEND=redis REDIS_URL=redis://... uv run pytest`` gives
each kit built with no turn backend a RedisTurnBackend whose own executor is
reached across Redis too, in a fresh context, as an executor in another
process would be: the intelligence channels' tests then run through the
distributed transport. A test that assumes the kit's turns run in its
process is marked ``in_process`` (naming why) and skipped in this mode.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

import roomkit.turns.owner as turns_owner

try:
    import redis
except ImportError:  # this mode only runs with the redis extra
    redis = None  # type: ignore[assignment]
from roomkit.turns.base import TurnTimeouts
from roomkit.turns.redis import RedisTurnBackend

if TYPE_CHECKING:
    from roomkit import RoomKit

ENABLED = os.environ.get("ROOMKIT_TEST_TURN_BACKEND") == "redis"
# This session's keys: another session on the same Redis keeps its own.
SESSION_PREFIX = f"roomkit:test:{uuid4().hex[:12]}"
# The backends built since the last test ended. A kit and its loop live
# within one test (no fixture shares them), so each is that test's.
_built: list[RedisTurnBackend] = []


def backend(timeouts: TurnTimeouts | None = None) -> RedisTurnBackend:
    """A backend of this session's, keeping *timeouts* when given."""
    built = RedisTurnBackend(
        os.environ["REDIS_URL"],
        key_prefix=f"{SESSION_PREFIX}:{uuid4().hex[:12]}",
        timeouts=timeouts,
        _loop_back=True,
    )
    _built.append(built)
    return built


async def close_left_open() -> None:
    """Close the backends of the kits the test left open, waiting for
    nothing: the test is over. Left open, a backend beats on from a thread
    of its own (RFC §10.6.8), which keeps it and its Redis connections
    alive for the rest of the worker; a worker past 1024 descriptors then
    hands the console shell a pipe ``select`` refuses (RMK-1224)."""
    loop = asyncio.get_running_loop()
    while _built:
        await _built.pop()._close_within(loop.time())


async def handled_across(kit: RoomKit) -> bool:
    """Wait until what *kit*'s backend posted across Redis before now was
    handled here: work a turn began, which reaches its owner once the turn
    ended (RFC §10.6.3), as an ASYNC hook's ``hook_timeout``. Whether any
    was still being handled.

    It reads the backend's internals, as a test's idle wait reads the kit's:
    a marker this process posts to its own inbox is read after what was
    posted there before it (one writer per inbox, one reader), so once its
    answer came back, what came before is handled or being handled.
    """
    backend = kit._turns.backend
    if not isinstance(backend, RedisTurnBackend) or not backend._started or backend._closed:
        # Its inbox is read only once it started: nothing reached it before.
        return False
    ref = uuid4().hex
    marker = asyncio.get_running_loop().create_future()
    backend._routes._answers[ref] = marker

    def failed(error: Exception) -> None:
        if not marker.done():
            marker.set_exception(error)

    try:
        backend._post(backend.node_id, "answer", "", "", ref, failed=failed)
        await marker
    finally:
        backend._routes._answers.pop(ref, None)
    # Only the work's handlers: a channel control or a room call is waited
    # for by what asked it.
    handling = [
        task
        for task in backend._tasks
        if not task.done() and task.get_name().startswith("roomkit-turns-owner-work-")
    ]
    if handling:
        await asyncio.wait(handling)
    return bool(handling)


def clean_up() -> None:
    """Delete the keys this session's backends left (their holders live a
    week), and only those."""
    assert redis is not None
    client = redis.Redis.from_url(os.environ["REDIS_URL"])
    try:
        for keys in _batches(client.scan_iter(f"{SESSION_PREFIX}:*", count=1000)):
            client.delete(*keys)
    finally:
        client.close()


def _batches(keys: Iterator[bytes], size: int = 500) -> Iterator[list[bytes]]:
    batch: list[bytes] = []
    for key in keys:
        batch.append(key)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def install(config: pytest.Config) -> None:
    """Give every kit built with no turn backend one across Redis."""
    if not os.environ.get("REDIS_URL"):
        raise pytest.UsageError("ROOMKIT_TEST_TURN_BACKEND=redis needs REDIS_URL")
    turns_owner.InMemoryTurnBackend = backend  # type: ignore[misc, assignment]
