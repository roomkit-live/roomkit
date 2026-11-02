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
from uuid import uuid4

import pytest

import roomkit.turns.owner as turns_owner

try:
    import redis
except ImportError:  # this mode only runs with the redis extra
    redis = None  # type: ignore[assignment]
from roomkit.turns.base import TurnTimeouts
from roomkit.turns.redis import RedisTurnBackend

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
