"""Shared test fixtures and helpers."""

from __future__ import annotations

import asyncio
import collections
import gc
import os
import traceback
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from typing import Any

import pytest

from roomkit.core import framework, task_utils
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import (
    CompositeContent,
    EventSource,
    MediaContent,
    RoomEvent,
    TextContent,
)
from roomkit.models.event import EventContent as EventContentType
from roomkit.models.room import Room
from roomkit.store.memory import InMemoryStore
from roomkit.turns import InMemoryTurnBackend
from roomkit.turns.base import TurnBackend
from roomkit.turns.redis import RedisTurnBackend
from tests import redis_turn_mode
from tests.redis_keys import delete_keys, key_prefix
from tests.tool_loop_modes import expect_streaming


@pytest.fixture
def advance() -> Callable[[int], Coroutine[Any, Any, None]]:
    """Yield control to let pending tasks run without real delay.

    Replaces ``await asyncio.sleep(0.05)`` patterns with zero-delay
    event loop yields::

        await advance()       # 5 yields (default)
        await advance(10)     # 10 yields for heavier workloads
    """

    async def _advance(n: int = 5) -> None:
        for _ in range(n):
            await asyncio.sleep(0)

    return _advance


@pytest.fixture
def close_grace(monkeypatch: pytest.MonkeyPatch) -> float:
    """The grace the kit's close gives its running work, 50 ms under test:
    work a test leaves running as the kit closes is cut then, not after the
    production 5 s (RFC §12.10.4)."""
    grace = 0.05
    monkeypatch.setattr(framework, "CLOSE_GRACE_S", grace)
    return grace


@pytest.fixture
def cut_unwind(monkeypatch: pytest.MonkeyPatch) -> float:
    """How long the kit's close lets the work it cut unwind, 30 ms under
    test: work that ignores its cut is abandoned then, not after the
    production 1 s (RFC §12.10.4)."""
    unwind = 0.03
    monkeypatch.setattr(framework, "CUT_UNWIND_S", unwind)
    monkeypatch.setattr(task_utils, "CUT_UNWIND_S", unwind)
    return unwind


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def room() -> Room:
    return Room(id="test-room")


@pytest.fixture(params=["memory", "redis"])
async def turn_backend(request: pytest.FixtureRequest) -> AsyncIterator[TurnBackend]:
    """A turn backend in memory, then across a real Redis when REDIS_URL is
    set, the owner reaching its own executor there as one elsewhere; its keys
    deleted after."""
    if request.param == "memory":
        yield InMemoryTurnBackend()
        return
    if not os.environ.get("REDIS_URL"):
        pytest.skip("REDIS_URL not set")
    pytest.importorskip("redis.asyncio")
    prefix = key_prefix()
    backend = RedisTurnBackend(os.environ["REDIS_URL"], key_prefix=prefix, _loop_back=True)
    yield backend
    # Closed even when the test failed before closing it (idempotent).
    await backend.close()
    await delete_keys(prefix)


@pytest.fixture(params=[False, True], ids=["non-streaming", "streaming"])
def streaming(request: pytest.FixtureRequest) -> Iterator[bool]:
    """Run a tool-loop test once per kind of provider: one that streams and
    one read through its ``generate()``.

    An AIChannel runs one tool loop whatever its provider streams (RFC §6.4);
    the two kinds reach it through different adapters. Pass the value to
    ``MockAIProvider(streaming=...)`` and drive the turn with
    ``tests/tool_loop_modes.py``, whose ``respond`` checks the provider runs in
    the mode the test does.

    A known divergence between the two kinds is marked on the test with
    ``@pytest.mark.xfail_streaming("RMK-…: …")`` (or ``xfail_non_streaming``):
    that variant alone becomes a strict xfail, so the fix turns it red until
    the mark goes.
    """
    marker = request.node.get_closest_marker(
        "xfail_streaming" if request.param else "xfail_non_streaming"
    )
    if marker is not None:
        request.applymarker(pytest.mark.xfail(strict=True, reason=marker.args[0]))
    expect_streaming(request.param)
    yield request.param
    expect_streaming(None)


if redis_turn_mode.ENABLED:

    @pytest.fixture(autouse=True)
    async def _redis_turn_backends_closed() -> AsyncIterator[None]:
        """Close what the test's kits left open across Redis
        (tests/redis_turn_mode.py). Defined in that mode only: elsewhere a
        synchronous test keeps no event loop it did not ask for."""
        yield
        await redis_turn_mode.close_left_open()


def pytest_configure(config: pytest.Config) -> None:
    if redis_turn_mode.ENABLED:
        redis_turn_mode.install(config)


def pytest_unconfigure(config: pytest.Config) -> None:
    if redis_turn_mode.ENABLED:
        redis_turn_mode.clean_up()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """A divergence mark needs its reason and the ``streaming`` fixture.

    Without the fixture the mark would be ignored and the divergence hidden;
    without a reason nobody knows which card owns it. An ``in_process`` mark
    needs its reason too, and skips its test when the kit's turns cross
    Redis (tests/redis_turn_mode.py).
    """
    for item in items:
        in_process = item.get_closest_marker("in_process")
        if in_process is not None:
            if not in_process.args or not isinstance(in_process.args[0], str):
                raise pytest.UsageError(f"{item.nodeid}: @in_process needs the reason")
            if redis_turn_mode.ENABLED:
                item.add_marker(pytest.mark.skip(reason=f"in process: {in_process.args[0]}"))
        for name in ("xfail_streaming", "xfail_non_streaming"):
            marker = item.get_closest_marker(name)
            if marker is None:
                continue
            if not marker.args or not isinstance(marker.args[0], str):
                raise pytest.UsageError(
                    f"{item.nodeid}: @{name} needs the reason, naming its card"
                )
            if "streaming" not in getattr(item, "fixturenames", ()):
                raise pytest.UsageError(
                    f"{item.nodeid}: @{name} applies only to a test that takes `streaming`"
                )


def pytest_collection_finish(session: pytest.Session) -> None:
    """Keep what the collection built out of the collector's reach.

    The modules, the tests and their parameters, about a million objects in
    a worker of the whole suite, live until the run ends. Rescanned by every
    full collection, they made each one pause the worker 0.25 to 0.4 s (on
    Python 3.12), longer on a loaded CI runner: enough to run out a test's
    sub-second bound on real work (RMK-1224). Frozen, they are never scanned
    again; what the tests allocate is collected as before.
    """
    gc.collect()
    gc.freeze()


class PoolCheckoutRecorder:
    """Stand-in for a store's connection pool that records every checkout.

    Substitute it for ``PostgresStore._ensure_pool()`` to count how many pooled
    connections a piece of work costs. A connection lent from an open
    ``store.connection()`` block never reaches the pool, so it is invisible
    here — which is exactly the quantity worth measuring: asyncpg resets a
    connection on release, so a checkout is a full extra round trip.

    Everything other than ``acquire`` is delegated to the real pool.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self.by_call_site: collections.Counter[str] = collections.Counter()

    @property
    def total(self) -> int:
        return sum(self.by_call_site.values())

    def reset(self) -> None:
        """Forget what was recorded — e.g. after a warm-up phase."""
        self.by_call_site.clear()

    def breakdown(self, per: int = 1) -> str:
        """The recorded checkouts per call site, divided by *per*."""
        return "\n".join(
            f"  {count / per:5.2f}  {site}" for site, count in self.by_call_site.most_common()
        )

    def acquire(self, *args: Any, **kwargs: Any) -> Any:
        self.by_call_site[_store_call_site()] += 1
        return self._pool.acquire(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)


def _store_call_site() -> str:
    """The innermost store method, plus the frame that asked it for a connection."""
    stack = traceback.extract_stack()[:-2]
    store_frame = next((f for f in reversed(stack) if "roomkit/store/" in f.filename), None)
    caller = next(
        (
            f
            for f in reversed(stack)
            if "roomkit/" in f.filename and "roomkit/store/" not in f.filename
        ),
        None,
    )
    return "{} <- {}:{} {}".format(
        store_frame.name if store_frame else "?",
        os.path.basename(caller.filename) if caller else "?",
        caller.lineno if caller else "?",
        caller.name if caller else "?",
    )


def make_event(
    room_id: str = "test-room",
    channel_id: str = "ch1",
    channel_type: ChannelType = ChannelType.SMS,
    body: str = "hello",
    participant_id: str | None = None,
    **kwargs: object,
) -> RoomEvent:
    return RoomEvent(
        room_id=room_id,
        source=EventSource(
            channel_id=channel_id, channel_type=channel_type, participant_id=participant_id
        ),
        content=TextContent(body=body),
        **kwargs,  # type: ignore[arg-type]
    )


def make_media_event(
    room_id: str = "test-room",
    channel_id: str = "ch1",
    channel_type: ChannelType = ChannelType.SMS,
    url: str = "https://example.com/image.jpg",
    mime_type: str = "image/jpeg",
    caption: str | None = None,
    extra_urls: list[str] | None = None,
    body: str | None = None,
    **kwargs: object,
) -> RoomEvent:
    """Create a RoomEvent with media content.

    Args:
        url: Primary media URL.
        mime_type: MIME type for the primary media.
        caption: Optional caption for single-media messages.
        extra_urls: Additional media URLs for composite content.
        body: Text body (used with extra_urls for composite content).
    """
    content: EventContentType
    if extra_urls:
        parts: list[TextContent | MediaContent] = []
        if body:
            parts.append(TextContent(body=body))
        parts.append(MediaContent(url=url, mime_type=mime_type))
        for extra in extra_urls:
            parts.append(MediaContent(url=extra, mime_type=mime_type))
        content = CompositeContent(parts=parts)  # type: ignore[arg-type]
    else:
        content = MediaContent(url=url, mime_type=mime_type, caption=caption)

    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id=channel_id, channel_type=channel_type),
        content=content,
        **kwargs,  # type: ignore[arg-type]
    )


def make_binding(
    channel_id: str = "ch1",
    room_id: str = "test-room",
    channel_type: ChannelType = ChannelType.SMS,
    media_types: list[ChannelMediaType] | None = None,
) -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id=room_id,
        channel_type=channel_type,
        capabilities=ChannelCapabilities(media_types=media_types or [ChannelMediaType.TEXT]),
    )
