"""A turn backend across processes' two services kept in this process: the
nodes that share a broker and a registry reach each other as across a
server. Nothing leaves the process, and nothing expires but what a view
forgets.

Both are reached from any thread and any event loop of the process: a
node's beat comes from a thread of its own, and two nodes may run on two
loops.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Callable, Sequence

from roomkit.turns.broker import (
    Beat,
    Beats,
    BeatWriter,
    BrokerMessage,
    InboxEntry,
    RegistryView,
    TurnBroker,
    TurnInbox,
    TurnRegistry,
)

_Wake = Callable[[], None]


class InMemoryTurnBroker(TurnBroker):
    """The messages of a turn backend across processes, carried within this
    process: for tests, and for nodes that share one process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Each node's inbox, its entries by the order they were posted in.
        self._inboxes: dict[str, dict[int, BrokerMessage]] = {}
        self._posted = 0
        # The reads waiting for a message, by the node they read.
        self._waiting: dict[str, set[_Wake]] = {}

    async def post(self, node: str, messages: Sequence[BrokerMessage]) -> None:
        with self._lock:
            inbox = self._inboxes.setdefault(node, {})
            for message in messages:
                self._posted += 1
                inbox[self._posted] = message
            waiting = list(self._waiting.get(node, ()))
        for wake in waiting:
            wake()

    async def inbox(self, node: str, block: float) -> TurnInbox:
        return _Reader(self, node, block)

    async def acknowledge(self, node: str, ids: Sequence[str]) -> None:
        with self._lock:
            inbox = self._inboxes.get(node, {})
            for entry in ids:
                if entry.isdecimal():
                    inbox.pop(int(entry), None)

    def _after(self, node: str, last: int) -> list[InboxEntry]:
        with self._lock:
            inbox = self._inboxes.get(node, {})
            return [InboxEntry(str(id_), message) for id_, message in inbox.items() if id_ > last]

    def _listen(self, node: str, wake: _Wake) -> None:
        with self._lock:
            self._waiting.setdefault(node, set()).add(wake)

    def _unlisten(self, node: str, wake: _Wake) -> None:
        with self._lock:
            waiting = self._waiting.get(node)
            if waiting is not None:
                waiting.discard(wake)
                if not waiting:
                    del self._waiting[node]


class _Reader(TurnInbox):
    def __init__(self, broker: InMemoryTurnBroker, node: str, block: float) -> None:
        self._broker = broker
        self._node = node
        self._block = block

    async def read(self, last: str) -> list[InboxEntry]:
        after = int(last) if last.isdecimal() else 0
        loop = asyncio.get_running_loop()
        posted = asyncio.Event()

        def wake() -> None:
            # Posted from another thread, or to a reader whose loop closed.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(posted.set)

        # Listening before the first look: a message posted between the
        # look and the wait still wakes this read.
        self._broker._listen(self._node, wake)
        try:
            entries = self._broker._after(self._node, after)
            if entries:
                return entries
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self._block):
                    await posted.wait()
            return self._broker._after(self._node, after)
        finally:
            self._broker._unlisten(self._node, wake)


class InMemoryTurnRegistry(TurnRegistry):
    """What the processes of a turn backend across processes share, kept
    within this process: for tests, and for nodes that share one process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.monotonic()
        # Each executor's last beat, and its declaration when it has one.
        self._alive: dict[str, int] = {}
        self._declared: dict[str, str] = {}
        # Each room owner's last beat.
        self._owners: dict[str, int] = {}
        # Each node's last generation and the turns it runs.
        self._generations: dict[str, int] = {}
        self._running: dict[str, frozenset[str]] = {}
        self._holders: dict[tuple[str, str], str] = {}
        self._presence: dict[tuple[str, str], bool] = {}

    async def connect(self) -> None:
        return None

    def connected(self) -> bool:
        return True

    async def beat(self, beat: Beat) -> None:
        self._write(beat)

    async def beater(self) -> BeatWriter:
        return _Writer(self)

    async def view(self, forget_after_ms: int) -> RegistryView:
        with self._lock:
            now = self._now()
            oldest = now - forget_after_ms
            for member in [m for m, beat in self._alive.items() if beat <= oldest]:
                del self._alive[member]
                self._declared.pop(member, None)
            for node in [n for n, beat in self._owners.items() if beat <= oldest]:
                del self._owners[node]
            executors = {
                member: (beat, self._declared.get(member)) for member, beat in self._alive.items()
            }
            return RegistryView(now=now, executors=executors, owners=dict(self._owners))

    async def forget_owner(self, node: str) -> None:
        with self._lock:
            self._owners.pop(node, None)

    async def beats(
        self, members: Sequence[str], running: Sequence[tuple[str, Sequence[str]]]
    ) -> Beats:
        with self._lock:
            return Beats(
                last=[self._alive.get(member) for member in members],
                running=[
                    [turn in self._running.get(node, frozenset()) for turn in turns]
                    for node, turns in running
                ],
            )

    async def holder(self, channel: str, room: str) -> str | None:
        with self._lock:
            return self._holders.get((channel, room))

    async def hold(self, channel: str, room: str, member: str) -> None:
        with self._lock:
            self._holders[(channel, room)] = member

    async def forget_holder(self, channel: str, room: str) -> None:
        with self._lock:
            self._holders.pop((channel, room), None)

    async def claim_presence(self, channel_id: str, room_id: str, online: bool | None) -> bool:
        key = (room_id, channel_id)
        with self._lock:
            if online is None:
                self._presence.pop(key, None)
                return False
            changed = self._presence.get(key) != online
            self._presence[key] = online
            return changed

    def _now(self) -> int:
        """The registry's time, in milliseconds."""
        return int((time.monotonic() - self._started) * 1000)

    def _write(self, beat: Beat) -> None:
        with self._lock:
            now = self._now()
            if beat.generation < self._generations.get(beat.node, 0):
                return
            self._generations[beat.node] = beat.generation
            for member, declared in beat.declared.items():
                self._alive[member] = now
                self._declared[member] = declared
            for member in beat.undeclared:
                self._alive[member] = now
                self._declared.pop(member, None)
            for member in beat.gone:
                self._alive.pop(member, None)
                self._declared.pop(member, None)
            if beat.owner:
                self._owners[beat.node] = now
            self._running[beat.node] = frozenset(beat.running)


class _Writer(BeatWriter):
    def __init__(self, registry: InMemoryTurnRegistry) -> None:
        self._registry = registry

    async def beat(self, beat: Beat) -> None:
        self._registry._write(beat)
