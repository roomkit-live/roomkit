"""How a process of the Redis turn backend beats, and reads the others' beats
(RFC §10.6.8).

The beat runs on a thread of its own, with an event loop and a writer of its
own, so a turn that keeps the kit's event loop busy never stalls it; what it
writes is built from what that loop replaces whole and never changes in
place. The view of
the others is read again each heartbeat interval on the event loop, with
whether the processes running this one's turns still run them.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from roomkit.turns._hosting import Hosted
from roomkit.turns._redis_presence import Declaration, View, member_of
from roomkit.turns.broker import Beat

if TYPE_CHECKING:
    from roomkit.turns._redis_routes import Routes
    from roomkit.turns.broker import BeatWriter, TurnRegistry
    from roomkit.turns.executor import TurnExecutor

logger = logging.getLogger("roomkit.turns")


@dataclass(frozen=True)
class _Served:
    generation: int = 0
    executors: tuple[TurnExecutor, ...] = ()
    closing: bool = False
    withdrawn: frozenset[str] = frozenset()


class BeatPayload:
    """What this process's next beat writes: what each executor it serves
    declares (its room hooks read live), the ones that closed, the turns it
    runs for owners elsewhere, and whether it is a room owner. A process
    closing beats for its executors without declaring them: none is placed
    there any more, while what they were doing is answered."""

    def __init__(self, node: str, routes: Routes) -> None:
        self._node = node
        self._routes = routes
        # The executors it beats for and whether the process closes, numbered
        # each time they change: a beat built from an older state never
        # undoes one built from a newer.
        self._served = _Served()
        self._declared: dict[str, str] = {}

    @property
    def executors(self) -> tuple[TurnExecutor, ...]:
        return self._served.executors

    @property
    def generation(self) -> int:
        return self._served.generation

    def serve(self, executor: TurnExecutor) -> bool:
        """Update this node's advertised set. Withdrawals travel in the same
        numbered beat as the declarations, including after a failed write."""
        previous = self._served
        kept = tuple(
            e for e in previous.executors if e.executor_id != executor.executor_id and not e.closed
        )
        if executor.hosted is not None:
            kept = (*kept, executor)
        elif kept == previous.executors and not previous.withdrawn:
            return False
        advertised = {e.executor_id for e in kept}
        removed = {e.executor_id for e in previous.executors} - advertised
        self._served = replace(
            previous,
            generation=previous.generation + 1,
            executors=kept,
            withdrawn=(previous.withdrawn | removed) - advertised,
        )
        return True

    def written(self, generation: int) -> None:
        """Acknowledge only the withdrawals in the snapshot Redis accepted."""
        if self.generation == generation and self._served.withdrawn:
            self._served = replace(self._served, withdrawn=frozenset())

    def close(self) -> None:
        """Beat from now without declaring the executors."""
        self._served = replace(self._served, generation=self.generation + 1, closing=True)

    @property
    def members(self) -> list[str]:
        served = self._served
        ids = served.withdrawn | {executor.executor_id for executor in served.executors}
        return [member_of(self._node, id_) for id_ in ids]

    def __call__(self) -> Beat:
        served = self._served
        executors = served.executors
        serving = [member_of(self._node, e.executor_id) for e in executors if not e.closed]
        declared = (
            {}
            if served.closing
            else {
                member_of(self._node, e.executor_id): self._declaration(e)
                for e in executors
                if not e.closed
            }
        )
        # A closed executor's member another of its id now serves stays.
        closed = (member_of(self._node, e.executor_id) for e in executors if e.closed)
        gone = [member for member in closed if member not in serving]
        gone.extend(member_of(self._node, eid) for eid in served.withdrawn)
        return Beat(
            node=self._node,
            generation=served.generation,
            declared=declared,
            undeclared=serving if served.closing else [],
            gone=gone,
            running=sorted(self._routes.running),
            owner=self._routes.owner is not None,
        )

    def withdrawn(self) -> Beat:
        """This process's last beat: every executor it beat for is gone, it
        runs no turn and is no room owner."""
        return Beat(node=self._node, generation=self.generation + 1, gone=self.members)

    def _declaration(self, executor: TurnExecutor) -> str:
        """*executor*'s declaration now, the last one kept when the event
        loop changed its room hooks while this thread read them."""
        assert executor.hosted is not None
        member = member_of(self._node, executor.executor_id)
        try:
            hooks = executor.hooks.defined_room_hooks()
        except RuntimeError:
            return self._declared.get(member, "")
        declared = Hosted(executor.hosted)
        dumped = Declaration(
            self._node, executor.executor_id, declared, hooks, executor.organizations
        ).dump()
        self._declared[member] = dumped
        return dumped


class LiveExecutors:
    """A process's view of the processes alive, read each heartbeat interval:
    the executors and what they declare, the owners alive, and which of the
    turns this process reads their processes still run.

    Args:
        registry: The processes' beats and declarations.
        interval: How often the view is read again.
        timeout: How long a silent process stays alive (RFC §10.6.8).
        open_turns: The turns this process reads, by the node running each.
        timeout_errors: What a read that hung raises: this view's own bound's
            error, and the client's when its connection's bound cut it.
    """

    def __init__(
        self,
        registry: TurnRegistry,
        interval: float,
        timeout: float,
        open_turns: Callable[[], dict[str, list[str]]],
        *,
        timeout_errors: tuple[type[Exception], ...] = (TimeoutError,),
    ) -> None:
        self._registry = registry
        self.interval = interval
        """How often the view is read again."""
        self._timeout = timeout
        self._timeout_ms = int(timeout * 1000)
        self._open_turns = open_turns
        self._timeout_errors = timeout_errors
        self._view = View()
        self._running: dict[str, frozenset[str]] = {}
        # The reads failing in a row: how many, from the first one's failure
        # to the last one's.
        self._failures = 0
        self._failing_since: float | None = None
        self._failed_at = 0.0
        self._read_at = 0.0
        self._refresher: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._refresher is None:
            self._refresher = asyncio.get_running_loop().create_task(
                self._refresh_each_interval(), name="roomkit-turns-live-executors"
            )

    async def current(self) -> dict[str, Declaration]:
        """The executors alive and what they declare, by member, read now."""
        await self.refresh()
        return self._alive_declared()

    async def declared(self) -> list[tuple[Declaration, int, bool]]:
        """Every executor declared, with its last beat (the server's ms) and
        whether it beat within the timeout, as read within the last heartbeat
        interval: one silent stays until its member is forgotten."""
        if time.monotonic() - self._read_at >= self.interval:
            await self.refresh()
        view = self._view
        return [
            (declared, view.beats.get(member, 0), self.alive(member))
            for member, declared in view.declared.items()
        ]

    def last_beat(self, member: str) -> int | None:
        """The server time of *member*'s last beat seen, in ms."""
        return self._view.beats.get(member)

    def alive(self, member: str) -> bool:
        """Whether *member* beat within the timeout, as last read."""
        beat = self._view.beats.get(member)
        return not self._unread() and beat is not None and self._view.now - beat < self._timeout_ms

    def runs(self, node: str, turn_id: str) -> bool:
        """Whether *node* still ran *turn_id* when it last beat, as last asked."""
        return turn_id in self._running.get(node, frozenset())

    def owner_alive(self, node: str) -> bool:
        """Whether the owner process *node* beat within the timeout."""
        beat = self._view.owners.get(node)
        return not self._unread() and beat is not None and self._view.now - beat < self._timeout_ms

    async def refresh(self) -> None:
        self._view = View.of(await self._registry.view(10 * self._timeout_ms))
        self._read_at = time.monotonic()
        self._failures = 0
        self._failing_since = None

    async def close(self) -> None:
        if self._refresher is not None:
            self._refresher.cancel()
            await asyncio.gather(self._refresher, return_exceptions=True)
            self._refresher = None

    def _alive_declared(self) -> dict[str, Declaration]:
        view = self._view
        return {
            member: declared
            for member, declared in view.declared.items()
            if view.now - view.beats.get(member, 0) < self._timeout_ms
        }

    def _unread(self) -> bool:
        """Whether the reads of the others kept failing (or hanging) over the
        heartbeat timeout, two of them at least: none of them is then known
        alive. Only the reads this process made count, so a stall of its own
        event loop (one read at most) is not the others' silence."""
        failing = self._failing_since
        return (
            self._failures >= 2
            and failing is not None
            and self._failed_at - failing >= self._timeout
        )

    async def _refresh_each_interval(self) -> None:
        while True:
            started = time.monotonic()
            try:
                # A read that hangs fails as one that raises.
                async with asyncio.timeout(self._timeout):
                    await self.refresh()
                    self._running = await self._still_running(self._open_turns())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._failed(started, hung=isinstance(exc, self._timeout_errors))
                logger.warning("Reading the turn processes alive failed", exc_info=True)
            await asyncio.sleep(self.interval)

    async def _still_running(self, turns: dict[str, list[str]]) -> dict[str, frozenset[str]]:
        """Which of *turns* (by node) each node ran at its last beat."""
        asked = [(node, ids) for node, ids in turns.items() if ids]
        if not asked:
            return {}
        found = (await self._registry.beats((), asked)).running
        return {
            node: frozenset(t for t, runs in zip(ids, flags, strict=True) if runs)
            for (node, ids), flags in zip(asked, found, strict=True)
        }

    def _failed(self, started: float, *, hung: bool) -> None:
        """A read begun at *started* failed. One that hung failed from its
        start, or from the last read that succeeded since; one whose timeout
        fired late (this process's own loop stalled meanwhile), and one that
        raised, from now."""
        now = time.monotonic()
        # A timeout that fired late fired after a stall of this process.
        stalled = now - started > self._timeout + min(0.1, self.interval / 4)
        since = max(started, self._read_at) if hung and not stalled else now
        self._failures += 1
        self._failed_at = now
        if self._failing_since is None:
            self._failing_since = since


class BeatThread:
    """Beats for this process from a thread of its own, on an event loop and
    a writer of its own (RFC §10.6.8): each beat is what *payload* builds
    then.

    Args:
        registry: What keeps the processes' beats: it gives the thread its
            writer.
        interval: How often it beats.
        payload: Builds what each beat writes.
        bound: How long one call of the writer waits for its peer.
        timeout_errors: What a call that bound cut raises: tried again at once.
    """

    def __init__(
        self,
        registry: TurnRegistry,
        interval: float,
        payload: Callable[[], Beat],
        *,
        bound: float,
        timeout_errors: tuple[type[Exception], ...],
    ) -> None:
        self._registry = registry
        self._interval = interval
        self._payload = payload
        self._bound = bound
        self._timeout_errors = timeout_errors
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # The thread's loop and what wakes its wait between two beats.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._woken: asyncio.Event | None = None
        self._writer: BeatWriter | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="roomkit-turns-beat", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        loop, woken = self._loop, self._woken
        if loop is not None and woken is not None:
            # Its loop closed: the thread is gone already.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(woken.set)
        if self._thread is not None:
            # A beat in flight connects, then calls, each within the bound.
            self._thread.join(timeout=2 * self._bound + 1.0)
            self._thread = None

    def _run(self) -> None:
        asyncio.run(self._beat_until_stopped())

    async def _beat_until_stopped(self) -> None:
        self._woken = woken = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        try:
            while True:
                # A beat its bound cut is tried again at once, on a new
                # connection, so it lands inside the silence bound (RFC
                # §10.6.8); a second cut in a row waits the interval.
                if not await self._beat_once() and not self._stop.is_set():
                    await self._beat_once()
                if self._stop.is_set():
                    return
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(self._interval):
                        await woken.wait()
                if self._stop.is_set():
                    return
        finally:
            if self._writer is not None:
                await self._writer.close()

    async def _beat_once(self) -> bool:
        """Beat; False when the call's bound cut it."""
        try:
            if self._writer is None:
                self._writer = await self._registry.beater()
            await self._writer.beat(self._payload())
        except Exception as exc:
            logger.warning("A turn process beat failed", exc_info=True)
            return not isinstance(exc, self._timeout_errors)
        return True
