"""Turns carried between processes by Redis (RFC §10.6).

The room owner's process submits each turn to one executor, chosen from what
the executors alive declare (RFC §10.6.7): its own executor alone for the
agents it holds live, else the one holding the room's state for the channel
while it lives, else one that takes it. A turn on an executor
of the owner's own process stays in the process, carried as the in-memory
backend carries it; one on an executor elsewhere crosses Redis, each message
the protocol's envelope written to the other process's inbox, in order, and
acknowledged where it must be: a request is taken or refused (and placed
again), a channel control is applied before the owner goes on. Each process
beats from a thread of its own, and an owner counts a turn's executor alive
while it beats and its process still runs the turn (RFC §10.6.8).

Requires ``pip install roomkit[redis]`` and a store every process shares
(RFC §10.6.9). Redis is the trust boundary: whoever writes to it can act as
any executor or owner, so restrict it (ACL, TLS) as the rooms it carries.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Callable, Coroutine, Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.core.exceptions import TurnLinkError
from roomkit.core.task_utils import cancel_and_wait
from roomkit.turns._codec import control_json, request_json
from roomkit.turns._redis_beat import BeatPayload, BeatThread, LiveExecutors
from roomkit.turns._redis_clients import call_bound, read_block, redis_clients
from roomkit.turns._redis_inbox import Envelope, Inbox
from roomkit.turns._redis_outbox import Failed, Outbox, Written
from roomkit.turns._redis_placement import Elsewhere, Placement
from roomkit.turns._redis_presence import Keys, Registry, member_of
from roomkit.turns._redis_routes import REFUSED, Routes
from roomkit.turns.base import TurnBackend, TurnTimeouts
from roomkit.turns.memory import InMemoryTurnBackend
from roomkit.turns.presence import ExecutorInfo, local_info
from roomkit.turns.wire import (
    ChannelCall,
    ChannelControlEnvelope,
    ChannelResult,
    CommittedEvent,
    RoomDetached,
)

if TYPE_CHECKING:
    from roomkit.turns._redis_links import RedisTurnLink
    from roomkit.turns.base import RoomOwner, TurnLink
    from roomkit.turns.executor import TurnExecutor
    from roomkit.turns.wire import ChannelControl, TurnRequest

logger = logging.getLogger("roomkit.turns")

_DEFAULT_URL = "redis://localhost:6379"


class RedisTurnBackend(TurnBackend):
    """Turns carried between processes by Redis (RFC §10.6).

    One backend per process, given to the kit (``RoomKit(turn_backend=...)``)
    and to the executors that process serves (``await backend.serve(executor)``).
    An executor whose channels are given as a collection
    (``TurnExecutor(build, hosts={"meteo"})``) takes turns from every
    process; one given a predicate, as the kit's own, from its process only.

    Args:
        url: The Redis server (``redis://localhost:6379`` by default). With
            *client*, still required: the process's beat thread connects
            with it.
        client: A ``redis.asyncio`` client to use for the backend's writes
            and reads instead of connecting; its calls keep its own timeouts.
        key_prefix: The prefix of every key the backend writes.
        timeouts: The heartbeat bounds (RFC §10.6.8), kept by the kit given
            this backend: give every process, and every executor's
            ``TurnExecutor(timeouts=...)``, the same.

    Raises:
        ValueError: *client* without *url*.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        client: Any = None,
        key_prefix: str = "roomkit:turns",
        timeouts: TurnTimeouts | None = None,
        _loop_back: bool = False,
    ) -> None:
        if client is not None and url is None:
            raise ValueError(
                "RedisTurnBackend(client=...) needs url= too: its beat connects with it"
            )
        self._timeouts = timeouts or TurnTimeouts()
        self.timeouts = self._timeouts
        clients = redis_clients(url or _DEFAULT_URL, self._timeouts, client)
        self._owns_client = client is None
        self._client = clients.pooled
        self.node_id = uuid4().hex[:12]
        """This process's id among the backend's processes."""
        self._ttl_ms = int(self._timeouts.heartbeat_timeout * 3000)
        keys = Keys(key_prefix)
        self._outbox = Outbox(self._client, keys.inbox, self._ttl_ms)
        self._inbox = Inbox(
            clients.inbox,
            keys.inbox(self.node_id),
            self._route,
            owns_client=True,
            block=read_block(self._timeouts),
            timeout_errors=clients.timeout_errors,
        )
        self._registry = Registry(self._client, keys)
        self._live = LiveExecutors(
            self._registry,
            self._timeouts.heartbeat_interval,
            self._timeouts.heartbeat_timeout,
            lambda: self._routes.open_turns(),
            timeout_errors=clients.timeout_errors,
        )
        self._beat = BeatThread(
            clients.beat,
            keys.beat(self.node_id),
            self._timeouts.heartbeat_interval,
            self._ttl_ms,
            lambda: self._payload(),
            bound=call_bound(self._timeouts),
            timeout_errors=clients.timeout_errors,
        )
        # Turns on executors of this process are carried in it (RFC §10.6.7);
        # the tests' loop back sends them across Redis too.
        self._here = InMemoryTurnBackend()
        self._loop_back = _loop_back
        self._placement = Placement(
            self._client, key_prefix, self.node_id, self._live, loop_back=_loop_back
        )
        self._tasks: set[asyncio.Task[Any]] = set()
        self._routes = Routes(
            self.node_id,
            self._post,
            self._spawn,
            self._live,
            self._here._port,
            end_bound=self._timeouts.heartbeat_timeout,
        )
        self._payload = BeatPayload(self.node_id, self._routes)
        self._started = False
        self._routing = True
        self._closed = False

    async def submit(self, request: TurnRequest) -> TurnLink:
        self._start()
        refused: set[str] = set()
        while True:
            placed = await self._placement.place(
                request.channel_id, request.room_id, request.room_hooks, excluded=refused
            )
            if not isinstance(placed, Elsewhere):
                return self._here._submit_to(placed, request)
            answer, link = await self._hand(request, placed)
            if answer != REFUSED:
                return link
            # An executor that closed: the turn goes to another, never twice.
            request.refused()
            self._routes.forget_owner_link(link)
            refused.add(placed.member)

    async def _hand(self, request: TurnRequest, placed: Elsewhere) -> tuple[str, RedisTurnLink]:
        """Hand *request* to *placed*'s process: its answer (taken or
        refused) and the owner's end of the turn. A process that does not
        answer within the accept bound has lost it: the turn is released
        there, never handed to another (it may have begun)."""
        # Written first: a request that cannot be leaves no link waiting.
        body = request_json(request)
        link = self._routes.owner_link(request, placed.member, placed.executor_id, placed.node)
        request.hand_over()
        try:
            answer = await self._routes.ask(
                placed.node, "request", body, placed.executor_id, bound=self._timeouts.accept
            )
        except TimeoutError as exc:
            link.release()
            raise TurnLinkError(
                "executor_lost", f"Executor {placed.member} did not answer the turn"
            ) from exc
        except BaseException:
            link.release()
            raise
        return answer, link

    async def serve(self, executor: TurnExecutor) -> None:
        """Serve *executor*, in place of one of its id served before; the
        executors that closed are let go."""
        self._start()
        kept = [e for e in _others(self._placement.here, executor) if not e.closed]
        self._placement.here = [*kept, executor]
        if executor.timeouts != self._timeouts:
            logger.warning(
                "Executor %s keeps other timeouts than its turn backend: give both the same "
                "TurnTimeouts (RFC §10.6.8)",
                executor.executor_id,
            )
        if self._loop_back or executor.hosted is not None:
            self._routes.shared[executor.executor_id] = executor
        if executor.hosted is not None:
            await self._beat_for(executor)
        # Work a turn began reaches the owner of that turn, here or elsewhere.
        executor.reach_owner(self._routes.port)
        # The owner's own executor covers nothing kit.executors() lists.
        if executor is not self._placement.own:
            executor._listen_closed(self._executors_changed)
            await self._executors_changed()

    async def claim_presence(self, channel_id: str, room_id: str, online: bool | None) -> bool:
        """Kept in Redis, so of the room owner processes that see the same
        change, one alone publishes it."""
        self._start()
        return await self._registry.claim_presence(channel_id, room_id, online)

    async def executors(self) -> list[ExecutorInfo]:
        """This process's executors as they stand, and the other processes'
        as their beats tell: online while beating within the heartbeat
        timeout, listed while their process has not withdrawn them."""
        self._start()
        own = self._placement.own
        here = [
            local_info(e, self.node_id)
            for e in self._placement.here
            if e is not own and not e.closed
        ]
        elsewhere = [
            ExecutorInfo(
                executor_id=declared.executor_id,
                node=declared.node,
                hosts=declared.channels.entries,
                room_hooks=declared.room_hooks,
                last_beat=datetime.fromtimestamp(beat / 1000, UTC),
                online=online,
            )
            for declared, beat, online in await self._live.declared()
            if declared.node != self.node_id
        ]
        return here + elsewhere

    async def _executors_now(self) -> list[ExecutorInfo]:
        """:meth:`executors` from the registry read now: the view read within
        the last interval misses an executor started since."""
        self._start()
        await self._live.refresh()
        return await self.executors()

    async def attend(self, owner: RoomOwner) -> None:
        self._start()
        await self._here.attend(owner)
        if self._routes.owner is None:
            self._routes.owner = owner
            self._placement.own = owner.executor
            await self._beat_now()

    async def control(self, message: ChannelControl) -> ChannelResult | None:
        self._start()
        raw = control_json(message)
        if isinstance(message, ChannelCall):
            channel_id, room_id = message.channel_id, message.room_id
            return await self._applied_once(
                raw,
                lambda refused: self._placement.place(channel_id, room_id, excluded=refused),
                message,
            )
        if isinstance(message, CommittedEvent):
            channel_id, room_id, hooks = message.channel_id, message.room_id, message.room_hooks
            await self._applied_once(
                raw,
                lambda refused: self._placement.hold(channel_id, room_id, hooks, excluded=refused),
            )
            return None
        if isinstance(message, RoomDetached):
            await self._placement.forget(message.channel_id, message.room_id)
        here, elsewhere = await self._placement.told(message.channel_id)
        for executor in here:
            await executor.control(_crossed(raw))
        results = await asyncio.gather(
            *(self._apply(executor, raw) for executor in elsewhere), return_exceptions=True
        )
        for executor, result in zip(elsewhere, results, strict=True):
            if isinstance(result, Exception) or result is None:
                logger.warning(
                    "Executor %s did not apply %s: %s",
                    executor.member,
                    message.type,
                    result or "its process is closing",
                )
        return None

    async def _applied_once(
        self,
        raw: str,
        choose: Callable[[set[str]], Coroutine[Any, Any, TurnExecutor | Elsewhere]],
        call: ChannelCall | None = None,
    ) -> ChannelResult:
        """Apply a channel control on the executor *choose* places it on,
        placed again, never twice, while a closing process refuses it; a
        *call* is marked handed as it goes to that executor."""
        refused: set[str] = set()
        while True:
            placed = await choose(refused)
            if call is not None:
                call.hand_over()
            result = await self._apply(placed, raw)
            if result is not None:
                return result
            if call is not None:
                call.refused()
            assert isinstance(placed, Elsewhere)
            refused.add(placed.member)

    async def close(self) -> None:
        await self._shut(None)

    async def _close_within(self, until: float) -> set[asyncio.Task[Any]]:
        """:meth:`close` as the kit's close runs it (RFC §12.10.4): every
        wait within one deadline, *until*, and the tasks still running then
        (a channel verb that ignores its cut) left, for the kit to name."""
        return await self._shut(until)

    async def _shut(self, until: float | None) -> set[asyncio.Task[Any]]:
        """Close the backend, each wait on its own bound, and on *until* at
        the latest when given: what is still running then."""
        if self._closed:
            return set()
        self._closed = True
        loop = asyncio.get_running_loop()

        def bound(own: float) -> float:
            return own if until is None else max(0.0, min(own, until - loop.time()))

        # From now what reaches the process is refused at once (a request is
        # placed again) and nothing is asked.
        self._routes.close()
        # Nothing is placed here any more; the process beats on while what
        # it was doing is answered and written (a turn's end among it).
        await self._undeclare(bound)
        await self._answer_what_is_under_way(bound)
        await self._inbox.close()
        self._routing = False
        await self._outbox.close(drain=bound(self._timeouts.cancel))
        # The beat stops first, so none declares again what is withdrawn.
        await asyncio.to_thread(self._beat.stop)
        await self._withdraw(bound)
        await self._live.close()
        await self._here.close()
        left: set[asyncio.Task[Any]] = set()
        if self._tasks and until is None:
            await cancel_and_wait(*self._tasks, log_errors_to=logger)
        elif self._tasks:
            tasks = set(self._tasks)
            for task in tasks:
                task.cancel()
            _, left = await asyncio.wait(tasks, timeout=bound(self._timeouts.cancel))
        if self._owns_client:
            await self._client.aclose()
        return left

    async def _apply(self, placed: TurnExecutor | Elsewhere, raw: str) -> ChannelResult | None:
        """Apply a channel control on *placed*, here or across Redis: its
        result once applied, waited for while the executor lives; None when
        a closing process refused it, having applied nothing."""
        if not isinstance(placed, Elsewhere):
            answer = await placed.control(_crossed(raw))
            if answer is None:
                return ChannelResult()
            return ChannelResult.model_validate_json(answer.model_dump_json())
        body = await self._routes.ask(
            placed.node, "channel", raw, placed.executor_id, alive=self._lives(placed)
        )
        return None if body == REFUSED else ChannelResult.model_validate_json(body)

    def _lives(self, placed: Elsewhere) -> Callable[[], bool]:
        """Whether *placed* still lives: by its beat, or (the tests' loop
        back) as an executor of this process."""
        if placed.node == self.node_id:
            local = self._routes.shared.get(placed.executor_id)
            return lambda: local is not None and not local.closed
        return lambda: self._live.alive(placed.member)

    def _start(self) -> None:
        if self._closed:
            raise TurnLinkError("backend_closed")
        if not self._started:
            self._started = True
            self._inbox.start()
            self._live.start()

    async def _answer_what_is_under_way(self, bound: Callable[[float], float]) -> None:
        """Let what this process was doing answer, within the cancel bound
        (the turns it runs for other processes cut, their ends written), and
        a turn still on its way to an executor it declared reach it and be
        refused: an owner that placed it from an older view writes it within
        an interval. Each wait on *bound* of its own."""
        wait = self._timeouts.heartbeat_interval if self._payload.executors else 0.0
        grace = asyncio.ensure_future(asyncio.sleep(bound(wait)))
        turns = asyncio.ensure_future(self._routes.turns_written(bound(self._timeouts.cancel)))
        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=bound(self._timeouts.cancel))
        await turns
        await grace

    async def _undeclare(self, bound: Callable[[float], float]) -> None:
        """Beat from now without declaring this process's executors."""
        self._payload.close()
        if not self._payload.executors and self._routes.owner is None:
            return  # it never beat
        try:
            async with asyncio.timeout(bound(self._timeouts.cancel)):
                await self._registry.beat(self.node_id, self._payload(), self._ttl_ms)
        except Exception:
            logger.debug("Withdrawing this process's declarations failed", exc_info=True)

    async def _withdraw(self, bound: Callable[[float], float]) -> None:
        """Withdraw this process's executors, and it as a room owner."""
        try:
            owner = self.node_id if self._routes.owner is not None else None
            async with asyncio.timeout(bound(self._timeouts.cancel)):
                await self._registry.withdraw(self._payload.members, owner=owner)
        except Exception:
            logger.debug("Withdrawing this process from Redis failed", exc_info=True)

    async def _beat_for(self, executor: TurnExecutor) -> None:
        """Beat for *executor* from now, in place of one of its id served
        before; the executors that closed are withdrawn, then let go."""
        others = _others(self._payload.executors, executor)
        closed = [member_of(self.node_id, e.executor_id) for e in others if e.closed]
        if closed:
            await self._registry.withdraw(closed)
        self._payload.executors = (*(e for e in others if not e.closed), executor)
        await self._beat_now()

    async def _beat_now(self) -> None:
        """Beat once from this task (alive from now), then from the thread."""
        await self._registry.beat(self.node_id, self._payload(), self._ttl_ms)
        self._beat.start()

    def _route(self, envelope: Envelope) -> None:
        self._routes.route(envelope)

    def _post(
        self,
        node: str,
        kind: str,
        body: str,
        target: str,
        ref: str = "",
        *,
        failed: Failed | None = None,
        writes: int | None = 1,
        within: float | None = None,
        written: Written | None = None,
    ) -> None:
        envelope = Envelope(kind, self.node_id, body, target=target, ref=ref)
        self._outbox.post(node, envelope, failed, writes=writes, within=within, written=written)

    def _spawn(self, work: Coroutine[Any, Any, None], name: str) -> None:
        """Run *work* on a task the backend holds until it closes, named
        *name* for what it serves."""
        if not self._routing:
            work.close()
            return
        task = asyncio.get_running_loop().create_task(
            work, context=contextvars.Context(), name=name
        )
        self._tasks.add(task)
        task.add_done_callback(self._settled)

    def _settled(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("A turn backend message failed", exc_info=error)


def _others(executors: Iterable[TurnExecutor], executor: TurnExecutor) -> list[TurnExecutor]:
    """*executors* but those of *executor*'s id."""
    return [e for e in executors if e.executor_id != executor.executor_id]


def _crossed(raw: str) -> ChannelControl:
    return ChannelControlEnvelope.model_validate_json(raw).message
