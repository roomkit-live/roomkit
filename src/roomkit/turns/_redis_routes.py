"""What reaches a process's inbox on the Redis turn backend, and where it goes.

Each kind of entry has its route: a frame to the owner's end of its turn, a
control message to the executor's, a request to the executor it names, a
channel control to that executor, an answer to the question waiting for it,
and the work a turn began after it ended to the room owner that submitted the
turn (RFC §10.6.3), whichever process that owner runs in. A request and a
channel control are answered, so the owner goes on only once the executor's
process took them, in order, as the in-memory backend does.

An entry can arrive twice: a client that runs a pipeline again when its
connection dropped after Redis executed it writes its entries twice. A
numbered message is admitted once by its turn's order; a question (a request,
a channel control, work after a turn) is handled once by its reference, and
asked again it is answered again.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.core.exceptions import TurnLinkError
from roomkit.turns._error_codec import error_record
from roomkit.turns._redis_links import RedisExecutorLink, RedisTurnLink
from roomkit.turns.base import RoomOwner
from roomkit.turns.wire import (
    ChannelCall,
    ChannelControlEnvelope,
    ChannelResult,
    HostResult,
    OwnerMessageEnvelope,
    RoomCall,
    RoomPublication,
    TurnRequest,
)

if TYPE_CHECKING:
    from roomkit.turns._redis_beat import LiveExecutors
    from roomkit.turns._redis_links import Post
    from roomkit.turns.broker import BrokerMessage
    from roomkit.turns.executor import TurnExecutor
    from roomkit.turns.wire import ChannelControl

logger = logging.getLogger("roomkit.turns")

# How many turns' owners are remembered for the work they began (RFC §10.6.3).
_TURN_OWNERS = 10_000
# How many questions handled are remembered, and for how long, so one written
# twice is not handled twice: a pipeline runs again within moments.
_HANDLED = 10_000
_HANDLED_FOR = 60.0
# The entries handled once by their reference, and answered.
_QUESTIONS = frozenset({"request", "channel", "owner"})
# Work after a turn that nothing answers is written again for _WORK_FOR
# while Redis refuses it, and remembered by its owner for longer, so it is
# published once.
_WORK_FOR = 300.0
_PUBLISHED = 100_000
_PUBLISHED_FOR = 600.0
TAKEN = "taken"
"""What an executor's process answers a request it handed to its executor."""
REFUSED = "refused"
"""What it answers a request for an executor that no longer serves."""


class Routes:
    """The ends of this process's turns, what it waits for, and the routes
    of what its inbox receives.

    Args:
        node: This process's node id.
        post: Queues a message for a node's inbox (it never waits).
        spawn: Runs a coroutine on a task the backend holds.
        live: The processes alive, as this one reads them.
        local_owner: What the owner of this process is reached by for a turn
            it submitted to an executor here.
        end_bound: How long a turn that ended stays running here while its
            end is being written (the heartbeat timeout).
    """

    def __init__(
        self,
        node: str,
        post: Post,
        spawn: Callable[[Coroutine[Any, Any, None], str], None],
        live: LiveExecutors,
        local_owner: RoomOwner,
        *,
        end_bound: float,
    ) -> None:
        self._node = node
        self._end_bound = end_bound
        self._post = post
        self._spawn = spawn
        self._live = live
        self._local_owner = local_owner
        self.shared: dict[str, TurnExecutor] = {}
        """The executors of this process other processes reach, by id."""
        self.owner: RoomOwner | None = None
        """The room owner of this process, once it attends."""
        self.running: frozenset[str] = frozenset()
        """The turns of other processes this one runs, each until its end is
        written: replaced, never changed, so the beat thread reads it whole."""
        self._owner_links: dict[str, RedisTurnLink] = {}
        self._executor_links: dict[str, RedisExecutorLink] = {}
        self._none_running = asyncio.Event()
        self._none_running.set()
        self._answers: dict[str, asyncio.Future[str]] = {}
        self._turn_owners: OrderedDict[str, str] = OrderedDict()
        # By reference: when it came, and the answer given (None while it is
        # worked out, or for work after a turn, which nothing answers).
        self._handled: OrderedDict[str, tuple[float, str | None]] = OrderedDict()
        # By reference, when it came: the work after a turn handled.
        self._published: OrderedDict[str, float] = OrderedDict()
        self._closing = False
        self._routes: dict[str, Callable[[BrokerMessage], None]] = {
            "frame": self._frame,
            "control": self._control,
            "released": self._released,
            "request": self._request,
            "channel": lambda envelope: self._spawn(
                self._channel(envelope), f"roomkit-turns-channel-control-{envelope.target}"
            ),
            "owner": lambda envelope: self._spawn(
                self._owner_message(envelope), f"roomkit-turns-owner-call-{envelope.sender}"
            ),
            "work": lambda envelope: self._spawn(
                self._owner_message(envelope), f"roomkit-turns-owner-work-{envelope.sender}"
            ),
            "answer": self._answer,
        }
        self.port: RoomOwner = _OwnerPort(self)
        """What this process's executors reach a room owner by outside a turn."""

    def route(self, envelope: BrokerMessage) -> None:
        """Hand one inbox entry on, in order, without waiting; a question
        written twice is answered again, not handled again, and one reaching
        a closing process is refused."""
        route = self._routes.get(envelope.kind)
        if route is None:
            logger.warning("A turn inbox entry of an unknown kind %r was dropped", envelope.kind)
            return
        if envelope.kind in _QUESTIONS and not self._first_time(envelope):
            return
        if envelope.kind == "work" and not self._first_work(envelope):
            return
        if self._closing and envelope.kind in (*_QUESTIONS, "work"):
            self._refuse_closing(envelope)
            return
        route(envelope)

    def owner_link(
        self, request: TurnRequest, member: str, executor_id: str, node: str
    ) -> RedisTurnLink:
        """The owner's end of a turn submitted to *executor_id* at *node*."""
        link = RedisTurnLink(
            request, member, executor_id, node, self._post, self._live, self.forget_owner_link
        )
        self._owner_links[request.turn_id] = link
        return link

    def forget_owner_link(self, link: RedisTurnLink) -> None:
        if self._owner_links.get(link.request.turn_id) is link:
            del self._owner_links[link.request.turn_id]

    async def ask(
        self,
        node: str,
        kind: str,
        body: str,
        target: str,
        *,
        alive: Callable[[], bool] | None = None,
        bound: float | None = None,
    ) -> str:
        """Post a question to *node* and wait for its answer's body, while
        *alive* says the other side lives (checked each heartbeat interval),
        and within *bound* when there is one; none is asked once closing."""
        if self._closing:
            raise TurnLinkError("backend_closed")
        ref = uuid4().hex
        answer: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._answers[ref] = answer

        def failed(error: Exception) -> None:
            if not answer.done():
                answer.set_exception(TurnLinkError("frame_lost", f"Not written: {error}"))

        try:
            self._post(node, kind, body, target, ref, failed=failed)
            async with asyncio.timeout(bound):
                return await self._answered_while_alive(answer, alive)
        finally:
            self._answers.pop(ref, None)

    def close(self) -> None:
        """The backend closes: every owner's end of a turn and every question
        learns it, and each turn run here for an owner elsewhere is cut, its
        end still written (``turns_written``). What still arrives is routed
        until the inbox closes, a request refused, so its owner places it
        again."""
        self._closing = True
        for link in list(self._owner_links.values()):
            link.end("backend_closed")
        for executor_link in list(self._executor_links.values()):
            executor_link.cut("backend_closed")
        for answer in self._answers.values():
            if not answer.done():
                answer.set_exception(TurnLinkError("backend_closed"))

    async def turns_written(self, bound: float) -> None:
        """Wait, within *bound*, until the turns run here ended and their
        ends are written; one still running then stops, its owner learning
        it from its silence."""
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(bound):
                await self._none_running.wait()
        for executor_link in list(self._executor_links.values()):
            executor_link.end("backend_closed")

    def open_turns(self) -> dict[str, list[str]]:
        """The turns this process submitted elsewhere and still reads, by
        the node of the executor running each."""
        turns: dict[str, list[str]] = {}
        for turn_id, link in self._owner_links.items():
            turns.setdefault(link.node, []).append(turn_id)
        return turns

    def turn_owner(self, turn_id: str) -> str | None:
        """The node of the owner that submitted *turn_id* to an executor here."""
        return self._turn_owners.get(turn_id)

    async def _answered_while_alive(
        self, answer: asyncio.Future[str], alive: Callable[[], bool] | None
    ) -> str:
        while True:
            done, _ = await asyncio.wait({answer}, timeout=self._live.interval)
            if done:
                return answer.result()
            if alive is not None and not alive():
                raise TurnLinkError("executor_lost", "The other side stopped beating")

    def _forget_executor_link(self, link: RedisExecutorLink) -> None:
        if self._executor_links.get(link.request.turn_id) is link:
            del self._executor_links[link.request.turn_id]
            self.running = frozenset(self._executor_links)
            if not self._executor_links:
                self._none_running.set()

    def _first_time(self, question: BrokerMessage) -> bool:
        """Whether *question* is new; one handled before is answered again
        with what it was answered."""
        handled = self._handled.get(question.ref)
        if handled is None:
            now = time.monotonic()
            self._handled[question.ref] = (now, None)
            while len(self._handled) > _HANDLED or (
                self._handled and next(iter(self._handled.values()))[0] < now - _HANDLED_FOR
            ):
                self._handled.popitem(last=False)
            return True
        logger.debug("A %s written twice was handled once", question.kind)
        if (answer := handled[1]) is not None:
            self._post_answer(question, answer)
        return False

    def _first_work(self, work: BrokerMessage) -> bool:
        """Whether *work* after a turn is new: written again, it is not
        published twice."""
        if work.ref in self._published:
            return False
        now = time.monotonic()
        self._published[work.ref] = now
        while len(self._published) > _PUBLISHED or (
            next(iter(self._published.values())) < now - _PUBLISHED_FOR
        ):
            self._published.popitem(last=False)
        return True

    def _refuse_closing(self, envelope: BrokerMessage) -> None:
        """Answer what reaches this closing process at once, applying
        nothing: a request or a channel control is placed again, a room call
        learns it failed."""
        closing = error_record(TurnLinkError("backend_closed"))
        if envelope.kind in ("request", "channel"):
            self._answer_back(envelope, REFUSED)
        elif envelope.kind == "owner":
            self._answer_back(envelope, HostResult(call_id="", error=closing).model_dump_json())
        else:
            logger.warning("Work after a turn reached this closing process and was dropped")

    # -- Routes ---------------------------------------------------------------------

    def _frame(self, envelope: BrokerMessage) -> None:
        if (link := self._owner_links.get(envelope.target)) is not None:
            link.deliver(envelope.body)

    def _control(self, envelope: BrokerMessage) -> None:
        if (link := self._executor_links.get(envelope.target)) is not None:
            link.deliver(envelope.body)

    def _released(self, envelope: BrokerMessage) -> None:
        if (link := self._executor_links.get(envelope.target)) is not None:
            link.end("backend_closed")

    def _answer(self, envelope: BrokerMessage) -> None:
        if (answer := self._answers.get(envelope.ref)) is not None and not answer.done():
            answer.set_result(envelope.body)

    def receiver(self, sender: str, target: str, channel_id: str) -> TurnExecutor | None:
        """Resolve a loopback id shared by a served worker and the room owner's executor."""
        served = self.shared.get(target)
        if sender == self._node and self.owner is not None:
            own = self.owner.executor
            if (
                own is not None
                and own is not served
                and own.executor_id == target
                and own.hosts_channel(channel_id)
            ):
                return own
        return served

    def _request(self, envelope: BrokerMessage) -> None:
        """A request of an owner elsewhere, for an executor of this process,
        answered at once: taken, or refused by an executor that closed."""
        served = self.shared.get(envelope.target)
        if served is None or (envelope.sender != self._node and served.closed):
            self._answer_back(envelope, REFUSED)
            return
        request = TurnRequest.model_validate_json(envelope.body)
        executor = self.receiver(envelope.sender, envelope.target, request.channel_id)
        if executor is None or executor.closed:
            self._answer_back(envelope, REFUSED)
            return
        link = RedisExecutorLink(
            request,
            envelope.sender,
            self._post,
            self._forget_executor_link,
            end_bound=self._end_bound,
        )
        self._executor_links[request.turn_id] = link
        self.running = frozenset(self._executor_links)
        self._none_running.clear()
        self._turn_owners[request.turn_id] = envelope.sender
        while len(self._turn_owners) > _TURN_OWNERS:
            self._turn_owners.popitem(last=False)
        executor.take(link)
        self._answer_back(envelope, TAKEN)

    async def _channel(self, envelope: BrokerMessage) -> None:
        """A channel control for an executor here, applied, then answered,
        with what it raised if it did: its owner waits for the answer."""
        served = self.shared.get(envelope.target)
        if served is not None and envelope.sender != self._node and served.closed:
            self._answer_back(envelope, REFUSED)
            return
        result: ChannelResult | None = ChannelResult()
        if served is None:
            logger.warning(
                "A channel control came for executor %s, which no longer serves", envelope.target
            )
            gone = TurnLinkError("no_executor", f"Executor {envelope.target} no longer serves")
            result = ChannelResult(error=error_record(gone))
        else:
            try:
                control = _crossed(envelope.body)
                executor = self.receiver(envelope.sender, envelope.target, control.channel_id)
                if executor is None or executor.closed:
                    self._answer_back(envelope, REFUSED)
                    return
                if isinstance(control, ChannelCall) and (task := asyncio.current_task()):
                    # Named for the verb it serves, as the kit's close names
                    # one still running past its cut.
                    task.set_name(
                        f"roomkit-turns-channel-call-{control.channel_id}-{control.verb}"
                    )
                result = await executor.control(control)
            except Exception as exc:
                logger.exception("A channel control for executor %s failed", envelope.target)
                result = ChannelResult(error=error_record(exc), executor_id=envelope.target)
        self._answer_back(envelope, (result or ChannelResult()).model_dump_json())

    async def _owner_message(self, envelope: BrokerMessage) -> None:
        """Work an executor elsewhere began in a turn of this owner (RFC
        §10.6.3); a room call is answered, with what it raised if it did."""
        try:
            message = OwnerMessageEnvelope.model_validate_json(envelope.body).message
        except ValueError as exc:
            # Whatever it was, a question waiting for it learns it failed.
            logger.warning("Work after a turn from %s was malformed", envelope.sender)
            self._answer_back(
                envelope, HostResult(call_id="", error=error_record(exc)).model_dump_json()
            )
            return
        if isinstance(message, RoomPublication):
            if self.owner is not None:
                await _apart(self.owner.room_publication(message))
            return
        self._answer_back(envelope, (await self._room_call(message)).model_dump_json())

    async def _room_call(self, message: RoomCall) -> HostResult:
        if self.owner is None:
            refusal = RuntimeError("No room owner is reached outside a turn on this backend")
            return HostResult(call_id=message.call_id, error=error_record(refusal))
        try:
            return await _apart(self.owner.room_call(message))
        except Exception as exc:
            logger.exception("A room call of turn %s failed", message.turn_id)
            return HostResult(call_id=message.call_id, error=error_record(exc))

    def _answer_back(self, question: BrokerMessage, body: str) -> None:
        """Answer *question*, the answer kept for the question asked again."""
        if (handled := self._handled.get(question.ref)) is not None:
            self._handled[question.ref] = (handled[0], body)
        self._post_answer(question, body)

    def _post_answer(self, question: BrokerMessage, body: str) -> None:
        def lost(error: Exception) -> None:
            if not self._closing:
                logger.error("An answer to %s could not be written: %s", question.sender, error)

        # Written until it is: its question waits for it while this process beats.
        self._post(question.sender, "answer", body, "", question.ref, failed=lost, writes=None)

    def post_work(self, node: str, body: str) -> None:
        """Post work after a turn that nothing answers to the owner *node*:
        its reference has the owner handle it once, however many times it
        is written."""

        def lost(error: Exception) -> None:
            logger.warning("Work after a turn could not reach its owner %s: %s", node, error)

        self._post(node, "work", body, "", uuid4().hex, failed=lost, writes=None, within=_WORK_FOR)


class _OwnerPort(RoomOwner):
    """What an executor served here reaches a room owner by outside a turn:
    the owner that submitted the turn the work began in, here or elsewhere,
    waited for while that owner's process beats."""

    def __init__(self, routes: Routes) -> None:
        self._routes = routes

    async def room_call(self, message: RoomCall) -> HostResult:
        routes = self._routes
        node = routes.turn_owner(message.turn_id)
        if node is None:
            return await routes._local_owner.room_call(message)
        body = OwnerMessageEnvelope(message=message).model_dump_json()
        try:
            answer = await routes.ask(
                node, "owner", body, "", alive=lambda: routes._live.owner_alive(node)
            )
        except TurnLinkError as exc:
            return HostResult(call_id=message.call_id, error=error_record(exc))
        return HostResult.model_validate_json(answer)

    async def room_publication(self, message: RoomPublication) -> None:
        routes = self._routes
        node = routes.turn_owner(message.turn_id)
        if node is None:
            await routes._local_owner.room_publication(message)
            return
        routes.post_work(node, OwnerMessageEnvelope(message=message).model_dump_json())


def _crossed(raw: str) -> ChannelControl:
    return ChannelControlEnvelope.model_validate_json(raw).message


async def _apart[T](work: Coroutine[Any, Any, T]) -> T:
    """Run *work* in a context of its own: the owner's side shares no context
    variable with the message's arrival."""
    return await asyncio.get_running_loop().create_task(work, context=contextvars.Context())
