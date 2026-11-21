"""The two ends of a turn carried by Redis (RFC §10.6.3, §10.6.8).

Each end numbers what it sends and admits what it receives in order, as on
any backend: the messages are the protocol's envelopes, posted to the other
process's inbox in the order they were numbered (one writer per destination),
and routed to the turn's link by its id there. A message Redis refuses breaks
its link: the turn fails at once instead of waiting for what will not come.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from roomkit.core.exceptions import TurnLinkError
from roomkit.turns.base import ExecutorLink, MessageOrder, TurnLink
from roomkit.turns.wire import Accepted, Cancel, ControlEnvelope, End, FrameEnvelope, TurnRequest

if TYPE_CHECKING:
    from roomkit.turns._redis_beat import LiveExecutors
    from roomkit.turns._redis_outbox import Failed, Written
    from roomkit.turns.wire import Control, Frame

logger = logging.getLogger("roomkit.turns")

Post = Callable[..., None]
"""Queue a message for a node's inbox: ``post(node, kind, body, target, ref="",
failed=None, writes=1, within=None, written=None)``; it never waits."""

# Put on a link's queue when its turn is gone: its reader learns why.
_GONE = "\x00gone"
# Put on a link's queue for a message of this side's own: its reader reads it
# in turn, outside the other end's numbering.
_OWN = "\x00own"


class _Inbound:
    """What reached one end of a turn, admitted in order."""

    def __init__(self, envelope: type[FrameEnvelope | ControlEnvelope]) -> None:
        self._envelope = envelope
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._order = MessageOrder()
        self.end_reason = "backend_closed"
        self._own: list[Frame | Control] = []
        self.ended = False
        """Whether the other end's ``end`` came."""

    def deliver(self, body: str) -> None:
        self._queue.put_nowait(body)

    def end(self, reason: str) -> None:
        self.end_reason = reason
        self._queue.put_nowait(_GONE)

    def hand(self, message: Frame | Control) -> None:
        """Have the reader read *message*, of this side's own, in turn."""
        self._own.append(message)
        self._queue.put_nowait(_OWN)

    async def next(self) -> tuple[int, Frame | Control]:
        while True:
            body = await self._queue.get()
            if body == _GONE:
                self._queue.put_nowait(_GONE)
                raise TurnLinkError(self.end_reason)
            if body == _OWN:
                return 0, self._own.pop(0)
            envelope = self._envelope.model_validate_json(body)
            if self._order.admit(envelope.seq):
                if isinstance(envelope.message, End):
                    self.ended = True
                return envelope.seq, envelope.message


class _Outbound:
    """What one end of a turn sends: serialized, numbered and queued in one
    step, so the numbers are the order the other end receives them in; a
    message that cannot cross takes no number."""

    def __init__(
        self,
        turn_id: str,
        envelope: type[FrameEnvelope | ControlEnvelope],
        post: Post,
        broken: Failed,
    ) -> None:
        self._turn_id = turn_id
        self._envelope = envelope
        self._post = post
        self._broken = broken
        self._sent = 0
        self.open = True

    def put(
        self, kind: str, node: str, message: Frame | Control, written: Written | None = None
    ) -> int:
        if not self.open:
            return self._sent
        body = self._envelope.model_validate(
            {"turn_id": self._turn_id, "seq": self._sent + 1, "message": message}
        ).model_dump_json()
        self._sent += 1
        self._post(node, kind, body, self._turn_id, failed=self._broken, written=written)
        return self._sent


class RedisTurnLink(TurnLink):
    """The owner's end of a turn an executor in another process took."""

    def __init__(
        self,
        request: TurnRequest,
        member: str,
        executor_id: str,
        executor_node: str,
        post: Post,
        live: LiveExecutors,
        forget: Callable[[RedisTurnLink], None],
    ) -> None:
        self.request = request
        self.member = member
        """The executor that took the turn, as its process runs it."""
        self.executor_id = executor_id
        self.node = executor_node
        self._frames = _Inbound(FrameEnvelope)
        self._controls = _Outbound(request.turn_id, ControlEnvelope, post, self._broken)
        self._post = post
        self._live = live
        self._forget = forget
        self._beat_seen = live.last_beat(member)
        self._released = False

    @property
    def taken(self) -> bool:
        # The executor's process acknowledged the request (RedisTurnBackend.submit).
        return True

    def deliver(self, body: str) -> None:
        self._frames.deliver(body)

    def end(self, reason: str) -> None:
        """The backend closed: the reader learns the turn is gone."""
        self._frames.end(reason)

    async def receive(self) -> tuple[int, Frame]:
        seq, frame = await self._frames.next()
        return seq, cast("Frame", frame)

    async def send(self, message: Control) -> None:
        self._controls.put("control", self.node, message)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._controls.open = False
        ended = self._frames.ended
        self._frames.end("backend_closed")
        self._forget(self)
        if not ended:
            # Given up before its end (a control Redis refused included): the
            # executor stops it, as one whose owner went.
            turn_id = self.request.turn_id
            self._post(self.node, "released", turn_id, turn_id)

    def beat_heard(self) -> bool:
        """Whether the executor beat again, its process still running the
        turn: a turn whose task ended without its end is lost."""
        beat = self._live.last_beat(self.member)
        if beat is None or beat == self._beat_seen:
            return False
        self._beat_seen = beat
        return self._live.runs(self.node, self.request.turn_id)

    def _broken(self, error: Exception) -> None:
        """A control could not be written: the turn's reader learns it lost
        its executor rather than wait for an answer to it."""
        logger.warning("Turn %s lost its link to its executor: %s", self.request.turn_id, error)
        self._controls.open = False
        self._frames.end("frame_lost")


class RedisExecutorLink(ExecutorLink):
    """The executor's end of a turn a room owner in another process submitted."""

    def __init__(
        self,
        request: TurnRequest,
        owner_node: str,
        post: Post,
        forget: Callable[[RedisExecutorLink], None],
        *,
        end_bound: float,
    ) -> None:
        self.request = request
        self.owner_node = owner_node
        self._accept_written = asyncio.Event()
        self._controls = _Inbound(ControlEnvelope)
        self._frames = _Outbound(request.turn_id, FrameEnvelope, post, self._broken)
        self._forget = forget
        # The turn's end is on its way: its process runs the turn until it is
        # written, so a stall meanwhile does not lose a turn that completed;
        # within end_bound, so a write that hangs does not keep it running.
        self._ending = False
        self._end_bound = end_bound
        self._end_timer: asyncio.TimerHandle | None = None

    def deliver(self, body: str) -> None:
        self._controls.deliver(body)

    def end(self, reason: str) -> None:
        """The owner released the turn, or the backend closed past the
        turn's cut: the turn stops, sending nothing more."""
        self._frames.open = False
        self._controls.end(reason)
        self._accept_written.set()

    def cut(self, reason: str) -> None:
        """The backend closes: the turn is cut as its owner's ``cancel``
        would cut it, and ends, its end still sent."""
        self._controls.hand(Cancel(reason=reason))

    async def receive(self) -> Control:
        _, message = await self._controls.next()
        return cast("Control", message)

    async def send(self, frame: Frame) -> int:
        if isinstance(frame, Accepted):
            # The channel may occupy this event loop on its very first poll.
            # Flush acceptance before giving it the thread (RFC §10.6.8).
            seq = self._frames.put(
                "frame", self.owner_node, frame, written=self._accept_written.set
            )
            if self._frames.open:
                await self._accept_written.wait()
            if not self._frames.open:
                raise TurnLinkError(self._controls.end_reason)
            return seq
        if not isinstance(frame, End):
            return self._frames.put("frame", self.owner_node, frame)
        # The turn's last frame: nothing more crosses.
        self._ending = self._frames.open
        seq = self._frames.put("frame", self.owner_node, frame, written=self._end_written)
        self._frames.open = False
        return seq

    def turn_ended(self) -> None:
        """The turn's task ended: let go of now, or once its end, sent, is
        written or refused, at the latest after ``end_bound``."""
        self._frames.open = False
        if not self._ending:
            self._forget(self)
            return
        loop = asyncio.get_running_loop()
        self._end_timer = loop.call_later(self._end_bound, self._end_unwritten)

    def _end_written(self) -> None:
        self._let_go()

    def _end_unwritten(self) -> None:
        """The turn's end was neither written nor refused in time (a
        connection dropped without a word): the turn stops counting as run
        here, and its owner fails it from its silence."""
        logger.warning(
            "Turn %s: its end was not written within %.1f s", self.request.turn_id, self._end_bound
        )
        self._end_timer = None
        self._let_go()

    def _let_go(self) -> None:
        if self._end_timer is not None:
            self._end_timer.cancel()
            self._end_timer = None
        self._forget(self)

    def _broken(self, error: Exception) -> None:
        """A frame could not be written: the turn is cut, its owner learning
        it from the turn's silence (its process no longer runs it)."""
        logger.warning("Turn %s lost its link to its owner: %s", self.request.turn_id, error)
        self.end("frame_lost")
        if self._ending:
            self._let_go()
