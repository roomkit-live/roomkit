"""The open human-input requests of a room owner's turns (RFC §10.6.5).

A request is answerable from its creation: an answer or a rejection the kit
is given for an id it does not know yet is held until its request arrives, or
for the hold bound (``input_hold_timeout``, 60 seconds by default), at most
:data:`HELD_KEPT` at once; and a request answered once is not answered again.
The outcome goes to the turn's executor as an ``input_answer``; a request
whose turn ends is withdrawn.

A request belongs to the turn that opened it: an id already open, or already
settled, is not opened again, and only its turn closes it. The executor, where
the tool waits, settles the request first come first served (an answer its
channel's handler received, a hook's block, its bound, a release) and closes
it here; an answer given here goes to the executor, which says whether it
settled the request (``input_taken``), and the kit's door answers that.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.turns.wire import InputAnswer

if TYPE_CHECKING:
    from roomkit.turns._owner_turn import OwnerTurn
    from roomkit.turns.wire import InputRequest

logger = logging.getLogger("roomkit.turns")

SETTLED_KEPT = 1024
"""How many settled request ids the owner remembers, refusing a second answer."""

HELD_KEPT = 1024
"""How many early answers the owner holds at once, the oldest let go first."""


@dataclass(slots=True)
class _Held:
    answer: InputAnswer
    until: float
    verdict: asyncio.Future[bool]


class OwnerInputs:
    """The open requests of a client's turns, and the answers held for ones
    not open yet."""

    def __init__(self, hold: float) -> None:
        self._hold = hold
        # Each open request with the turn that opened it.
        self._open: dict[str, tuple[OwnerTurn, InputRequest]] = {}
        self._held: dict[str, _Held] = {}
        # Answers sent to a request's executor, awaiting whether they settled it.
        self._awaiting: dict[str, tuple[OwnerTurn, asyncio.Future[bool]]] = {}
        self._settled: OrderedDict[str, None] = OrderedDict()
        self._sending: set[asyncio.Task[Any]] = set()

    def opened(self, turn: OwnerTurn, request: InputRequest) -> None:
        """*turn* opened a request: an answer held for it is sent now."""
        self._prune()
        request_id = request.request_id
        if request_id in self._open or request_id in self._settled:
            # Another turn's, or settled: a request is opened once.
            return
        held = self._held.pop(request_id, None)
        if held is not None:
            self._deliver(turn, held.answer, held.verdict)
            return
        self._open[request_id] = (turn, request)

    def closed(self, turn: OwnerTurn, request_id: str) -> None:
        """*turn*'s request settled where its tool waits."""
        entry = self._open.get(request_id)
        if entry is not None and entry[0] is turn:
            del self._open[request_id]
            self._remember(request_id)

    def taken(self, turn: OwnerTurn, request_id: str, taken: bool) -> None:
        """Whether the answer sent to *turn*'s request settled it."""
        awaited = self._awaiting.get(request_id)
        if awaited is None or awaited[0] is not turn:
            return
        del self._awaiting[request_id]
        _decide(awaited[1], taken)

    def turn_ended(self, turn: OwnerTurn) -> None:
        """*turn* is over: its open requests are withdrawn, and an answer
        still awaiting its verdict was not taken."""
        for request_id in [rid for rid, (opener, _) in self._open.items() if opener is turn]:
            del self._open[request_id]
            self._remember(request_id)
        for request_id, (awaited_turn, verdict) in list(self._awaiting.items()):
            if awaited_turn is turn:
                del self._awaiting[request_id]
                _decide(verdict, False)

    def request(self, request_id: str) -> tuple[OwnerTurn, InputRequest] | None:
        """The open request *request_id* and the turn that opened it; ``None``
        once it is answered, settled or withdrawn, or before it opens."""
        return self._open.get(request_id)

    async def answer(self, request_id: str, value: str) -> bool:
        """Answer *request_id* with *value*: whether the waiting tool reads it."""
        return await self._record(
            InputAnswer(request_id=request_id, outcome="answered", value=value)
        )

    async def reject(self, request_id: str, reason: str) -> bool:
        """Reject *request_id*: whether the waiting tool reads the rejection."""
        return await self._record(
            InputAnswer(request_id=request_id, outcome="rejected", reason=reason)
        )

    async def _record(self, answer: InputAnswer) -> bool:
        """Send *answer* to the request's executor and wait for whether it
        settled the request; for a request not open yet, hold it until it
        opens, at most the hold bound. ``False`` for a request settled, or
        answered, already."""
        self._prune()
        request_id = answer.request_id
        if request_id in self._settled or request_id in self._held:
            return False
        verdict: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        entry = self._open.pop(request_id, None)
        if entry is not None:
            self._deliver(entry[0], answer, verdict)
            return await verdict
        self._hold_early(answer, verdict)
        try:
            return await asyncio.wait_for(asyncio.shield(verdict), self._hold)
        except TimeoutError:
            if self._held.pop(request_id, None) is not None:
                _decide(verdict, False)
                return False
            # Opened just now: its verdict follows.
            return await verdict

    def _hold_early(self, answer: InputAnswer, verdict: asyncio.Future[bool]) -> None:
        """Hold *answer* for its request, the oldest let go past the cap."""
        until = asyncio.get_running_loop().time() + self._hold
        self._held[answer.request_id] = _Held(answer, until, verdict)
        while len(self._held) > HELD_KEPT:
            _decide(self._held.pop(next(iter(self._held))).verdict, False)

    def _deliver(
        self, turn: OwnerTurn, answer: InputAnswer, verdict: asyncio.Future[bool]
    ) -> None:
        self._remember(answer.request_id)
        self._awaiting[answer.request_id] = (turn, verdict)
        task = asyncio.get_running_loop().create_task(_sent(turn, answer, verdict))
        self._sending.add(task)
        task.add_done_callback(self._sending.discard)

    def _remember(self, request_id: str) -> None:
        self._settled[request_id] = None
        while len(self._settled) > SETTLED_KEPT:
            self._settled.popitem(last=False)

    def _prune(self) -> None:
        """Let go of the answers held past their bound: none was taken."""
        now = asyncio.get_running_loop().time()
        for request_id in [rid for rid, held in self._held.items() if held.until <= now]:
            _decide(self._held.pop(request_id).verdict, False)


def _decide(verdict: asyncio.Future[bool], taken: bool) -> None:
    if not verdict.done():
        verdict.set_result(taken)


async def _sent(turn: OwnerTurn, answer: InputAnswer, verdict: asyncio.Future[bool]) -> None:
    try:
        await turn.send_control(answer)
    except Exception:
        logger.warning(
            "The answer to human-input request %s could not reach its turn",
            answer.request_id,
            exc_info=True,
        )
        _decide(verdict, False)
