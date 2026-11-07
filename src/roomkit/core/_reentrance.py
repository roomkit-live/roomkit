"""What a call into a room is to the room's own work (RFC §9.5.1 Reentrance,
§8.3, §10.5, §10.1 steps 16 and 18).

The execution context of a call tells it, for the room it names:

- under the room's lock, in the section a pass holds (a locked hook, code
  under the lock): it is *nested*, adopted by that pass, drawing on the
  reentry budget of the cascade whose section holds the lock, and what its
  own hooks inject joins that section's queue (§9.4);
- in the room's admission order (an off-lock hook's body, a message's parse):
  it is adopted *beside* the event that work is for, takes no ticket and
  draws on that event's budget;
- in the room's delivery work (a delivery set's deliveries and turns, and the
  body of an ASYNC hook the cascade fires once an event's delivery set has
  run, ``AFTER_BROADCAST`` or a change's mutation trigger): it is a new
  *arrival*, with its ticket and its ``process_timeout``, drawing on the
  budget of the cascade that delivers, so a transport whose delivery calls
  into its room on every event stops at the cap, and so does a hook that
  posts after every event; and so does a message it sends that creates a
  room, routed or named, that work's own (``opening``), while a call
  into another existing room has a budget of its own;
- elsewhere: a new arrival with a budget of its own.

A call the room's work made is charged one pass as it starts, and one made
from an ASYNC hook's body one of the cascade's hook calls, a counter of the
same size kept apart from its passes. A direct change made from a hook's
body (``update_event()``, ``delete_event()``, ``regenerate_response()``)
draws a hook call too, refused with ``ReentryLoopCapError`` past the cap,
and a direct change's triggers run on the budget of the work it is made in
(:meth:`RoomWork.direct_change`). A call made while a turn or a tool call
runs, into whatever room, is at least as deep as the turn's response or the
call's (§8.3, §10.6), and one made while an ASYNC hook's body runs at least
one deeper than the event the hook runs for; only the budget is the room's.

A frame stands for its work only while that work runs: a task the work
spawns inherits the frame and finds it inactive once the work is over, as a
room lock's lease is (:mod:`roomkit.core.locks`). A turn's depth floor is a
frame too, and a tool call's ends with the call: work a turn, a delivery or
a hook's body began starts fresh once that work is over, at depth 0 and on a
budget of its own. An ASYNC hook's body runs at least as deep as its trigger
fired, for as long as the body runs: the floors are lent to the body, not to
what it spawns (:func:`floors_lent_while_it_runs`).
"""

from __future__ import annotations

import contextvars
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from roomkit.core.exceptions import ReentrantCallError, ReentryLoopCapError
from roomkit.core.locks import _has_room_lock, lent_call
from roomkit.tools.context import _current_loop, _live_turn_chain_depth

if TYPE_CHECKING:
    from roomkit.core._queued_injections import QueuedInjections
    from roomkit.core.locks import RoomLockManager
    from roomkit.core.locks import _Call as _LendingCall
    from roomkit.tools.context import _ToolLoopContext


REENTRY_LOOP_CAP = "reentry_loop_cap"
"""The ``blocked_by`` of what a cascade's budget no longer admits (RFC §8.3)."""


class ReentryBudget:
    """The passes a cascade may still run (RFC §8.3): its responses, chained
    streams and injected events, and the entries the room's work adopted or
    sent into it; and, counted apart on a counter of the same size, the
    calls the bodies of its ASYNC hooks may still make into the room
    (§9.5.1), so that a hook's notices cost its agents no answer and their
    answers block no notice. Shared by reference between the cascades that
    draw on it."""

    __slots__ = ("_hook_calls", "_passes")

    def __init__(self, passes: int) -> None:
        self._passes = passes
        self._hook_calls = passes

    def take(self) -> bool:
        """Take one pass; ``False`` once the passes are spent."""
        if self._passes <= 0:
            return False
        self._passes -= 1
        return True

    def take_hook_call(self) -> bool:
        """Take one call of an ASYNC hook's body; ``False`` once they are spent."""
        if self._hook_calls <= 0:
            return False
        self._hook_calls -= 1
        return True


class Placement(Enum):
    """Where a call into a room stands against the room's work."""

    NESTED = "nested"
    """Under the room's lock: in the section that holds it."""
    BESIDE = "beside"
    """In the room's admission order: beside the event its caller works for."""
    ARRIVAL = "arrival"
    """A new arrival: its own ticket and ``process_timeout``."""


@dataclass(frozen=True, slots=True)
class Entry:
    """What a call into a room is to the room's work."""

    placement: Placement
    budget: ReentryBudget
    charged: bool
    """Whether the room's work made the call: it draws one pass as it starts."""
    queued: QueuedInjections | None = None
    """Nested in a pass's section: that section's queue of injected events."""
    hook_body: bool = False
    """Whether an ASYNC hook's body made the call: it draws one of the
    budget's hook calls rather than one of its passes."""
    follows: _LendingCall | None = None
    """Nested by a task of a hook's body while the body runs: that body's
    call, whose section waits for this one and ends it with itself
    (:mod:`roomkit.core._nested_calls`)."""
    deadline: float | None = None
    """Nested in a pass's section: that section's deadline, the
    ``process_timeout`` of its event, when it has one."""

    @property
    def adopted(self) -> bool:
        """Whether the call takes no ticket and runs every hook under the lock."""
        return self.placement is not Placement.ARRIVAL

    def draw(self) -> bool:
        """Take what the call draws as it starts, a hook call or a pass;
        ``False`` once the budget holds none."""
        return self.budget.take_hook_call() if self.hook_body else self.budget.take()


class _Kind(Enum):
    SECTION = "section"
    ADMITTED = "admitted"
    DELIVERING = "delivering"


@dataclass(eq=False, slots=True)
class _Frame:
    kind: _Kind
    room_id: str
    budget: ReentryBudget
    queued: QueuedInjections | None = None
    active: bool = True
    hook_body: bool = False
    deadline: float | None = None


@dataclass(eq=False, slots=True)
class _Call:
    running: bool = True


@dataclass(frozen=True, slots=True)
class _LentFrame:
    """An admission order lent to a call (a hook's body) while the call runs:
    a task the call spawned keeps a copy of the context, and stands in the
    order no more once the call is over, though the work goes on."""

    frame: _Frame | _LentFrame
    call: _Call

    @property
    def kind(self) -> _Kind:
        return self.frame.kind

    @property
    def room_id(self) -> str:
        return self.frame.room_id

    @property
    def budget(self) -> ReentryBudget:
        return self.frame.budget

    @property
    def queued(self) -> QueuedInjections | None:
        return self.frame.queued

    @property
    def hook_body(self) -> bool:
        return self.frame.hook_body

    @property
    def deadline(self) -> float | None:
        return self.frame.deadline

    @property
    def active(self) -> bool:
        return self.call.running and self.frame.active


_frames: contextvars.ContextVar[tuple[_Frame | _LentFrame, ...]] = contextvars.ContextVar(
    "_roomkit_room_work_frames", default=()
)


@dataclass(eq=False, slots=True)
class _Turn:
    response_depth: int
    active: bool = True


@dataclass(frozen=True, slots=True)
class _LentTurn:
    """A turn's floor lent to a call (an ASYNC hook's body) while the call
    runs: a task the call spawned keeps a copy of the context, and has the
    floor no more once the call is over, though the turn goes on."""

    turn: _Turn | _LentTurn
    call: _Call

    @property
    def response_depth(self) -> int:
        return self.turn.response_depth

    @property
    def active(self) -> bool:
        return self.call.running and self.turn.active


# The turns the context runs within (RFC §8.3), each the floor of a call into
# any room while it runs. Made inert once the turn is over, as a frame is: a
# task the turn spawned that calls after it is work of its own, at depth 0
# (the maintainer's decision, RMK-930), and a result delivered later on the
# turn's behalf names its depth itself (a delegation's, §23.3).
_turns: contextvars.ContextVar[tuple[_Turn | _LentTurn, ...]] = contextvars.ContextVar(
    "_roomkit_turns", default=()
)

# The loop contexts of the turns and tool calls the body of a hook run
# asynchronously was lent the depth of, each with the body's call: once the
# body is over, a task it spawned has that depth no more, though the turn or
# the call still runs.
_loops_lent: contextvars.ContextVar[tuple[tuple[_ToolLoopContext, _Call], ...]] = (
    contextvars.ContextVar("_roomkit_loops_lent", default=())
)


@contextmanager
def _framed(frame: _Frame) -> Generator[None, None, None]:
    token = _frames.set((*_frames.get(), frame))
    try:
        yield
    finally:
        # Copies the work's tasks hold share this frame: over, it is inert.
        frame.active = False
        _frames.reset(token)


@contextmanager
def orders_lent_while_it_runs() -> Generator[None, None, None]:
    """Run a call (a hook's body) with the admission orders this context
    stands in lent to it while it runs (RFC §9.5.1 Reentrance): a task it
    spawns, awaited or not, stands in them while it runs; once it is over,
    that task stands in none, though the work goes on (an off-lock check's
    next hook), and its calls into the room are new arrivals. The room
    locks it holds are lent the same way (:func:`~roomkit.core.locks.lent_while_it_runs`)."""
    frames = _frames.get()
    if not any(frame.kind is _Kind.ADMITTED for frame in frames):
        yield
        return
    call = _Call()
    lent = tuple(
        _LentFrame(frame, call) if frame.kind is _Kind.ADMITTED else frame for frame in frames
    )
    token = _frames.set(lent)
    try:
        yield
    finally:
        call.running = False
        _frames.reset(token)


def section(
    room_id: str, budget: ReentryBudget, queued: QueuedInjections, deadline: float | None = None
) -> AbstractContextManager[None]:
    """The locked section of a pass of *room_id*, on the cascade's *budget*,
    the events its hooks inject queued on *queued*, its decision due by
    *deadline* when it has one (RFC §13.6)."""
    return _framed(_Frame(_Kind.SECTION, room_id, budget, queued, deadline=deadline))


def admission_order(room_id: str, budget: ReentryBudget) -> AbstractContextManager[None]:
    """Work standing in *room_id*'s admission order for an event drawing on
    *budget*: an off-lock hook's body, a message's parse (RFC §9.5.1)."""
    return _framed(_Frame(_Kind.ADMITTED, room_id, budget))


def delivering(room_id: str, budget: ReentryBudget) -> AbstractContextManager[None]:
    """The deliveries and turns of a delivery set of *room_id*, run for the
    cascade drawing on *budget*."""
    return _framed(_Frame(_Kind.DELIVERING, room_id, budget))


@contextmanager
def after_delivery(room_id: str, budget: ReentryBudget, depth: int) -> Generator[None, None, None]:
    """The body of one ASYNC hook a cascade of *room_id* drawing on *budget*
    fires once an event's delivery set has run, for an event *depth* deep:
    its ``AFTER_BROADCAST`` hooks (RFC §10.1 step 16) and a change's mutation
    trigger (§10.3). It is the cascade's delivery work while it runs: what it
    sends into the room draws on the budget's hook calls, and a call it makes
    into any room is at least one deeper than that event, as a call made
    within a turn is at least as deep as its response (§8.3), so a hook that
    asks an agent again after each of its answers ends at the depth limit. A
    task it starts that calls once the body has returned is work of its
    own."""
    frame = _Frame(_Kind.DELIVERING, room_id, budget, hook_body=True)
    with _framed(frame), within_turn(depth + 1):
        yield


def work_budget(room_id: str) -> ReentryBudget | None:
    """The budget of the room's work this context runs in: the innermost
    frame of *room_id* whose work still runs (a pass's section while the
    context holds the lock, an event's admission work, a delivery set's
    work, an ASYNC hook's body); ``None`` outside the room's work. The
    observers of a SYNC trigger fired from that work run on it (RFC
    §9.5.1, :meth:`~roomkit.core.hooks.HookEngine.run_observers`)."""
    frame = _innermost(room_id, locked=_has_room_lock(room_id))
    return frame.budget if frame is not None else None


def charge_direct_change(change: Entry, room_id: str, call: str) -> None:
    """Draw the hook call *change*, a direct change of *room_id* made from
    an ASYNC hook's body, costs before it writes (RFC §9.5.1, §10.3);
    nothing for one made elsewhere (:meth:`RoomWork.direct_change`).

    Raises:
        ReentryLoopCapError: the budget the body runs on holds no hook call;
            nothing is written.
    """
    if not change.charged or change.draw():
        return
    raise ReentryLoopCapError(
        f"{call}() from an ASYNC hook's body in room {room_id} is past the reentry budget "
        "that body runs on: nothing written"
    )


@contextmanager
def within_turn(response_depth: int) -> Generator[None, None, None]:
    """A turn whose response is *response_depth* deep: what the context calls
    into a room while it runs is at least that deep (RFC §8.3, §10.6), the
    work it spawns included until the turn is over."""
    turn = _Turn(response_depth)
    token = _turns.set((*_turns.get(), turn))
    try:
        yield
    finally:
        # Copies the turn's tasks hold share it: over, it is inert.
        turn.active = False
        _turns.reset(token)


@contextmanager
def floors_lent_while_it_runs(floor: int) -> Generator[None, None, None]:
    """Run a call (the body of a hook run asynchronously) at least *floor*
    deep, the depth its trigger fired at, for as long as it runs, the work
    that fired it over or not (RFC §8.3, §9.5.1): a hook that tells an
    agent again from its body ends at the depth limit. The turns and the
    tool call the context runs within are lent to it the same way. A task
    it spawns, awaited or not, has these floors while the call runs; once
    the call is over, that task has none of them, though the turn or the
    tool call goes on: its calls are at the depth of their own event, as
    RMK-930 decided for work that outlives what began it (a delegation's
    result, a quality gate's relaunch). Only the depth is lent here: the
    budget a call draws on is the room's work it runs in (:func:`work_budget`)."""
    call = _Call()
    lent = tuple(_LentTurn(turn, call) for turn in _turns.get())
    turns_token = _turns.set((*lent, _LentTurn(_Turn(floor), call)))
    loop = _current_loop()
    loops = _loops_lent.get() if loop is None else (*_loops_lent.get(), (loop, call))
    loops_token = _loops_lent.set(loops)
    try:
        yield
    finally:
        call.running = False
        _loops_lent.reset(loops_token)
        _turns.reset(turns_token)


def _live_loop_depth() -> int:
    """The depth of the turn or tool call this context runs in while it
    runs, unless it was lent to a hook's body that is over."""
    loop = _current_loop()
    if loop is not None and any(
        lent is loop and not call.running for lent, call in _loops_lent.get()
    ):
        return 0
    return _live_turn_chain_depth()


def answers_past_limit(chain_depth: int, max_chain_depth: int) -> bool:
    """Whether an answer to an event *chain_depth* deep would reach
    *max_chain_depth*, so that no agent is asked to make it (RFC §8.3)."""
    return chain_depth + 1 >= max_chain_depth


def chain_floor() -> int:
    """The least chain depth of a call made in this context, into whatever
    room: the response depth of the deepest turn it runs within that still
    runs, or of the tool call it runs in while that runs (realtime and
    conference calls, a call the owner serves, a tool handler), or the depth
    an ASYNC hook's body runs at (:func:`floors_lent_while_it_runs`); 0
    outside them and once they are over."""
    turns = (turn.response_depth for turn in _turns.get() if turn.active)
    return max(max(turns, default=0), _live_loop_depth())


class RoomWork:
    """Tells a call into a room what it is to the room's work, from the
    frames of its execution context and the room locks it holds.

    *lock_manager* is the kit's room lock; *passes* the reentry budget of a
    cascade of its own (``10 × max_chain_depth``, RFC §8.3).
    """

    def __init__(self, lock_manager: RoomLockManager, passes: int) -> None:
        self._lock_manager = lock_manager
        self._passes = passes

    def budget(self) -> ReentryBudget:
        """A cascade's budget of its own."""
        return ReentryBudget(self._passes)

    def entry(self, room_id: str) -> Entry:
        """What a call into *room_id* made in this context is (RFC §9.5.1)."""
        locked = _has_room_lock(room_id, self._lock_manager)
        frame = _innermost(room_id, locked=locked)
        if locked:
            if frame is None:
                # Code under the lock outside any of the room's work (a host
                # serialising its changes): nested, on a budget of its own.
                return Entry(Placement.NESTED, self.budget(), False)
            section = _innermost(room_id, locked=True, kind=_Kind.SECTION)
            return Entry(
                Placement.NESTED,
                frame.budget,
                True,
                section.queued if section is not None else None,
                frame.hook_body,
                follows=lent_call(room_id, self._lock_manager),
                deadline=section.deadline if section is not None else None,
            )
        if frame is not None and frame.kind is _Kind.ADMITTED:
            return Entry(Placement.BESIDE, frame.budget, True)
        if frame is not None and frame.kind is _Kind.DELIVERING:
            return Entry(Placement.ARRIVAL, frame.budget, True, hook_body=frame.hook_body)
        return Entry(Placement.ARRIVAL, self.budget(), False)

    def opening(self) -> Entry | None:
        """What a message this context sends is to the work it opens a room
        for, routed or named (RFC §9.5.1, gap G-070): an arrival charged on
        the budget of the cascade whose delivery work (a delivery, a turn,
        an ASYNC hook's body) still runs in the innermost such frame, since
        the room it opens is that work's own; ``None`` outside every
        delivery, for a budget of its own."""
        for frame in reversed(_frames.get()):
            if frame.active and frame.kind is _Kind.DELIVERING:
                return Entry(Placement.ARRIVAL, frame.budget, True, hook_body=frame.hook_body)
        return None

    def direct_change(self, room_id: str) -> Entry:
        """What a direct change of *room_id* made in this context is (RFC
        §10.3, §9.5.1): ``update_event()``, ``delete_event()``,
        ``regenerate_response()``. Its triggers run on the budget of the
        room's work it is made in, the innermost frame's, or on one of its
        own outside that work; it draws one hook call only from an ASYNC
        hook's body (``charged``)."""
        frame = _innermost(room_id, locked=_has_room_lock(room_id, self._lock_manager))
        if frame is None:
            return Entry(Placement.ARRIVAL, self.budget(), False)
        return Entry(Placement.ARRIVAL, frame.budget, frame.hook_body, hook_body=frame.hook_body)

    def refuse_routed(self, room_id: str) -> None:
        """Refuse a message routed into *room_id* from under its lock or from
        its admission order (RFC §9.5.1, §10.1 step 2): known only once
        routed, the room cannot adopt it, and committed it would come ahead
        of the event its caller works for, outside every order.

        Raises:
            ReentrantCallError: the context holds the room's lock or stands in
                its admission order.
        """
        locked = _has_room_lock(room_id, self._lock_manager)
        frame = _innermost(room_id, locked=locked)
        if locked or (frame is not None and frame.kind is _Kind.ADMITTED):
            raise ReentrantCallError(
                f"A message routed into room {room_id} from its own work (under its lock, "
                "in an off-lock hook or a parse) cannot be adopted: name the room to send "
                "into it"
            )


def _innermost(
    room_id: str, *, locked: bool, kind: _Kind | None = None
) -> _Frame | _LentFrame | None:
    """The innermost frame of *room_id* whose work still runs, of *kind* when
    given; a section counts only for a context that holds the room's lock
    (*locked*), which a task started in a context holding none does not."""
    for frame in reversed(_frames.get()):
        if not frame.active or frame.room_id != room_id:
            continue
        if frame.kind is _Kind.SECTION and not locked:
            continue
        if kind is None or frame.kind is kind:
            return frame
    return None
