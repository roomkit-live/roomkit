"""The turns of one room's discussion, one at a time (RFC §19.7.5 rules 6, 7, 9, 11, 15, 16).

A process that installed the discussion runs a driver; the one holding the
room's lease gives the turns. A turn is a rerun of the event it answers,
planned for one agent under the room lock and run in the room's delivery lane
(``RoomKit._plan_discussion_turn``). The driver reads the stored queue again
at least once a second, so a turn another process queued, or an agent another
process set to listen, reaches it; it renews the lease while a turn runs. Once
handed to the lane, a turn is always run to its end or abandoned.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit._media import content_text
from roomkit._text import quoted
from roomkit.channels._discussion_turn import issue_mark, retire_mark
from roomkit.core._changes import withdrawn
from roomkit.core.event_router import CHAIN_DEPTH_LIMIT, chain_depth_exceeded, unanswered
from roomkit.core.locks import _has_room_lock
from roomkit.core.task_utils import acting_tasks, hold_task
from roomkit.models.enums import ChannelType, EventStatus, RoomStatus

from ._dispatching import Dispatcher
from ._notes import AgentLine, TurnKind, turn_notes
from ._people import person_label
from .models import SpeakQueueChange

if TYPE_CHECKING:
    from roomkit.core.lanes import DeliveryCascade, DeliveryPlan
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent

    from ._queue import Entry, Pending, SpeakQueueState
    from ._room import DiscussionRoom

logger = logging.getLogger("roomkit.orchestration.discussion")

POLL_SECONDS = 1.0
"""How often the driver reads the stored queue again (rule 16)."""

_REFUSING = frozenset({RoomStatus.CLOSED, RoomStatus.ARCHIVED})
_QUOTE = 160
_LABEL = 48
"""A person's label in the notes is quoted, never given as the runtime's words."""
_RETRY_MAX = 30.0


@dataclass
class _Turn:
    entry: Entry
    cascade: DeliveryCascade | None
    """The turn's run in the room's lane, which it enters once the queue that
    gives it is stored; None for a posted result's turn."""
    mark: dict[str, Any]
    plan: DeliveryPlan | None = None
    """What enters the lane; None for a posted result's turn."""
    started: bool = False
    cut: bool = False
    posted: RoomEvent | None = None
    """A task's result the turn posts as the agent's message (rule 19)."""
    listening: bool = False
    """Its agent only listens, as last read: a person named it (rule 12), and
    only setting the state while the turn runs cuts it."""


class TurnDriver:
    """Gives a room's turns while this process holds the lease."""

    def __init__(self, room: DiscussionRoom) -> None:
        self._room = room
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._turn: _Turn | None = None
        self._stopping = False
        self._drain = True
        """Whether a turn cut by a stop waits for its delivery to unwind."""
        self._seen_version = -1
        self.settled = False
        """Whether its last look found nothing to do, and no wake came since:
        a test's way to tell a driver at rest from one about to give a turn."""
        self._dispatcher = Dispatcher(room)

    @property
    def speaking(self) -> str | None:
        """The agent whose turn this process runs, if any."""
        return self._turn.entry.agent if self._turn is not None else None

    @property
    def posting(self) -> RoomEvent | None:
        """The task's result the turn running posts, if it is a post turn."""
        return self._turn.posted if self._turn is not None else None

    def start(self) -> None:
        # A context of its own: the room lock is reentrant per context, and the
        # driver must not inherit one its installer held.
        self._task = asyncio.get_running_loop().create_task(
            self._drive(),
            context=contextvars.Context(),
            name=f"roomkit-discussion-{self._room.room_id}",
        )
        self._wake.set()

    def wake(self) -> None:
        self.settled = False
        self._wake.set()

    async def stop(self) -> None:
        """Stop giving turns; a turn running is cut, what it committed kept.
        From the driver's own task (a ``done`` that uninstalls), the loop ends
        once that call returns. A caller holding the room lock does not wait
        for the cut turn to unwind, which may take that lock (RFC §19.7)."""
        self._stopping = True
        self._drain = not _has_room_lock(self._room.room_id)
        task, self._task = self._task, None
        if task is None or task in acting_tasks():
            return
        task.cancel()
        await asyncio.wait([task])

    async def cut(self, agent: str, reason: str) -> None:
        """Cut *agent*'s turn when it is given but not running yet, which no
        ``Cancel`` reaches: its delivery is abandoned."""
        turn = self._turn
        if turn is not None and turn.entry.agent == agent:
            turn.cut = True
            await _abandon(turn, reason)

    async def _drive(self) -> None:
        failures = 0
        try:
            while not self._stopping:
                # Cleared before the queue is read, so a wake meanwhile is kept.
                self._wake.clear()
                try:
                    turn = await self._next_turn()
                    failures = 0
                except Exception:
                    failures += 1
                    logger.exception(
                        "The discussion of room %s could not give a turn", self._room.room_id
                    )
                    await self._wait(min(2.0 ** (failures - 1), _RETRY_MAX))
                    continue
                if turn is None:
                    self.settled = not self._wake.is_set()
                    await self._wait(POLL_SECONDS)
                    self.settled = False
                else:
                    await self._run(turn)
        except asyncio.CancelledError:
            turn = self._turn
            if turn is not None and not turn.started:
                await self._abandon_orphan(turn)
            raise

    async def _wait(self, timeout: float) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._wake.wait(), timeout)

    # -- Giving a turn --

    async def _next_turn(self) -> _Turn | None:
        room = self._room
        shared = room.shared
        installs = room.installs
        peek = await room.read()
        if shared.config is None or room.forgotten:
            # Another process uninstalled the discussion, whether or not one
            # was installed again since: this process leaves it (rule 16).
            await room.kit._forget_discussion(room.room_id, room, installs=installs)
            return None
        if not await self._gives_turns(peek):
            return None
        if peek.over or shared.held_elsewhere(peek):
            return None
        unchanged = peek.version == self._seen_version
        if unchanged and shared.holds(peek) and not shared.renew_due(peek):
            return None
        # Asked off the room lock, of a queue that holds something: done() may
        # uninstall the discussion it is asked about.
        done = await self._done() if _holds_work(peek) else None
        if self._stopping or room.closed:
            return None
        try:
            return await self._planned_turn(done)
        except Exception:
            self._drop_turn_not_stored()
            raise

    def _drop_turn_not_stored(self) -> None:
        """The queue that gave a turn was not stored: no turn was given, and
        its entry, still stored, gives it on the next look."""
        turn, self._turn = self._turn, None
        if turn is not None:
            retire_mark(turn.mark)

    async def _planned_turn(self, done: bool | None) -> _Turn | None:
        """The turn to give, planned under the room lock, once the dispatch
        policy decided the first message waiting for it, off the lock;
        *done* is what the discussion's ``done`` answered, ``None`` when the
        queue peeked held nothing to ask it about. One that holds something
        now is planned on the next look, once ``done`` was asked (rule 15)."""
        room = self._room
        shared = room.shared
        async with room.kit._lock_manager.locked(room.room_id):
            context = await room.kit._build_context(room.room_id, reads_history=True)
            if context.room.status in _REFUSING:
                return None
            async with room.editing() as state:
                # Uninstalled since the peek: no turn, the next look leaves it.
                if shared.config is None or not shared.take(state):
                    return None
                if done is None and _holds_work(state):
                    self.wake()
                    return None
                done = bool(done)
                pending = self._waiting_decision(state, done=done)
                turn, stopped = None, []
                if pending is None:
                    turn, stopped = await self._pick(state, context, done=done)
            self._seen_version = room.state.version
            if turn is not None and turn.plan is not None:
                turn.cascade = room.kit._enter_discussion_turn(
                    room.room_id, turn.plan, context, max_depth=room.max_depth
                )
            # Committed off the queue's edit: a failure here never loses the
            # turn already handed to the lane.
            for entry, regenerated in stopped:
                await self._record_depth_stop(entry, regenerated=regenerated)
        if pending is not None:
            # Off the lock, before any other turn: the policy may take its bound.
            await self._dispatcher.decide(pending)
            self.wake()
        return turn

    async def _gives_turns(self, peek: SpeakQueueState) -> bool:
        """Whether this process gives the turns of the discussion the room
        stores: one whose host installed another gives up the lease and the
        turns only it could give, queued before (rules 16 and 19)."""
        room = self._room
        if room.drives:
            return True
        held = any(e.holder == room.shared.me for e in peek.entries)
        if held or room.shared.holds(peek):
            await room.give_up()
        return False

    def _waiting_decision(self, state: SpeakQueueState, *, done: bool) -> Pending | None:
        """The first message waiting for the dispatch policy, unless the
        discussion is over or ends now (its queue dropped with them)."""
        if not state.dispatching or state.over or done or self._max_turns_given(state):
            return None
        return state.dispatching[0]

    async def _pick(
        self, state: SpeakQueueState, context: RoomContext, *, done: bool
    ) -> tuple[_Turn | None, list[tuple[Entry, bool]]]:
        """The turn to give, planned, and the turns the depth limit stopped on
        the way. While editing *state*, under the room lock."""
        room = self._room
        stopped: list[tuple[Entry, bool]] = []
        if state.over:
            return None, stopped
        if done or self._max_turns_given(state):
            self._end(state)
            return None, stopped
        while True:
            pick = state.next_turn(room.max_depth)
            stopped.extend((entry, False) for entry in pick.stopped)
            for entry in pick.dropped:
                if entry.regenerate:
                    stopped.append((entry, True))
                room.drop_instructions([entry], keep=state)
            entry = pick.entry
            if entry is None:
                self._wait_or_idle(state, context)
                return None, stopped
            if self._hand_over(state, entry):
                return None, stopped
            if entry not in state.entries:
                continue
            turn = await self._give(state, entry, context)
            if turn is not None:
                return turn, stopped

    def _hand_over(self, state: SpeakQueueState, entry: Entry) -> bool:
        """An instruction whose text another process holds: the lease goes to
        that process, once; whether it went. Back here, the instruction is
        dropped: its process did not take its turn (rule 16)."""
        room = self._room
        if entry.instruction is None or entry.holder in (None, room.shared.me):
            return False
        if entry.handed:
            state.entries.remove(entry)
            room.drop_instructions([entry], keep=state)
            return False
        entry.handed = True
        room.shared.grant(state, entry.holder)
        return True

    async def _give(
        self, state: SpeakQueueState, entry: Entry, context: RoomContext
    ) -> _Turn | None:
        """Plan *entry*'s turn; None, the entry dropped, when it cannot be given
        (the event it answers is gone, the agent may no longer read it, or it
        is offline)."""
        room = self._room
        answered = await self._answered(entry)
        if answered is None:
            return self._not_given(state, entry)
        if entry.post:
            return self._handed(state, _Turn(entry, None, {}, posted=answered), answered.id)
        if await room.kit._known_offline(room.room_id, [entry.agent], room.organization_id):
            return self._skipped(state, entry, answered.id)
        answered = _at_ask_depth(answered, entry)
        mark = issue_mark([self._notes(entry, answered, context)])
        try:
            plan = room.kit._plan_discussion_turn(
                answered, entry.agent, context, mark=mark, max_depth=room.max_depth
            )
        except BaseException:
            retire_mark(mark)
            raise
        if plan is None:
            retire_mark(mark)
            return self._not_given(state, entry)
        # From here the turn is the driver's to finish.
        return self._handed(state, _Turn(entry, None, mark, plan=plan), answered.id)

    def _handed(self, state: SpeakQueueState, turn: _Turn, answered_id: str) -> _Turn:
        """*turn* given: taken from the queue, the driver's to finish."""
        room = self._room
        self._turn = turn
        turn.listening = turn.entry.agent in state.listening
        state.take(turn.entry)
        room.turn_rows = []
        if self._max_turns_given(state):
            # Over once the last turn is given; that turn ends as it would.
            self._end(state)
        room.fire(SpeakQueueChange.TURN_GIVEN, [turn.entry.agent], answered_id)
        return turn

    def _not_given(self, state: SpeakQueueState, entry: Entry) -> None:
        room = self._room
        state.entries.remove(entry)
        room.drop_instructions([entry], keep=state)
        logger.info(
            "Discussion turn of %s in room %s not given: nothing it can answer",
            entry.agent,
            room.room_id,
        )

    def _skipped(self, state: SpeakQueueState, entry: Entry, answered_id: str) -> None:
        """*entry*'s agent is offline, no executor serving it (RFC §10.6.7):
        its turn is dropped and said so, never given to fail; an instruction
        it was to take is dropped after."""
        room = self._room
        hooks = room.kit._turns.room_hook_names(room.room_id)
        logger.info(
            "Discussion turn of %s in room %s skipped: no executor online serves it%s",
            entry.agent,
            room.room_id,
            f" with the room's hooks {hooks}" if hooks else "",
        )
        state.entries.remove(entry)
        room.fire(SpeakQueueChange.TURN_SKIPPED, [entry.agent], answered_id)
        room.drop_instructions([entry], keep=state)

    def _max_turns_given(self, state: SpeakQueueState) -> bool:
        max_turns = self._room.config.max_turns
        return max_turns is not None and state.turns_given >= max_turns

    async def _answered(self, entry: Entry) -> RoomEvent | None:
        room = self._room
        if entry.instruction is not None:
            return room.instructions.get(entry.instruction)
        ask = entry.answered()
        if ask is None:
            return None
        event = await room.kit.store.get_event(ask.event_id)
        # A message deleted while the turn waited is gone too (RFC §10.3).
        if event is None or event.room_id != room.room_id or withdrawn(event):
            return None
        return event

    def _notes(self, entry: Entry, answered: RoomEvent, context: RoomContext) -> str:
        room = self._room
        others = [
            AgentLine(
                handle=room.handle(agent_id),
                name=getattr(agent, "name", None),
                role=getattr(agent, "role", None),
                description=getattr(agent, "description", None),
            )
            for agent_id, agent in room.agents.items()
            if agent_id != entry.agent
        ]
        kind = (
            TurnKind.INSTRUCTION
            if entry.instruction is not None
            else TurnKind.REGENERATE
            if entry.regenerate
            else TurnKind.ANSWER
        )
        return turn_notes(
            room.handle(entry.agent),
            others,
            room.people(context),
            kind=kind,
            answering=self._answering(answered, context),
            askers=[
                quoted(asker, _LABEL) if person else room.label(asker)
                for asker, person in entry.askers()
            ],
            silent_token=room.silent.token,
            rules=room.config.rules,
        )

    def _answering(self, answered: RoomEvent, context: RoomContext) -> str:
        """The event a turn answers, as the notes name it: its author's label
        and a quote of what it says."""
        if answered.source.channel_id in self._room.agents:
            author = self._room.label(answered.source.channel_id)
        else:
            author = quoted(person_label(answered, context), _LABEL)
        body = content_text(answered.content) or ""
        return f"the message from {author}, {quoted(body, _QUOTE)}"

    # -- Running it --

    async def _run(self, turn: _Turn) -> None:
        if turn.cascade is None:
            await self._post(turn)
            return
        room = self._room
        turn.started = True
        watch = asyncio.get_running_loop().create_task(self._watch(turn))
        try:
            await room.kit._finish_cascade(turn.cascade, room.room_id)
        except asyncio.CancelledError:
            await _abandon(turn, "discussion_stopped", drain=self._drain)
            raise
        except Exception:
            logger.exception("Turn of %s in room %s failed", turn.entry.agent, room.room_id)
        finally:
            watch.cancel()
            await asyncio.wait([watch])
            retire_mark(turn.mark)
            self._turn = None
            await self._ended(turn)

    async def _post(self, turn: _Turn) -> None:
        """A posted result's turn (rule 19): the task's result committed as
        the agent's message, with no generation, off the room lock."""
        room = self._room
        turn.started = True
        # Held by the kit, apart from the driver: a stop while it commits does
        # not lose the result (a closing kit cuts it); a failure is logged there.
        posting = (
            hold_task(room.kit, room.posts.post(turn.posted), context=contextvars.Context())
            if turn.posted is not None
            else None
        )
        try:
            if posting is not None:
                with contextlib.suppress(Exception):
                    await asyncio.shield(posting)
        finally:
            self._turn = None
            await self._ended(turn)

    async def _watch(self, turn: _Turn) -> None:
        """While *turn* runs: what another process stored cuts it (the
        discussion uninstalled, its agent set to listen or removed), and the
        lease is renewed before it expires."""
        room = self._room
        while True:
            await asyncio.sleep(POLL_SECONDS)
            try:
                installs = room.installs
                state = await room.read()
                # A read begun before an install here tells nothing of it.
                reason = self._cut_reason(turn, state, outdated=room.installs != installs)
                turn.listening = turn.entry.agent in state.listening
                if reason is not None and not turn.cut:
                    turn.cut = True
                    if reason == "discussion_uninstalled":
                        # Gone for every process: its end is recorded nowhere (rule 1).
                        room.forgotten = True
                    await room.cut(turn.entry.agent, reason)
                if room.shared.renew_due(state):
                    async with room.editing() as editing:
                        room.shared.take(editing)
            except Exception:
                logger.warning("Watching the turn in room %s failed", room.room_id, exc_info=True)

    def _cut_reason(self, turn: _Turn, state: SpeakQueueState, *, outdated: bool) -> str | None:
        """Why the stored queue, just read as *state*, cuts *turn*, if it does;
        a read *outdated* by an install here since tells no uninstall."""
        room = self._room
        agent = turn.entry.agent
        if room.shared.config is None:
            return None if outdated else "discussion_uninstalled"
        if agent in state.listening and not turn.listening:
            return "listen_only"
        if agent not in room.agents:
            return "agent_removed"
        return None

    async def _ended(self, turn: _Turn) -> None:
        """The turn has ended (rule 6): announced, once its end is recorded,
        unless the discussion is being forgotten and its queue goes (rule 1)."""
        room = self._room
        agent = turn.entry.agent
        # Its queue is about to go: recording the end would wait on the room
        # lock an uninstalling host may hold, for nothing.
        queued = [] if room.forgotten else await self._record_end(turn)
        room.fire(SpeakQueueChange.TURN_ENDED, [agent])
        if queued:
            room.fire(SpeakQueueChange.QUEUED, list(dict.fromkeys(queued)))

    async def _record_end(self, turn: _Turn) -> list[str]:
        """Record the end of *turn* in the stored queue, where what it
        delivered queues whom it names; the agents it queued."""
        room = self._room
        agent = turn.entry.agent
        queued: list[str] = []
        try:
            async with room.editing() as state:
                if room.shared.config is None:
                    # Uninstalled elsewhere meanwhile: no queue to write back.
                    return []
                state.ended(agent)
                if turn.entry.instruction is not None:
                    room.forget_instruction(turn.entry.instruction, state)
                responses = turn.cascade.response_events if turn.cascade is not None else ()
                rows = {e.id: e for e in (*room.turn_rows, *responses)}
                room.turn_rows = []
                # A posted result's names ask no one: its worker wrote it
                # outside the room (rule 19).
                if not state.over and turn.posted is None:
                    context = await room.kit._build_context(room.room_id)
                    for event in rows.values():
                        queued.extend(room.queue_turn_names(state, agent, event, context))
        except Exception:
            logger.exception("The end of %s's turn in room %s was not read", agent, room.room_id)
            return []
        return queued

    async def _abandon_orphan(self, turn: _Turn) -> None:
        """A turn given that the driver will not run: abandoned if it entered
        the lane, its mark retired, its end read."""
        await _abandon(turn, "discussion_stopped", drain=self._drain)
        retire_mark(turn.mark)
        self._turn = None
        if turn.posted is not None:
            # A result its turn never posted: posted all the same (rule 19).
            self._room.posts.post_later(turn.posted)
        await self._ended(turn)

    # -- Stopping, waiting, ending --

    async def _record_depth_stop(self, entry: Entry, *, regenerated: bool) -> None:
        """The record of a turn the depth limit stopped, once (RFC §8.3)."""
        room = self._room
        ask = entry.answered()
        try:
            event = await room.kit.store.get_event(ask.event_id) if ask is not None else None
            if event is None or event.room_id != room.room_id:
                return
            agent = room.agents.get(entry.agent)
            channel_type = agent.channel_type if agent is not None else ChannelType.AI
            record = unanswered(event, entry.agent, channel_type).model_copy(
                update={"status": EventStatus.BLOCKED, "blocked_by": CHAIN_DEPTH_LIMIT}
            )
            await room.kit._commit_blocked_response(
                room.room_id, record, max_chain_depth=room.max_depth
            )
            await room.kit.store.add_observation(chain_depth_exceeded(record, room.max_depth))
        except Exception:
            logger.exception(
                "The depth stop of %s's %s in room %s was not recorded",
                entry.agent,
                "regenerated answer" if regenerated else "turn",
                room.room_id,
            )

    def _wait_or_idle(self, state: SpeakQueueState, context: RoomContext) -> None:
        """No agent can take a turn: wait for a person when one is owed (rule
        10), else stay idle until an event asks for a turn."""
        room = self._room
        if state.waiting or not state.owes_a_person(has_people=room.has_people(state, context)):
            return
        state.waiting = True
        room.fire(SpeakQueueChange.WAITING, [])

    async def _done(self) -> bool:
        strategy = self._room.strategy
        done = strategy.done if strategy is not None else None
        if done is None:
            return False
        try:
            result = done(self._room.room_id)
            return bool(await result) if inspect.isawaitable(result) else bool(result)
        except Exception:
            logger.exception("The discussion's done() failed in room %s", self._room.room_id)
            return False

    def _end(self, state: SpeakQueueState) -> None:
        """The discussion is over (rule 15): no further turn, the queue dropped."""
        room = self._room
        if state.over:
            return
        state.over = True
        room.drop_instructions(state.drop(), keep=state)
        room.fire(SpeakQueueChange.OVER, [])


def _holds_work(state: SpeakQueueState) -> bool:
    """Whether *state* holds a turn to give or a message waiting for the
    dispatch policy: what the discussion's ``done`` is asked about."""
    return bool(state.entries or state.dispatching)


async def _abandon(turn: _Turn, reason: str, *, drain: bool = True) -> None:
    """Abandon *turn*'s delivery: waiting for it to unwind when *drain*, else
    only cancelled, its caller holding the room lock it may take."""
    if turn.cascade is None:
        return
    if not drain:
        turn.cascade.cancel(reason)
        return
    try:
        await turn.cascade.abandon(reason)
    except Exception:
        logger.exception("A discussion turn of %s was not abandoned cleanly", turn.entry.agent)


def _at_ask_depth(answered: RoomEvent, entry: Entry) -> RoomEvent:
    """*answered* at the depth its ask stands at when that is deeper than its
    own: a regeneration a hook asked for answers one deeper than the answer
    that fired the hook (RFC §8.3, §19.7.5 rule 11)."""
    ask = entry.answered()
    if ask is None or ask.depth <= answered.chain_depth:
        return answered
    return answered.model_copy(update={"chain_depth": ask.depth})
