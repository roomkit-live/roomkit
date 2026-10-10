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

from roomkit._text import quoted
from roomkit.channels._discussion_turn import issue_mark, retire_mark
from roomkit.core.event_router import CHAIN_DEPTH_LIMIT, chain_depth_exceeded, unanswered
from roomkit.models.enums import ChannelType, EventStatus, RoomStatus
from roomkit.models.event import TextContent

from ._dispatching import Dispatcher
from ._notes import AgentLine, TurnKind, turn_notes
from ._people import person_label
from .models import SpeakQueueChange

if TYPE_CHECKING:
    from roomkit.core.lanes import DeliveryCascade
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
    cascade: DeliveryCascade
    mark: dict[str, Any]
    started: bool = False
    cut: bool = False


class TurnDriver:
    """Gives a room's turns while this process holds the lease."""

    def __init__(self, room: DiscussionRoom) -> None:
        self._room = room
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._turn: _Turn | None = None
        self._stopping = False
        self._seen_version = -1
        self._dispatcher = Dispatcher(room)

    @property
    def speaking(self) -> str | None:
        """The agent whose turn this process runs, if any."""
        return self._turn.entry.agent if self._turn is not None else None

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
        self._wake.set()

    async def stop(self) -> None:
        """Stop giving turns; a turn running is cut, what it committed kept.
        From the driver's own task (a ``done`` that uninstalls), the loop ends
        once that call returns."""
        self._stopping = True
        task, self._task = self._task, None
        if task is None or task is asyncio.current_task():
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
                    await self._wait(POLL_SECONDS)
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
        peek = await shared.read()
        if shared.config is None:
            # Another process uninstalled the discussion.
            await room.kit._forget_discussion(room.room_id, room)
            return None
        if peek.over or shared.held_elsewhere(peek):
            return None
        unchanged = peek.version == self._seen_version
        if unchanged and shared.holds(peek) and not shared.renew_due(peek):
            return None
        done = await self._done() if peek.entries or peek.dispatching else False
        if self._stopping or room.closed:
            # done() may uninstall the discussion it is asked about.
            return None
        async with room.kit._lock_manager.locked(room.room_id):
            context = await room.kit._build_context(room.room_id, reads_history=True)
            if context.room.status in _REFUSING:
                return None
            async with room.editing() as state:
                if not shared.take(state):
                    return None
                pending = self._waiting_decision(state, done=done)
                turn, stopped = None, []
                if pending is None:
                    turn, stopped = await self._pick(state, context, done=done)
            self._seen_version = room.state.version
            # Committed off the queue's edit: a failure here never loses the
            # turn already handed to the lane.
            for entry, regenerated in stopped:
                await self._record_depth_stop(entry, regenerated=regenerated)
        if pending is not None:
            # Off the lock, before any other turn: the policy may take its bound.
            await self._dispatcher.decide(pending)
            self.wake()
        return turn

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
        (the event it answers is gone, or the agent may no longer read it)."""
        room = self._room
        answered = await self._answered(entry)
        if answered is None:
            return self._not_given(state, entry)
        mark = issue_mark([self._notes(entry, answered, context)])
        try:
            cascade = await room.kit._plan_discussion_turn(
                room.room_id, answered, entry.agent, context, mark=mark, max_depth=room.max_depth
            )
        except BaseException:
            retire_mark(mark)
            raise
        if cascade is None:
            retire_mark(mark)
            return self._not_given(state, entry)
        # Handed to the lane: from here the turn is the driver's to finish.
        turn = self._turn = _Turn(entry, cascade, mark)
        state.take(entry)
        room.turn_rows = []
        if self._max_turns_given(state):
            # Over once the last turn is given; that turn ends as it would.
            self._end(state)
        room.fire(SpeakQueueChange.TURN_GIVEN, [entry.agent], answered.id)
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
        return event if event is not None and event.room_id == room.room_id else None

    def _notes(self, entry: Entry, answered: RoomEvent, context: RoomContext) -> str:
        room = self._room
        others = [
            AgentLine(
                handle=agent_id,
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
            entry.agent,
            others,
            room.people(context),
            kind=kind,
            answering=self._answering(answered, context),
            askers=[
                quoted(asker, _LABEL) if person else f"@{asker}"
                for asker, person in entry.askers()
            ],
            silent_token=room.silent.token,
        )

    def _answering(self, answered: RoomEvent, context: RoomContext) -> str:
        """The event a turn answers, as the notes name it: its author's label
        and a quote of what it says."""
        if answered.source.channel_id in self._room.agents:
            author = f"@{answered.source.channel_id}"
        else:
            author = quoted(person_label(answered, context), _LABEL)
        body = answered.content.body if isinstance(answered.content, TextContent) else ""
        return f"the message from {author}, {quoted(body, _QUOTE)}"

    # -- Running it --

    async def _run(self, turn: _Turn) -> None:
        room = self._room
        turn.started = True
        watch = asyncio.get_running_loop().create_task(self._watch(turn))
        try:
            await room.kit._finish_cascade(turn.cascade, room.room_id)
        except asyncio.CancelledError:
            await _abandon(turn, "discussion_stopped")
            raise
        except Exception:
            logger.exception("Turn of %s in room %s failed", turn.entry.agent, room.room_id)
        finally:
            watch.cancel()
            await asyncio.wait([watch])
            retire_mark(turn.mark)
            self._turn = None
            await self._ended(turn)

    async def _watch(self, turn: _Turn) -> None:
        """While *turn* runs: its agent set to listen elsewhere cuts it, and
        the lease is renewed before it expires."""
        room = self._room
        while True:
            await asyncio.sleep(POLL_SECONDS)
            try:
                state = await room.shared.read()
                if turn.entry.agent in state.listening and not turn.cut:
                    turn.cut = True
                    await room.cut(turn.entry.agent)
                if room.shared.renew_due(state):
                    async with room.editing() as editing:
                        room.shared.take(editing)
            except Exception:
                logger.warning("Watching the turn in room %s failed", room.room_id, exc_info=True)

    async def _ended(self, turn: _Turn) -> None:
        """The turn has ended (rule 6): what it delivered queues whom it names."""
        room = self._room
        agent = turn.entry.agent
        queued: list[str] = []
        try:
            async with room.editing() as state:
                state.ended(agent)
                if turn.entry.instruction is not None:
                    room.forget_instruction(turn.entry.instruction, state)
                rows = {e.id: e for e in (*room.turn_rows, *turn.cascade.response_events)}
                room.turn_rows = []
                if not state.over:
                    context = await room.kit._build_context(room.room_id)
                    for event in rows.values():
                        queued.extend(room.queue_turn_names(state, agent, event, context))
        except Exception:
            logger.exception("The end of %s's turn in room %s was not read", agent, room.room_id)
        room.fire(SpeakQueueChange.TURN_ENDED, [agent])
        if queued:
            room.fire(SpeakQueueChange.QUEUED, list(dict.fromkeys(queued)))

    async def _abandon_orphan(self, turn: _Turn) -> None:
        """A turn handed to the lane that the driver will not run: abandoned,
        its mark retired, its end read."""
        await _abandon(turn, "discussion_stopped")
        retire_mark(turn.mark)
        self._turn = None
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


async def _abandon(turn: _Turn, reason: str) -> None:
    try:
        await turn.cascade.abandon(reason)
    except Exception:
        logger.exception("A discussion turn of %s was not abandoned cleanly", turn.entry.agent)
