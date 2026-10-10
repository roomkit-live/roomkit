"""The turns of one room's discussion, one at a time (RFC §19.7.5 rules 6, 7, 9, 11, 15).

A turn is a rerun of the event it answers, planned for one agent under the
room lock and run in the room's delivery lane (``RoomKit._plan_discussion_turn``).
The driver waits for it off every lock, then reads what the turn delivered.
Once handed to the lane, a turn is always run to its end or abandoned: a
failure after that point is logged, never allowed to orphan it.
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

from ._notes import AgentLine, TurnKind, turn_notes
from .models import SpeakQueueChange

if TYPE_CHECKING:
    from roomkit.core.lanes import DeliveryCascade
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent

    from ._queue import Entry
    from ._room import DiscussionRoom

logger = logging.getLogger("roomkit.orchestration.discussion")

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


class TurnDriver:
    """Gives a room's turns, the next once the current one has ended."""

    def __init__(self, room: DiscussionRoom) -> None:
        self._room = room
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._turn: _Turn | None = None
        self._stopping = False

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
        if task is None:
            return
        if task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.wait([task])

    async def cut(self, agent: str, reason: str) -> None:
        """Cut *agent*'s turn when it is given but not running yet, which no
        ``Cancel`` reaches: its delivery is abandoned."""
        turn = self._turn
        if turn is not None and turn.entry.agent == agent:
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
                    await self._wait(min(2.0**failures, _RETRY_MAX))
                    continue
                if turn is None:
                    await self._wait(None)
                else:
                    await self._run(turn)
        except asyncio.CancelledError:
            turn = self._turn
            if turn is not None and not turn.started:
                await self._abandon_orphan(turn)
            raise

    async def _wait(self, timeout: float | None) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._wake.wait(), timeout)

    # -- Giving a turn --

    async def _next_turn(self) -> _Turn | None:
        room = self._room
        if room.state.over:
            return None
        done = await self._done()
        if self._stopping or room.closed:
            # done() may uninstall the discussion it is asked about.
            return None
        if done:
            async with room.lock:
                await self._end()
            return None
        async with room.kit._lock_manager.locked(room.room_id):
            context = await room.kit._build_context(room.room_id, reads_history=True)
            if context.room.status in _REFUSING:
                return None
            async with room.lock:
                turn, stopped = await self._pick(context)
            # Committed off the state lock: a commit's announcements reach the
            # host, which may change the queue. A failure here never loses the
            # turn already handed to the lane.
            for entry, regenerated in stopped:
                await self._record_depth_stop(entry, regenerated=regenerated)
        return turn

    async def _pick(self, context: RoomContext) -> tuple[_Turn | None, list[tuple[Entry, bool]]]:
        """The turn to give, planned, and the turns the depth limit stopped on
        the way. Under the room lock and the state lock."""
        room = self._room
        stopped: list[tuple[Entry, bool]] = []
        if self._max_turns_given():
            await self._end()
            return None, stopped
        while True:
            pick = room.state.next_turn(room.max_depth)
            stopped.extend((entry, False) for entry in pick.stopped)
            for entry in pick.dropped:
                if entry.regenerate:
                    stopped.append((entry, True))
                room.drop_instructions([entry])
            if pick.entry is None:
                await self._wait_or_idle(context)
                return None, stopped
            turn = await self._give(pick.entry, context)
            if turn is not None:
                return turn, stopped

    async def _give(self, entry: Entry, context: RoomContext) -> _Turn | None:
        """Plan *entry*'s turn; None, the entry dropped, when it cannot be given
        (the event it answers is gone, or the agent may no longer read it)."""
        room = self._room
        answered = await self._answered(entry)
        if answered is None:
            return self._not_given(entry)
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
            return self._not_given(entry)
        # Handed to the lane: from here the turn is the driver's to finish.
        turn = self._turn = _Turn(entry, cascade, mark)
        room.state.take(entry)
        room.turn_rows = []
        if self._max_turns_given():
            # Over once the last turn is given; that turn ends as it would.
            await self._end()
        await self._save_logged()
        room.fire(SpeakQueueChange.TURN_GIVEN, [entry.agent], answered.id)
        return turn

    def _not_given(self, entry: Entry) -> None:
        room = self._room
        room.state.entries.remove(entry)
        room.drop_instructions([entry])
        logger.info(
            "Discussion turn of %s in room %s not given: nothing it can answer",
            entry.agent,
            room.room_id,
        )

    def _max_turns_given(self) -> bool:
        max_turns = self._room.strategy.max_turns
        return max_turns is not None and self._room.state.turns_given >= max_turns

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
            askers=[f"@{a}" if a in room.agents else quoted(a, _LABEL) for a in entry.askers()],
            silent_token=room.silent.token,
        )

    def _answering(self, answered: RoomEvent, context: RoomContext) -> str:
        """The event a turn answers, as the notes name it: its author's label
        and a quote of what it says."""
        room = self._room
        source = answered.source.channel_id
        if source in room.agents:
            author = f"@{source}"
        else:
            author = quoted(room.person_label(answered, context), _LABEL)
        body = answered.content.body if isinstance(answered.content, TextContent) else ""
        return f"the message from {author}, {quoted(body, _QUOTE)}"

    # -- Running it --

    async def _run(self, turn: _Turn) -> None:
        room = self._room
        turn.started = True
        try:
            await room.kit._finish_cascade(turn.cascade, room.room_id)
        except asyncio.CancelledError:
            await _abandon(turn, "discussion_stopped")
            raise
        except Exception:
            logger.exception("Turn of %s in room %s failed", turn.entry.agent, room.room_id)
        finally:
            retire_mark(turn.mark)
            self._turn = None
            await self._ended(turn)

    async def _ended(self, turn: _Turn) -> None:
        """The turn has ended (rule 6): what it delivered queues whom it names."""
        room = self._room
        agent = turn.entry.agent
        queued: list[str] = []
        try:
            async with room.lock:
                room.state.ended(agent)
                if turn.entry.instruction is not None:
                    room.forget_instruction(turn.entry.instruction)
                rows = {e.id: e for e in (*room.turn_rows, *turn.cascade.response_events)}
                room.turn_rows = []
                if not room.state.over:
                    context = await room.kit._build_context(room.room_id)
                    for event in rows.values():
                        queued.extend(room.queue_turn_names(agent, event, context))
                await room.save()
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

    async def _wait_or_idle(self, context: RoomContext) -> None:
        """No agent can take a turn: wait for a person when one is owed (rule
        10), else stay idle until an event asks for a turn."""
        room = self._room
        state = room.state
        if state.waiting or not state.owes_a_person(has_people=room.has_people(context)):
            return
        state.waiting = True
        await room.save()
        room.fire(SpeakQueueChange.WAITING, [])

    async def _done(self) -> bool:
        done = self._room.strategy.done
        if done is None:
            return False
        try:
            result = done(self._room.room_id)
            return bool(await result) if inspect.isawaitable(result) else bool(result)
        except Exception:
            logger.exception("The discussion's done() failed in room %s", self._room.room_id)
            return False

    async def _end(self) -> None:
        """The discussion is over (rule 15): no further turn, the queue dropped.
        Under the state lock."""
        room = self._room
        if room.state.over:
            return
        room.state.over = True
        room.drop_instructions(room.state.drop())
        await self._save_logged()
        room.fire(SpeakQueueChange.OVER, [])

    async def _save_logged(self) -> None:
        try:
            await self._room.save()
        except Exception:
            logger.exception("The speak queue of room %s was not stored", self._room.room_id)


async def _abandon(turn: _Turn, reason: str) -> None:
    try:
        await turn.cascade.abandon(reason)
    except Exception:
        logger.exception("A discussion turn of %s was not abandoned cleanly", turn.entry.agent)
