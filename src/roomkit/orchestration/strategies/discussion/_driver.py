"""The turns of one room's discussion, one at a time (RFC §19.7.5 rules 6, 7, 9, 11, 15).

A turn is a rerun of the event it answers, planned for one agent under the
room lock and run in the room's delivery lane (``RoomKit._plan_discussion_turn``).
The driver waits for it off every lock, then reads what the turn delivered.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.channels._discussion_turn import issue_mark, retire_mark
from roomkit.core.event_router import CHAIN_DEPTH_LIMIT, chain_depth_exceeded, unanswered
from roomkit.models.enums import ChannelType, EventStatus, RoomStatus

from ._notes import AgentLine, turn_notes
from .models import SpeakQueueChange

if TYPE_CHECKING:
    from roomkit.core.lanes import DeliveryCascade
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent

    from ._queue import Entry
    from ._room import DiscussionRoom

logger = logging.getLogger("roomkit.orchestration.discussion")

_REFUSING = frozenset({RoomStatus.CLOSED, RoomStatus.ARCHIVED})


@dataclass
class _Turn:
    entry: Entry
    cascade: DeliveryCascade
    mark: dict[str, Any]


class TurnDriver:
    """Gives a room's turns, the next once the current one has ended."""

    def __init__(self, room: DiscussionRoom) -> None:
        self._room = room
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

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
        """Stop giving turns; a turn running is cut, what it committed kept."""
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.wait([self._task])
        self._task = None

    async def _drive(self) -> None:
        while True:
            # Cleared before the queue is read, so a wake meanwhile is kept.
            self._wake.clear()
            try:
                turn = await self._next_turn()
            except Exception:
                logger.exception(
                    "The discussion of room %s could not give a turn", self._room.room_id
                )
                turn = None
            if turn is None:
                await self._wake.wait()
            else:
                await self._run(turn)

    # -- Giving a turn --

    async def _next_turn(self) -> _Turn | None:
        room = self._room
        if room.state.over:
            return None
        if await self._done():
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
            # host, which may change the queue.
            for entry in stopped:
                await self._record_depth_stop(entry)
        return turn

    async def _pick(self, context: RoomContext) -> tuple[_Turn | None, list[Entry]]:
        """The turn to give, planned, and the turns the depth limit stopped on
        the way. Under the room lock and the state lock."""
        room = self._room
        stopped: list[Entry] = []
        max_turns = room.strategy.max_turns
        if max_turns is not None and room.state.turns_given >= max_turns:
            await self._end()
            return None, stopped
        while True:
            pick = room.state.next_turn(room.max_depth)
            stopped.extend(pick.stopped)
            if pick.entry is None:
                await self._wait_or_idle()
                return None, stopped
            turn = await self._give(pick.entry, context)
            if turn is not None:
                return turn, stopped

    async def _give(self, entry: Entry, context: RoomContext) -> _Turn | None:
        """Plan *entry*'s turn; None, the entry dropped, when it cannot be given
        (the event it answers is gone, or the agent may no longer read it)."""
        room = self._room
        answered = await self._answered(entry)
        mark = issue_mark([self._notes(entry, context)]) if answered is not None else None
        cascade = (
            await room.kit._plan_discussion_turn(
                room.room_id, answered, entry.agent, context, mark=mark, max_depth=room.max_depth
            )
            if answered is not None and mark is not None
            else None
        )
        if cascade is None or mark is None or answered is None:
            if mark is not None:
                retire_mark(mark)
            room.state.entries.remove(entry)
            room.drop_instructions([entry])
            logger.info(
                "Discussion turn of %s in room %s not given: nothing it can answer",
                entry.agent,
                room.room_id,
            )
            return None
        room.state.take(entry)
        await room.save()
        room.fire(SpeakQueueChange.TURN_GIVEN, [entry.agent], answered.id)
        return _Turn(entry, cascade, mark)

    async def _answered(self, entry: Entry) -> RoomEvent | None:
        room = self._room
        if entry.instruction is not None:
            return room.instructions.get(entry.instruction)
        ask = entry.answered()
        if ask is None:
            return None
        event = await room.kit.store.get_event(ask.event_id)
        return event if event is not None and event.room_id == room.room_id else None

    def _notes(self, entry: Entry, context: RoomContext) -> str:
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
        return turn_notes(
            entry.agent,
            others,
            room.people(context),
            [a for a in entry.askers() if a],
            silent_token=room.silent.token,
            instruction=entry.instruction is not None,
        )

    # -- Running it --

    async def _run(self, turn: _Turn) -> None:
        room = self._room
        try:
            await room.kit._finish_cascade(turn.cascade, room.room_id)
        except asyncio.CancelledError:
            await turn.cascade.abandon("discussion_stopped")
            raise
        except Exception:
            logger.exception("Turn of %s in room %s failed", turn.entry.agent, room.room_id)
        finally:
            retire_mark(turn.mark)
            await self._ended(turn)

    async def _ended(self, turn: _Turn) -> None:
        """The turn has ended (rule 6): what it delivered queues whom it names."""
        room = self._room
        agent = turn.entry.agent
        try:
            async with room.lock:
                room.state.ended(agent)
                if turn.entry.instruction is not None:
                    room.instructions.pop(turn.entry.instruction, None)
                context = await room.kit._build_context(room.room_id)
                queued = [
                    named
                    for event in turn.cascade.response_events
                    for named in room.queue_turn_names(agent, event, context)
                ]
                await room.save()
        except Exception:
            logger.exception("The end of %s's turn in room %s was not read", agent, room.room_id)
            return
        room.fire(SpeakQueueChange.TURN_ENDED, [agent])
        if queued:
            room.fire(SpeakQueueChange.QUEUED, list(dict.fromkeys(queued)))

    # -- Stopping, waiting, ending --

    async def _record_depth_stop(self, entry: Entry) -> None:
        """The record of a turn the depth limit stopped, once (RFC §8.3)."""
        room = self._room
        ask = entry.answered()
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

    async def _wait_or_idle(self) -> None:
        """No agent can take a turn: wait for a person when one is owed (rule
        10), else stay idle until an event asks for a turn."""
        state = self._room.state
        if state.waiting or not state.owes_a_person():
            return
        state.waiting = True
        await self._room.save()
        self._room.fire(SpeakQueueChange.WAITING, [])

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
        room.state.over = True
        room.drop_instructions(room.state.drop())
        await room.save()
        room.fire(SpeakQueueChange.OVER, [])
