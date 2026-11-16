"""What the room owner publishes of what a turn's channel published on its
executor (RFC §10.6.2, §10.6.3): during the turn, from its ``ephemeral`` and
``framework_event`` frames; after it, from a ``room_publication``. Only what
names the turn's room and channel, and only the framework events a turn's
executor emits, are published; anything else is dropped. An answer's own text
is published only for a turn whose reader commits it as a delegated task's
trace."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from roomkit.core._reentrance import within_turn
from roomkit.core.visibility import answers_reach
from roomkit.models.enums import Visibility
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.turns.base import response_depth
from roomkit.turns.wire import Ephemeral

if TYPE_CHECKING:
    import contextvars
    from datetime import datetime

    from roomkit.turns.base import OwnerEvents
    from roomkit.turns.wire import FrameworkEventFrame, TurnRequest

logger = logging.getLogger("roomkit.turns")

TURN_FRAMEWORK_EVENTS = frozenset(
    {
        "tool_call",
        "before_tool_use",
        "user_input_required",
        "ai_response",
        "before_ai_generation",
        "hook_timeout",
    }
)
"""The framework events a turn's executor emits (RFC §8.2): the owner emits no
other a ``framework_event`` frame names."""


def names_its_room(
    room_id: str, channel_id: str, named_room: str | None, named: str | None
) -> bool:
    """Whether what a turn of *channel_id* in *room_id* published names that
    room and channel (or none)."""
    return named_room in (None, room_id) and named in (None, channel_id)


async def publish_frame(
    events: OwnerEvents,
    room_id: str,
    channel_id: str,
    frame: Ephemeral | FrameworkEventFrame,
    *,
    label: str,
    unscoped: bool,
    traced: bool = False,
    room_created_at: datetime | None = None,
) -> None:
    """Publish *frame*, published by a turn of *channel_id* in *room_id*
    (*label* names it in the log), as that channel in that room: the room of
    creation mark *room_created_at*, which an ephemeral carries (RFC §8.4,
    §17.2).

    An ephemeral is published only when the turn's answer reaches the whole
    room (RFC §8.4): its trigger is *unscoped*, and the channel's binding, as
    the room holds it at this publish, lets it (``events.reaches_room``). Its
    subscribers are not bound channels, so a turn whose answer the room
    narrows publishes none, from the moment the room narrows it.

    The answer's own text (``TEXT_DELTA``) is published only for a turn
    *traced*: one whose reader commits its rows as a delegated task's trace,
    crossing no hook (RFC §6.4, §23.3 step 5). ``BEFORE_BROADCAST`` judges
    the message a room stores, never this stream, so the text of an answer a
    hook judges is never published, nor any after its turn ended."""
    if isinstance(frame, Ephemeral):
        if _carries_answer_text(frame) and not traced:
            logger.debug("%s: its answer is not a trace, its text not published", label)
            return
        if not (unscoped and await events.reaches_room(room_id, channel_id)):
            logger.debug("%s: its answer does not reach the room, ephemeral not published", label)
            return
        event = EphemeralEvent.from_dict(frame.event)
        event.room_created_at = room_created_at
        if not names_its_room(room_id, channel_id, event.room_id, event.channel_id) or (
            frame.channel != f"room:{room_id}"
        ):
            logger.warning("%s published outside its room: dropped", label)
            return
        await events.publish(frame.channel, event)
        return
    if frame.event_type not in TURN_FRAMEWORK_EVENTS or not names_its_room(
        room_id, channel_id, frame.room_id, frame.channel_id
    ):
        logger.warning("%s emitted %s outside its room or kind: dropped", label, frame.event_type)
        return
    await events.emit(
        frame.event_type,
        room_id=room_id,
        channel_id=channel_id,
        event_id=frame.event_id,
        data=frame.data,
    )


def _carries_answer_text(frame: Ephemeral) -> bool:
    """Whether *frame* carries the text of the turn's answer."""
    return frame.event.get("type") == EphemeralEventType.TEXT_DELTA.value


class TurnPublisher:
    """What one turn published on its executor, published here in order by a
    task of its own: a handler that waits on the turn (an answer to an input
    request, a close) never stalls the frames it waits for."""

    def __init__(self, events: OwnerEvents | None, request: TurnRequest) -> None:
        self._events = events
        self._request = request
        self._unscoped = answers_reach(request.event) == Visibility.ALL
        self.traced = False
        """Whether the turn's reader commits it as a delegated task's trace:
        only then is its answer's text published (RFC §6.4)."""
        self._queue: asyncio.Queue[Ephemeral | FrameworkEventFrame | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    def publish_later(
        self, frame: Ephemeral | FrameworkEventFrame, waiting: contextvars.Context | None
    ) -> None:
        """Queue *frame*, the task started on the first in a copy of
        *waiting*, the context of the task waiting on the turn (RFC §10.6.5)."""
        if self._events is None:
            return
        if self._task is None:
            context = waiting.copy() if waiting is not None else None
            self._task = asyncio.get_running_loop().create_task(self._run(), context=context)
        self._queue.put_nowait(frame)

    def close(self) -> None:
        """What was queued is still published; nothing is added after."""
        self._queue.put_nowait(None)

    async def _run(self) -> None:
        while (frame := await self._queue.get()) is not None:
            await self._publish(frame)

    async def _publish(self, frame: Ephemeral | FrameworkEventFrame) -> None:
        """Publish on the owner what the turn published on its executor, as
        the turn's channel in the turn's room (RFC §10.6.2)."""
        assert self._events is not None
        request = self._request
        try:
            # Published as the turn: the handlers of a framework event it
            # emitted run at least as deep as its response, its turn over
            # or not, each for as long as it runs (RFC §8.3).
            with within_turn(response_depth(request)):
                await publish_frame(
                    self._events,
                    request.room_id,
                    request.channel_id,
                    frame,
                    label=f"Turn {request.turn_id}",
                    unscoped=self._unscoped,
                    traced=self.traced,
                    room_created_at=request.context.room.created_at,
                )
        except Exception:
            # What the turn published never stops the turn.
            logger.exception("Turn %s: a %s could not be published", request.turn_id, frame.type)
