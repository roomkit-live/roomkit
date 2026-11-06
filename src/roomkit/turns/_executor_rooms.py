"""The rooms an executor's channels hold state for (RFC §10.6.2), and the
channel control it applies to them (RFC §10.6.3).

A channel keeps state per room between two turns on its executor. The
executor records which turn, or channel call, left it, and drops it before a
turn or call whose previous one is another: the room's turns ran elsewhere
since, and what they made obsolete must not come back. Dropping is the
channel's own ``on_room_detached``, which lets go of everything it holds for
the room. A record the executor no longer vouches for (it fell out, or a
drop failed) never matches a previous turn, so the state is dropped again.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from roomkit.core.exceptions import ChannelNotRegisteredError
from roomkit.turns._error_codec import error_record
from roomkit.turns._rooms import RoomRecords, stored_room_state
from roomkit.turns.wire import ChannelCall, ChannelResult, RoomAttached, RoomDetached

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.turns.wire import PreviousTurn, TurnRequest

logger = logging.getLogger("roomkit.turns")

_UNVOUCHED = "\x00unvouched"
"""The record of a room whose state a failed drop left: it matches no
previous turn, so the next one drops the state again."""


class ExecutorRooms:
    """Which turn or call left each room's state of a channel on one executor.

    Args:
        executor_id: The executor's id, which a channel call's result names.
        channel: The executor's channel for a channel id (its factory),
            which a record falling out drops the state of.
    """

    def __init__(
        self, executor_id: str, channel: Callable[[str], Awaitable[Channel | None]]
    ) -> None:
        self._executor_id = executor_id
        self._channel = channel
        self._left: RoomRecords[str] = RoomRecords(evicted=self._evicted)
        # The drops of states whose record fell out, still running: a turn or
        # call of that room waits for its own.
        self._dropping: dict[tuple[str, str], asyncio.Task[None]] = {}

    def enter(
        self, channel_id: str, room_id: str, entry_id: str, previous: PreviousTurn | None
    ) -> bool:
        """Record the turn or call *entry_id* as the one leaving the room's
        state of the channel: whether the state held now was left by another
        than *previous*, and must be dropped before it runs."""
        previous_id = previous.turn_id if previous is not None else None
        left = self._left.get(channel_id, room_id)
        self._left.put(channel_id, room_id, entry_id)
        return left != previous_id

    async def prepare(self, channel: Channel, request: TurnRequest, *, stale: bool) -> None:
        """Ready *channel* for *request*'s turn: what it holds for the room
        dropped when *stale*, then what the room keeps for it handed back."""
        room = request.context.room
        stored = stored_room_state(room.metadata, request.channel_id) if room else {}
        await self._ready(channel, request.channel_id, request.room_id, stored, stale=stale)

    async def apply(
        self, channel: Channel | None, message: RoomAttached | RoomDetached | ChannelCall
    ) -> ChannelResult | None:
        """Apply one channel control message to *channel* (``None``: the
        factory has none): a ``channel_call``'s result; nothing for the
        others, whose failure is the channel's to log."""
        if isinstance(message, ChannelCall):
            return await self._call(channel, message)
        if isinstance(message, RoomDetached):
            self._left.pop(message.channel_id, message.room_id)
        if channel is None:
            return None
        try:
            if isinstance(message, RoomAttached):
                await channel.on_room_attached(message.room_id, message.binding)
            else:
                await channel.on_room_detached(message.room_id)
        except Exception:
            logger.exception(
                "Channel %s failed to apply %s for room %s",
                message.channel_id,
                message.type,
                message.room_id,
            )
        return None

    async def _ready(
        self,
        channel: Channel,
        channel_id: str,
        room_id: str,
        stored: dict[str, object],
        *,
        stale: bool,
    ) -> None:
        """Drop what *channel* holds for the room when *stale*, after a drop
        already running for it, then hand back what the room keeps (*stored*)."""
        pending = self._dropping.get((channel_id, room_id))
        if pending is not None:
            await asyncio.wait({pending})
            # A drop that failed left the state it was dropping.
            stale = stale or pending.cancelled() or pending.exception() is not None
        if stale:
            try:
                await channel.on_room_detached(room_id)
            except BaseException:
                # Nothing vouches for the room's state now: drop it next time.
                self._left.put(channel_id, room_id, _UNVOUCHED)
                raise
        channel.restore_room_state(room_id, stored)

    async def _call(self, channel: Channel | None, message: ChannelCall) -> ChannelResult:
        """Run a per-room verb of *channel*, on the room's state as a turn
        would find it: its result, and what the room keeps for the channel
        when the verb changed it."""
        executor_id = self._executor_id
        if channel is None:
            missing = ChannelNotRegisteredError(
                f"No channel {message.channel_id!r} on executor {executor_id}"
            )
            return ChannelResult(error=error_record(missing), executor_id=executor_id)
        room_id, channel_id = message.room_id, message.channel_id
        stale = self.enter(channel_id, room_id, message.call_id, message.previous_turn)
        before: dict[str, object] | None = None
        try:
            await self._ready(channel, channel_id, room_id, message.room_state, stale=stale)
            before = channel.room_state(room_id)
            result = await channel.channel_call(room_id, message.verb, message.arguments)
        except Exception as exc:
            changed = None if before is None else _changed(channel, room_id, before)
            return ChannelResult(
                error=error_record(exc), room_state=changed, executor_id=executor_id
            )
        answer = ChannelResult(
            result=result, room_state=_changed(channel, room_id, before), executor_id=executor_id
        )
        try:
            answer.model_dump_json()
        except (TypeError, ValueError) as exc:
            refusal = TypeError(
                f"The result of {message.verb!r} cannot cross the turn backend: {exc}"
            )
            return ChannelResult(
                error=error_record(refusal), room_state=answer.room_state, executor_id=executor_id
            )
        return answer

    def _evicted(self, channel_id: str, room_id: str) -> None:
        """A record that fell out vouches for nothing: the state it vouched
        for is dropped, in the background."""
        key = (channel_id, room_id)
        task = asyncio.get_running_loop().create_task(self._drop(channel_id, room_id))
        self._dropping[key] = task
        task.add_done_callback(functools.partial(self._dropped, key))

    async def _drop(self, channel_id: str, room_id: str) -> None:
        channel = await self._channel(channel_id)
        if channel is not None:
            await channel.on_room_detached(room_id)

    def _dropped(self, key: tuple[str, str], task: asyncio.Task[None]) -> None:
        if self._dropping.get(key) is task:
            del self._dropping[key]
        if not task.cancelled() and (failure := task.exception()) is not None:
            logger.error(
                "Dropping the state of room %s for channel %s failed",
                key[1],
                key[0],
                exc_info=failure,
            )


def _changed(
    channel: Channel, room_id: str, before: dict[str, object]
) -> dict[str, object] | None:
    """What the room keeps for the channel once a verb ran, when the verb
    changed it from *before*: a copy the channel held apart from the room's
    (a value set on the channel object directly) is not stored by a verb
    that changed nothing."""
    state = channel.room_state(room_id)
    return state if state != before else None
