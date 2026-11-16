"""The room owner's side of what an executor sends after a turn ended (RFC
§10.6.3, §10.6.5).

The owner accepts what comes from the executor of the last turn it sent for
the channel and room, as its record of that turn (``previous_turn``) names
it: a host verb only for that very turn, whose work it is; a publication for
any turn of that executor, which still holds the room's state (an external
agent's session update handled under the turn that prompted it, which ended
since). Anything else is refused, an owner that restarted included, until
its next turn.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roomkit.core._reentrance import within_turn
from roomkit.turns._error_codec import error_record
from roomkit.turns._owner_publish import publish_frame
from roomkit.turns.wire import HostResult

if TYPE_CHECKING:
    from roomkit.turns._rooms import RoomRecords
    from roomkit.turns.base import OwnerEvents, OwnerHost
    from roomkit.turns.wire import PreviousTurn, RoomCall, RoomPublication

logger = logging.getLogger("roomkit.turns")


async def answer_room_call(
    previous: RoomRecords[PreviousTurn],
    host: OwnerHost | None,
    message: RoomCall,
) -> HostResult:
    """Run *message*'s verb on *host* if the room's last turn of its channel
    is the one it names: its result, or its refusal. The verb is work its
    turn began, which starts fresh once the turn is over (RFC §8.3): no
    depth of the turn floors what it sends."""
    refusal = _refusal(previous, message, by_turn=True)
    if refusal is None and host is None:
        refusal = "this owner serves no host verb"
    if refusal is not None:
        logger.warning("A %s after turn %s is refused: %s", message.verb, message.turn_id, refusal)
        return HostResult(call_id=message.call_id, error=error_record(PermissionError(refusal)))
    assert host is not None
    held = previous.get(message.channel_id, message.room_id)
    # The room its turn ran in: one created again under its id since, by
    # another process, serves it not (RFC §17.2).
    message._room_created_at = None if held is None else held._room_created_at
    try:
        result = await host.call_between_turns(message)
    except Exception as exc:
        return HostResult(call_id=message.call_id, error=error_record(exc))
    return HostResult(call_id=message.call_id, result=result)


async def publish_between_turns(
    previous: RoomRecords[PreviousTurn], events: OwnerEvents | None, message: RoomPublication
) -> None:
    """Publish what *message* carries if its executor holds the room's state
    of its channel; drop it otherwise."""
    refusal = _refusal(previous, message, by_turn=False)
    if refusal is not None:
        logger.warning("A publication after turn %s is dropped: %s", message.turn_id, refusal)
        return
    if events is None:
        return
    try:
        held = previous.get(message.channel_id, message.room_id)
        # Still the turn's: the handlers of a framework event its work emits
        # after it run at least as deep as its response (RFC §8.3).
        depth = held._response_depth(message.turn_id) if held is not None else 0
        with within_turn(depth):
            await publish_frame(
                events,
                message.room_id,
                message.channel_id,
                message.frame,
                label=f"The work of turn {message.turn_id}",
                unscoped=held is not None and held._unscoped(message.turn_id),
                room_created_at=None if held is None else held._room_created_at,
            )
    except Exception:
        logger.exception("A publication after turn %s could not be published", message.turn_id)


def _refusal(
    previous: RoomRecords[PreviousTurn], message: RoomCall | RoomPublication, *, by_turn: bool
) -> str | None:
    held = previous.get(message.channel_id, message.room_id)
    if held is None or held.executor_id != message.executor_id:
        return (
            f"executor {message.executor_id} did not take the last turn of "
            f"{message.channel_id!r} in room {message.room_id!r}"
        )
    if by_turn and held.turn_id != message.turn_id:
        return f"turn {message.turn_id} is no longer the room's last of {message.channel_id!r}"
    return None
