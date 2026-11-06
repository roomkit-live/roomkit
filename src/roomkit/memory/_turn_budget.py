"""What the history may take of a memory's window for the turn being read."""

from __future__ import annotations

from roomkit.memory.base import MemoryResult
from roomkit.memory.token_estimator import estimate_notes_tokens, history_budget
from roomkit.models.event import RoomEvent
from roomkit.tools.context import current_turn_footprint


def turn_history_budget(
    *,
    max_context_tokens: int,
    safety_margin_ratio: float,
    current_event: RoomEvent,
    reserved_tokens: int = 0,
    carried: MemoryResult | None = None,
) -> int:
    """:func:`~roomkit.memory.token_estimator.history_budget` for the turn
    whose memory reads the room (RFC §20).

    The turn's input the AI channel measured (``current_turn_footprint()``) is
    reserved, or *reserved_tokens* when larger: a floor on that same input,
    never added to it. The reply is reserved once, the larger of the safety
    margin and the turn's reply budget. *carried* is what a wrapped memory
    returned: its messages and its notes occupy the window and no trimming
    cuts them. Outside a turn nothing is measured, and the window less the
    margin, *reserved_tokens*, *carried* and the current event is left.
    """
    footprint = current_turn_footprint()
    measured = footprint.input_tokens if footprint is not None else 0
    notes = estimate_notes_tokens(carried.notes) if carried is not None else 0
    return history_budget(
        max_context_tokens=max_context_tokens,
        safety_margin_ratio=safety_margin_ratio,
        reserved_tokens=max(reserved_tokens, measured) + notes,
        messages=carried.messages if carried is not None else None,
        current_event=current_event,
        reply_tokens=footprint.reply_tokens if footprint is not None else 0,
    )
