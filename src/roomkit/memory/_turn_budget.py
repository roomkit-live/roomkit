"""What the history may take of a memory's window for the turn being read."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence

from roomkit.memory.base import MemoryResult
from roomkit.memory.token_estimator import (
    estimate_event_tokens,
    estimate_notes_tokens,
    history_budget,
)
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


def history_event_costs(events: Sequence[RoomEvent]) -> list[int]:
    """What each of *events*, a history oldest first, costs the window with
    what the turn's history replays with it (RFC §20): its own tokens, and
    each earlier round of tool calls and instruction the AI channel measured
    (``current_turn_footprint().replayed``), charged to the event it reads
    with, so that a memory keeps or cuts a turn whole.

    A replayed piece reads with its turn's first event after it (the turn's
    answer, which an instruction and its rounds read before), else after the
    last event before it; one charged to no event (its turn and its place
    out of *events*) is not replayed, and costs nothing.
    """
    costs = [estimate_event_tokens(event) for event in events]
    footprint = current_turn_footprint()
    if footprint is None or not footprint.replayed or not events:
        return costs
    indexes = [event.index for event in events]
    for index, turn, tokens in footprint.replayed:
        at = _charged(events, indexes, index, turn)
        if at is not None:
            costs[at] += tokens
    return costs


def _charged(
    events: Sequence[RoomEvent], indexes: list[int], index: int, turn: str | None
) -> int | None:
    """The position among *events* of the one a replayed piece at timeline
    *index* of *turn* reads with, ``None`` for none."""
    after = bisect_left(indexes, index)
    if turn is not None:
        for at in range(after, len(events)):
            if events[at].correlation_id == turn:
                return at
    return after - 1 if after > 0 else None
