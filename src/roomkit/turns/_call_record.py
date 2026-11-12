"""The turn's response-metadata record across an owner-served call (RFC §10.6.5).

The record is the executor's. A call made while the turn runs carries it as it
stands; the owner runs the handler on a copy of its own, and answers with what
the handler changed, measured against what the call carried; the executor
applies that to the turn's record before the call returns. Against what the
call carried, not the live record: a key a concurrent call wrote meanwhile is
neither seen as removed nor overwritten entry by entry, so two calls served at
once that each add entries to one mapping keep both.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from roomkit.tools.context import current_response_metadata
from roomkit.turns._codec import record_payload
from roomkit.turns.wire import RecordChanges


def carried_record() -> dict[str, Any] | None:
    """The turn's record as a call made now carries it; ``None`` outside a turn."""
    record = current_response_metadata()
    return None if record is None else record_payload(record)


def changes_of(carried: Mapping[str, Any], worked: Mapping[str, Any]) -> RecordChanges | None:
    """What a handler that worked on a copy of *carried* left changed of it;
    ``None`` when nothing."""
    now = record_payload(worked)
    written = {key: value for key, value in now.items() if carried.get(key, _ABSENT) != value}
    removed = [key for key in carried if key not in now]
    if not written and not removed:
        return None
    return RecordChanges(written=written, removed=removed)


def apply_changes(
    record: MutableMapping[str, Any], carried: Mapping[str, Any], changes: RecordChanges | None
) -> None:
    """Apply what a handler changed of *carried* to the turn's *record*."""
    if changes is None:
        return
    for key in changes.removed:
        record.pop(key, None)
    for key, value in changes.written.items():
        record[key] = _merged(record.get(key), carried.get(key), value)


def _merged(current: Any, carried: Any, value: Any) -> Any:
    """The value a key written as *value* takes in the turn's record: a
    mapping the handler changed, where the call carried a mapping or nothing
    and the record holds a mapping, takes the change entry by entry; any other
    value replaces what the record holds."""
    if not isinstance(value, dict) or not isinstance(current, Mapping):
        return value
    if carried is not None and not isinstance(carried, dict):
        return value
    base = carried or {}
    merged = dict(current)
    merged.update(
        {entry: item for entry, item in value.items() if base.get(entry, _ABSENT) != item}
    )
    for entry in base:
        if entry not in value:
            merged.pop(entry, None)
    return merged


_ABSENT = object()
