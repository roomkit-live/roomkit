"""What a run produced, normalized as the suite compares it.

roomkit-specs ``conformance/README.md`` ("Normalization") defines it: a room
the run created is named ``$r1``, ``$r2``... in the order of its
``room_created``; an index is read without the lifecycle records (§5.1, which
an implementation MAY write) at or below it, an event's ``index`` and a room's
``event_count``, ``latest_index`` and ``delivered_index`` alike; an event of a
timeline is named ``<room>#<index>``, as a value or as an object's key; any
other event id is ``$e1``, ``$e2``... by first appearance; a timestamp is
``@1``, ``@2``... by its order among the run's instants.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_INSTANT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
_ID_KEYS = frozenset({"id", "event_id", "responds_to", "parent_event_id", "target_event_id"})


def is_lifecycle_record(event: dict[str, Any]) -> bool:
    """A record the framework writes of a lifecycle change (§5.1): its source
    is the framework's, its visibility ``internal``, its content a system one."""
    source = event.get("source") or {}
    content = event.get("content") or {}
    return (
        source.get("channel_id") == "system"
        and event.get("visibility") == "internal"
        and content.get("type") == "system"
    )


@dataclass
class Observed:
    """What a run produced, in the RFC's serialized form, before normalization."""

    literal_rooms: set[str]
    created_rooms: list[str] = field(default_factory=list)
    timelines: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    rooms: dict[str, Any] = field(default_factory=dict)
    bindings: dict[str, dict[str, Any]] = field(default_factory=dict)
    framework_events: list[dict[str, Any]] = field(default_factory=list)
    ai_events: list[dict[str, Any]] = field(default_factory=list)
    hooks: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    hook_order: list[str] = field(default_factory=list)
    deliveries: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    contexts: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    streamed: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    thinking_events: dict[str, list[Any]] = field(default_factory=dict)
    tool_events: dict[str, list[Any]] = field(default_factory=dict)
    text_events: dict[str, list[Any]] = field(default_factory=dict)
    room_callbacks: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    turns: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    ingested: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    compacted: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    dispatch_turns: dict[str, list[Any]] = field(default_factory=dict)
    observations: dict[str, list[Any]] = field(default_factory=dict)
    tasks: dict[str, list[Any]] = field(default_factory=dict)
    steps: list[Any] = field(default_factory=list)
    actions: dict[str, list[Any]] = field(default_factory=dict)
    models: dict[str, list[Any]] = field(default_factory=dict)
    acp_wire: dict[str, list[Any]] = field(default_factory=dict)
    executors: dict[str, list[Any]] = field(default_factory=dict)
    participants: dict[str, list[Any]] = field(default_factory=dict)
    resolver: list[dict[str, Any]] = field(default_factory=list)
    router_calls: list[dict[str, Any]] = field(default_factory=list)
    media_resolver: list[dict[str, Any]] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)
    tool_audit: list[Any] = field(default_factory=list)
    session_audit: list[Any] = field(default_factory=list)


class Normalizer:
    """Rewrites the ids, indexes and instants of one run's records."""

    def __init__(self, observed: Observed) -> None:
        self._rooms: dict[str, str] = {room: room for room in observed.literal_rooms}
        for room_id in observed.created_rooms:
            if room_id not in self._rooms:
                self._rooms[room_id] = (
                    f"$r{sum(a.startswith('$r') for a in self._rooms.values()) + 1}"
                )
        self._events: dict[str, str] = {}
        self._lifecycle: dict[str, list[int]] = {}
        for room_id, events in observed.timelines.items():
            self._lifecycle[room_id] = [e["index"] for e in events if is_lifecycle_record(e)]
            for event in events:
                if not is_lifecycle_record(event):
                    index = self.index(room_id, event["index"])
                    self._events[event["id"]] = f"{self.room(room_id)}#{index}"
        self._others: dict[str, str] = {}
        # Human-input requests are $p1, $p2... in the order the run showed them.
        self._requests = {
            pending_id: f"$p{number}" for number, pending_id in enumerate(observed.requests, 1)
        }
        self._instants = sorted(set(_instants(observed)))
        # The ids of no timeline are named in the order the step outcomes,
        # then the framework events, show them first.
        self.value(observed.steps)
        self.value(observed.framework_events)

    def room(self, room_id: str) -> str:
        """The alias of a room."""
        return self._rooms.get(room_id, room_id)

    def index(self, room_id: str, index: int) -> int:
        """*index* without the room's lifecycle records at or below it."""
        return index - bisect.bisect_right(self._lifecycle.get(room_id, []), index)

    def count(self, room_id: str, count: int) -> int:
        """A count of indexes given out, without the lifecycle records'."""
        return count - bisect.bisect_left(self._lifecycle.get(room_id, []), count)

    def value(self, value: Any, key: str | None = None) -> Any:
        """*value* with its ids, indexes and instants normalized."""
        if isinstance(value, dict):
            return self._object(value)
        if isinstance(value, list):
            return [self.value(item, key) for item in value]
        if isinstance(value, str):
            return self._string(value, key)
        return value

    def _object(self, value: dict[str, Any]) -> dict[str, Any]:
        indexes = self._indexes_of(value)
        # A participant's or an identity's id is the scenario's own (§5.5,
        # §5.6), never an event's: kept as written.
        own_id = "identification" in value or "channel_addresses" in value
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in indexes and isinstance(item, int) and not isinstance(item, bool):
                result[self._key(key)] = indexes[key](item)
            elif own_id and key == "id":
                result[key] = item
            else:
                result[self._key(key)] = self.value(item, key)
        return result

    def _key(self, key: str) -> str:
        """An object's key, an event of a timeline's id named as its event."""
        return self._events.get(key, key)

    def _indexes_of(self, value: dict[str, Any]) -> dict[str, Any]:
        """How the members of *value* that hold an index of its room are read."""
        room_id = value.get("room_id")
        if isinstance(room_id, str) and "source" in value and "index" in value:
            return {"index": lambda raw: self.index(room_id, raw)}
        room = value.get("id")
        if isinstance(room, str) and "event_count" in value and "status" in value:

            def latest(raw: int) -> int | None:
                index = self.index(room, raw)
                return index if index >= 0 else None

            return {
                "event_count": lambda raw: self.count(room, raw),
                "latest_index": latest,
                "delivered_index": lambda raw: max(self.index(room, raw), -1),
            }
        if isinstance(room_id, str) and value.get("event") == "delivery_skipped":
            # The indexes a delivery lane skipped (§8.2), read as an event's.
            return {
                "from_index": lambda raw: self.index(room_id, raw),
                "to_index": lambda raw: self.index(room_id, raw),
            }
        return {}

    def _string(self, value: str, key: str | None) -> str:
        if value in self._rooms:
            return self._rooms[value]
        if value in self._events:
            return self._events[value]
        if value in self._others:
            return self._others[value]
        if value in self._requests:
            return self._requests[value]
        if key in _ID_KEYS:
            if value not in self._others:
                self._others[value] = f"$e{len(self._others) + 1}"
            return self._others[value]
        if _INSTANT.match(value):
            return f"@{bisect.bisect_left(self._instants, _parse(value)) + 1}"
        return value


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _instants(observed: Observed) -> list[datetime]:
    found: list[datetime] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and _INSTANT.match(value):
            found.append(_parse(value))

    walk(
        [
            observed.steps,
            observed.actions,
            observed.framework_events,
            observed.hooks,
            observed.turns,
            observed.ingested,
            observed.dispatch_turns,
            observed.deliveries,
            observed.streamed,
            observed.room_callbacks,
            observed.timelines,
            observed.rooms,
            observed.bindings,
            observed.observations,
            observed.tasks,
            observed.participants,
        ]
    )
    return found
