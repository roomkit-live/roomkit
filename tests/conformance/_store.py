"""The store's calls a scenario names, made on the kit's store.

A call ``store.<operation>`` is the operation of the store's own interface
(RFC §14.1, roomkit-specs ``conformance/README.md``, "Calls on the store"),
its arguments by the operation's names: a record is a template completed as
the record requires (README, "Templates"), the rest is passed as it is.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit
from roomkit.models.channel import ChannelBinding
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter
from tests.conformance._calls import Call, read
from tests.conformance._script import SYSTEM_SOURCE, complete_event, complete_participant

if TYPE_CHECKING:
    from tests.conformance._harness import Harness


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _room(template: Mapping[str, Any], spec: Mapping[str, Any]) -> Room:
    stamp = _now()
    return Room.model_validate({"created_at": stamp, "updated_at": stamp, **template})


def _binding(template: Mapping[str, Any], spec: Mapping[str, Any]) -> ChannelBinding:
    return ChannelBinding.model_validate({"attached_at": _now(), **template})


def _participant(template: Mapping[str, Any], spec: Mapping[str, Any]) -> Any:
    return complete_participant(template, template["room_id"])


def _event(template: Mapping[str, Any], spec: Mapping[str, Any]) -> Any:
    return complete_event(template, spec["room_id"], SYSTEM_SOURCE)


def _event_filter(value: Mapping[str, Any], spec: Mapping[str, Any]) -> EventFilter:
    return EventFilter.model_validate(value)


# How an argument is read, by its name; any other is passed as it is.
_RECORDS: dict[str, Callable[[Any, Mapping[str, Any]], Any]] = {
    "room": _room,
    "binding": _binding,
    "participant": _participant,
    "event": _event,
    "target": _event,
    "event_filter": _event_filter,
}

# Each store call's arguments (README, "Calls on the store").
STORE_ARGUMENTS: dict[str, set[str]] = {
    "store.create_room": {"room"},
    "store.delete_room": {"room_id"},
    "store.add_binding": {"binding"},
    "store.update_binding": {"binding"},
    "store.remove_binding": {"room_id", "channel_id"},
    "store.add_participant": {"participant"},
    "store.commit_event": {"room_id", "event", "target", "activity"},
    "store.room_exists": {"room_id"},
    "store.binding_exists": {"room_id", "channel_id"},
    "store.find_latest_room": {"participant_id", "channel_type", "status"},
    "store.find_room_id_by_channel": {"channel_id", "status"},
    "store.get_timeline": {"room_id", "event_filter", "limit", "after_index", "newest_first"},
    "store.get_thread_summaries": {"room_id", "root_event_ids"},
}


def _argument(name: str, spec: Mapping[str, Any]) -> Any:
    if name not in _RECORDS:
        return spec[name]
    return read(lambda: _RECORDS[name](spec[name], spec))


async def _store_call(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    operation = spec["call"].removeprefix("store.")
    arguments = {
        name: _argument(name, spec) for name in STORE_ARGUMENTS[spec["call"]] if name in spec
    }
    return await getattr(kit.store, operation)(**arguments)


STORE_CALLS: dict[str, Call] = dict.fromkeys(STORE_ARGUMENTS, _store_call)
