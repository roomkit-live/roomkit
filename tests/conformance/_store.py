"""The store's calls a scenario names, made on the kit's store, and the kit's
calls on a room's read markers.

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
from roomkit.models.enums import RoomStatus
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter
from roomkit.models.verification import Verification, VerificationRequest
from tests.conformance._calls import Call, read
from tests.conformance._script import (
    SYSTEM_SOURCE,
    complete_event,
    complete_participant,
    complete_task,
)

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
    room_id = spec.get("room_id", template.get("room_id", ""))
    return complete_event(template, room_id, SYSTEM_SOURCE)


def _event_filter(value: Mapping[str, Any], spec: Mapping[str, Any]) -> EventFilter:
    return EventFilter.model_validate(value)


def _task(template: Mapping[str, Any], spec: Mapping[str, Any]) -> Any:
    return complete_task(template, template["room_id"])


def _verification(record: Mapping[str, Any], spec: Mapping[str, Any]) -> Verification:
    return Verification.model_validate(record)


def _verification_request(
    record: Mapping[str, Any], spec: Mapping[str, Any]
) -> VerificationRequest:
    return VerificationRequest.model_validate(record)


def _instant(value: str | None, spec: Mapping[str, Any]) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _status(value: str, spec: Mapping[str, Any]) -> RoomStatus:
    return RoomStatus(value)


# How an argument is read, by its name; any other is passed as it is.
_RECORDS: dict[str, Callable[[Any, Mapping[str, Any]], Any]] = {
    "room": _room,
    "binding": _binding,
    "participant": _participant,
    "event": _event,
    "target": _event,
    "event_filter": _event_filter,
    "task": _task,
    "verification": _verification,
    "request": _verification_request,
    "paused_at": _instant,
    "last_activity_at": _instant,
}

# The arguments of update_room_if, the status read as a room's.
_CONDITIONS: dict[str, Callable[[Any, Mapping[str, Any]], Any]] = {"status": _status}

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
    "store.add_event": {"event"},
    "store.add_event_auto_index": {"room_id", "event"},
    "store.delete_event": {"room_id", "event_id", "cascade_replies"},
    "store.mark_read": {"room_id", "channel_id", "event_id"},
    "store.mark_all_read": {"room_id", "channel_id"},
    "store.get_unread_count": {"room_id", "channel_id"},
    "store.list_read_markers": {"room_id"},
    "store.get_event": {"event_id"},
    "store.check_idempotency": {"room_id", "key", "channel_id"},
    "store.get_event_by_idempotency_key": {"room_id", "key", "channel_id"},
    "store.update_event": {"event"},
    "store.record_delivery_results": {"room_id", "event_id", "results"},
    "store.write_change_target": {"room_id", "target"},
    "store.update_room_if": {"room", "status", "paused_at", "last_activity_at"},
    "store.get_room": {"room_id"},
    "store.patch_room_metadata": {"room_id", "patch", "unset"},
    "store.add_task": {"task"},
    "store.get_task": {"task_id"},
    "store.update_task": {"task"},
    "store.list_tasks": {"room_id", "status"},
    "store.put_verification": {"verification"},
    "store.get_verification": {"identity_id", "organization_id"},
    "store.list_verifications_in_force": set(),
    "store.create_verification_request": {"request"},
    "store.get_verification_request": {"request_id"},
    "store.update_verification_request": {"request"},
}

# The kit's calls on a room's read markers (README, "Calls"), beside
# `organization_id`.
READ_MARKER_ARGUMENTS: dict[str, set[str]] = {
    "mark_read": {"room_id", "channel_id", "event_id"},
    "mark_all_read": {"room_id", "channel_id"},
    "list_read_markers": {"room_id"},
}


def _argument(name: str, spec: Mapping[str, Any]) -> Any:
    readers = _RECORDS | (_CONDITIONS if spec["call"] == "store.update_room_if" else {})
    if name not in readers:
        return spec[name]
    return read(lambda: readers[name](spec[name], spec))


async def _store_call(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    operation = spec["call"].removeprefix("store.")
    arguments = {
        name: _argument(name, spec) for name in STORE_ARGUMENTS[spec["call"]] if name in spec
    }
    return await getattr(kit.store, operation)(**arguments)


async def _read_marker_call(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    names = READ_MARKER_ARGUMENTS[spec["call"]] | {"organization_id"}
    arguments = {name: spec[name] for name in names if name in spec}
    return await getattr(kit, spec["call"])(**arguments)


STORE_CALLS: dict[str, Call] = {
    **dict.fromkeys(STORE_ARGUMENTS, _store_call),
    **dict.fromkeys(READ_MARKER_ARGUMENTS, _read_marker_call),
}
