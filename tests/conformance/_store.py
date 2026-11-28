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
from roomkit.models.identity import Identity
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter, RoomFilter
from roomkit.models.verification import Verification, VerificationRequest
from tests.conformance._calls import Call, read
from tests.conformance._script import (
    SYSTEM_SOURCE,
    complete_event,
    complete_observation,
    complete_participant,
    complete_task,
)

if TYPE_CHECKING:
    from roomkit.store.base import ConversationStore
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


def _event_filter(value: Mapping[str, Any] | None, spec: Mapping[str, Any]) -> EventFilter | None:
    return None if value is None else EventFilter.model_validate(value)


def _room_filter(value: Mapping[str, Any] | None, spec: Mapping[str, Any]) -> RoomFilter | None:
    return None if value is None else RoomFilter.model_validate(value)


def _observation(template: Mapping[str, Any], spec: Mapping[str, Any]) -> Any:
    return complete_observation(template, template["room_id"], template["channel_id"])


def _task(template: Mapping[str, Any], spec: Mapping[str, Any]) -> Any:
    return complete_task(template, template["room_id"])


def _verification(record: Mapping[str, Any], spec: Mapping[str, Any]) -> Verification:
    return Verification.model_validate(record)


def _verification_request(
    record: Mapping[str, Any], spec: Mapping[str, Any]
) -> VerificationRequest:
    return VerificationRequest.model_validate(record)


def _identity(record: Mapping[str, Any], spec: Mapping[str, Any]) -> Identity:
    return Identity.model_validate(record)


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
    "room_filter": _room_filter,
    "task": _task,
    "observation": _observation,
    "verification": _verification,
    "request": _verification_request,
    "identity": _identity,
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
    "store.update_participant": {"participant"},
    "store.create_identity": {"identity"},
    "store.get_identity": {"identity_id"},
    "store.resolve_identity": {"channel_type", "address", "organization_id"},
    "store.link_address": {"identity_id", "channel_type", "address", "organization_id"},
    "store.unlink_address": {"identity_id", "channel_type", "address", "organization_id"},
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
    "store.update_room": {"room"},
    "store.list_rooms": {"offset", "limit"},
    "store.find_rooms": {"organization_id", "status", "metadata_filter", "limit", "offset"},
    "store.load_room_context": {"room_id"},
    "store.get_conversation": {"room_id", "limit", "after_index"},
    "store.get_delivered_index": {"room_id"},
    "store.advance_delivered_index": {"room_id", "index", "force"},
    "store.get_private_state": {"room_id", "key"},
    "store.set_private_state": {"room_id", "key", "value"},
    "store.find_room_id_by_binding": {
        "channel_id",
        "participant_id",
        "identity_id",
        "room_filter",
        "recipient_key",
    },
    "store.find_room_id_by_participant": {
        "channel_id",
        "participant_id",
        "identity_id",
        "room_filter",
    },
    "store.get_event_count": {"room_id", "event_filter"},
    "store.get_binding": {"room_id", "channel_id"},
    "store.list_bindings": {"room_id"},
    "store.delete_participant": {"room_id", "participant_id"},
    "store.list_participants": {"room_id"},
    "store.add_observation": {"observation"},
    "store.list_observations": {"room_id"},
}

# The store calls whose room may leave out its ``created_at``, which the
# stored room's then gives (README, "Calls on the store").
_WHOLE_ROOM_WRITES = {"store.update_room", "store.update_room_if"}

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


async def _kept_created_at(store: ConversationStore, spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """*spec* with its room's template given the stored room's ``created_at``
    when it names none: a write of a partial room updates that room, it
    creates none, and a room is its id and its ``created_at`` (RFC §13.5)."""
    template = spec.get("room")
    if not isinstance(template, Mapping) or "created_at" in template:
        return spec
    room_id = template.get("id")
    stored = await store.get_room(room_id) if isinstance(room_id, str) else None
    if stored is None:
        return spec
    return {**spec, "room": {**template, "created_at": stored.created_at}}


async def _store_call(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    operation = spec["call"].removeprefix("store.")
    store = harness.store_of(kit)
    if spec["call"] in _WHOLE_ROOM_WRITES:
        spec = await _kept_created_at(store, spec)
    arguments = {
        name: _argument(name, spec) for name in STORE_ARGUMENTS[spec["call"]] if name in spec
    }
    result = await getattr(store, operation)(**arguments)
    # load_room_context's (room, bindings, participants), read as a list.
    return list(result) if isinstance(result, tuple) else result


async def _read_marker_call(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    names = READ_MARKER_ARGUMENTS[spec["call"]] | {"organization_id"}
    arguments = {name: spec[name] for name in names if name in spec}
    return await getattr(kit, spec["call"])(**arguments)


STORE_CALLS: dict[str, Call] = {
    **dict.fromkeys(STORE_ARGUMENTS, _store_call),
    **dict.fromkeys(READ_MARKER_ARGUMENTS, _read_marker_call),
}
