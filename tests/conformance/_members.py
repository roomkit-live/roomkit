"""The kit's calls on a room's members (RFC §5.5), made on the Python core
(roomkit-specs ``conformance/README.md``, "Calls"), and the identities a
scenario's setup stores for them to read."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit
from roomkit.models.enums import ParticipantRole, ParticipantStatus
from roomkit.models.identity import Identity
from tests.conformance._calls import Call, read

if TYPE_CHECKING:
    from tests.conformance._harness import Harness

# Each call's arguments, beside `call` and `organization_id`.
MEMBER_ARGUMENTS: dict[str, set[str]] = {
    "ensure_participant": {"room_id", "channel_id", "participant_id", "display_name"},
    "resolve_participant": {"room_id", "participant_id", "identity_id", "resolved_by"},
    "add_member": {
        "room_id",
        "channel_id",
        "participant_id",
        "identity_id",
        "display_name",
        "role",
    },
    "remove_member": {"room_id", "participant_id", "status"},
    "rename_member": {"room_id", "participant_id", "display_name"},
    "is_member": {"room_id", "identity_id"},
}


def _org(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {"organization_id": spec["organization_id"]} if "organization_id" in spec else {}


async def store_identities(kit: RoomKit, identities: list[Mapping[str, Any]]) -> None:
    """Stores each of a setup's ``identities`` in the kit's store."""
    for spec in identities:
        await kit.store.create_identity(_identity(spec))


def _identity(spec: Mapping[str, Any]) -> Identity:
    return read(lambda: Identity.model_validate(spec))


async def _ensure_participant(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.ensure_participant(
        spec["room_id"],
        spec["channel_id"],
        spec["participant_id"],
        spec.get("display_name"),
        **_org(spec),
    )


async def _resolve_participant(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    resolved_by = {"resolved_by": spec["resolved_by"]} if "resolved_by" in spec else {}
    return await kit.resolve_participant(
        spec["room_id"], spec["participant_id"], spec["identity_id"], **resolved_by, **_org(spec)
    )


async def _add_member(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    options: dict[str, Any] = {k: spec[k] for k in ("identity_id", "display_name") if k in spec}
    if "role" in spec:
        options["role"] = read(lambda: ParticipantRole(spec["role"]))
    return await kit.add_member(
        spec["room_id"], spec["channel_id"], spec["participant_id"], **options, **_org(spec)
    )


async def _remove_member(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    options: dict[str, Any] = _org(spec)
    if "status" in spec:
        options["status"] = read(lambda: ParticipantStatus(spec["status"]))
    return await kit.remove_member(spec["room_id"], spec["participant_id"], **options)


async def _rename_member(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.rename_member(
        spec["room_id"], spec["participant_id"], spec["display_name"], **_org(spec)
    )


async def _is_member(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.is_member(spec["room_id"], spec["identity_id"], **_org(spec))


MEMBER_CALLS: dict[str, Call] = {
    "ensure_participant": _ensure_participant,
    "resolve_participant": _resolve_participant,
    "add_member": _add_member,
    "remove_member": _remove_member,
    "rename_member": _rename_member,
    "is_member": _is_member,
}
