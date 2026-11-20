"""A room's ephemerals as a subscriber reads them (RFC §8.4, §17.2)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit.realtime.base import EphemeralEvent

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._calls import Call
    from tests.conformance._harness import Harness


async def _subscribe_room(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    """The kit's ``subscribe_room``, its callback recording each event it
    reads under the subscriber's name."""
    read = harness.observed.subscribers.setdefault(spec["subscriber"], [])

    async def record(event: EphemeralEvent) -> None:
        if "types" in spec and event.type.value not in spec["types"]:
            return
        if "gate" in spec:
            harness.gates.reach(spec["gate"])
        read.append({"type": event.type.value, "room_id": event.room_id, "user_id": event.user_id})

    return await kit.subscribe_room(
        spec["room_id"], record, organization_id=spec.get("organization_id")
    )


async def _publish_typing(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    await kit.publish_typing(
        spec["room_id"],
        spec["user_id"],
        spec.get("is_typing", True),
        organization_id=spec.get("organization_id"),
    )


SUBSCRIPTION_CALLS: dict[str, Call] = {
    "subscribe_room": _subscribe_room,
    "publish_typing": _publish_typing,
}
