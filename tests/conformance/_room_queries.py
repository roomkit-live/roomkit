"""Public room reads and the timer sweep (RFC §5.1), without store shortcuts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness


async def call(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    """Forward only arguments the public API takes; omitted ones keep its defaults."""
    name = spec["call"]
    names = {
        "get_room": (),
        "get_binding": ("channel_id",),
        "list_bindings": (),
        "list_members": ("include_left",),
        "get_timeline": (
            "offset",
            "limit",
            "visibility_filter",
            "after_index",
            "before_index",
            "newest_first",
            "include_blocked",
        ),
        "check_all_timers": (),
    }[name]
    options = {key: spec[key] for key in (*names, "organization_id") if key in spec}
    method = getattr(kit, name)
    if name == "check_all_timers":
        return await method(**options)
    return await method(spec["room_id"], **options)
