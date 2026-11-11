"""Public hooks on the kit's own executor (RFC §10.6.6)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit
from roomkit.models.enums import ChannelType, EventDirection, HookExecution, HookTrigger
from tests.conformance._script import hook_handler

if TYPE_CHECKING:
    from tests.conformance._calls import Call
    from tests.conformance._harness import Harness


async def _hook(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    hook = spec["hook"]
    options: dict[str, Any] = {
        "execution": HookExecution(hook.get("execution", "sync")),
        "priority": hook.get("priority", 0),
        "timeout": hook.get("timeout_ms", 30_000) / 1000,
        "fail_closed": hook.get("fail_closed", False),
    }
    if spec["call"] == "kit_executor.room_hook":
        register = kit.turn_executor.room_hook(
            hook["name"], HookTrigger(hook["trigger"]), **options
        )
    else:
        for field, kind in (("channel_types", ChannelType), ("directions", EventDirection)):
            if field in hook:
                options[field] = {kind(value) for value in hook[field]}
        if "channel_ids" in hook:
            options["channel_ids"] = set(hook["channel_ids"])
        register = kit.turn_executor.hook(
            HookTrigger(hook["trigger"]), name=hook["name"], **options
        )
    register(hook_handler(hook, harness))


async def _remove(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> bool:
    return kit.turn_executor.remove_room_hook(spec["name"])


async def _disable(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> bool:
    return await kit.disable_room_hook(
        spec["room_id"], spec["name"], organization_id=spec.get("organization_id")
    )


KIT_EXECUTOR_CALLS: dict[str, Call] = {
    "kit_executor.hook": _hook,
    "kit_executor.room_hook": _hook,
    "kit_executor.remove_room_hook": _remove,
    "disable_room_hook": _disable,
}
KIT_EXECUTOR_ARGUMENTS = {
    "kit_executor.hook": {"hook"},
    "kit_executor.room_hook": {"hook"},
    "kit_executor.remove_room_hook": {"name"},
    "disable_room_hook": {"room_id", "name"},
}
