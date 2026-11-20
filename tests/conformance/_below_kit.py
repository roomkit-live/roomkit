"""Room hooks registered below the kit, which reads no room (RFC §9.1, §17.2)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import HookRegistration, RoomKit
from roomkit.models.enums import HookExecution, HookTrigger
from tests.conformance._calls import Call, read
from tests.conformance._script import hook_handler

if TYPE_CHECKING:
    from tests.conformance._harness import Harness


def _registration(harness: Harness, hook: Mapping[str, Any]) -> HookRegistration:
    """A room hook as ``add_room_hook`` takes it: no filter, the default timeout."""
    return HookRegistration(
        trigger=read(lambda: HookTrigger(hook["trigger"])),
        execution=read(lambda: HookExecution(hook.get("execution", "sync"))),
        fn=hook_handler(hook, harness),
        priority=hook.get("priority", 0),
        name=hook.get("name", ""),
        fail_closed=hook.get("fail_closed", False),
        needs_lock=hook.get("needs_lock", True),
    )


async def _on_engine(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    kit.hook_engine.add_room_hook(spec["room_id"], _registration(harness, spec["hook"]))


async def _given_to_executor(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    kit.turn_executor.hooks.add_owner_room_hook(
        spec["room_id"], _registration(harness, spec["hook"])
    )


BELOW_KIT_CALLS: dict[str, Call] = {
    "hook_engine.add_room_hook": _on_engine,
    "kit_executor.add_owner_room_hook": _given_to_executor,
}
BELOW_KIT_ARGUMENTS = {name: {"room_id", "hook"} for name in BELOW_KIT_CALLS}
