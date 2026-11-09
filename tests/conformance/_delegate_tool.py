"""The ``delegate_task`` tool kit (README, "Calls"; RFC §23.4).

``delegate_handler`` builds the implementation's DelegateHandler under a name the
later calls use; ``setup_delegation`` serves the tool on an agent, whose model
then calls it; ``delegate.handle`` is the handler's own call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit
from roomkit.tasks import (
    CompletedTaskCache,
    DelegateHandler,
    build_delegate_tool,
    setup_delegation,
)

if TYPE_CHECKING:
    from tests.conformance._calls import Call
    from tests.conformance._harness import Harness


async def _delegate_handler(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    options = {
        name: spec[name]
        for name in (
            "notify",
            "default_share_channels",
            "serialize_per_room",
            "max_conversation",
            "max_result_chars",
        )
        if name in spec
    }
    cache = CompletedTaskCache() if spec.get("cache") else None
    harness.delegate_handlers[spec["handler"]] = DelegateHandler(kit, cache=cache, **options)


async def _setup_delegation(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    targets = spec.get("targets")
    setup_delegation(
        spec["agent_id"],
        harness.delegate_handlers[spec["handler"]],
        tool=None if targets is None else build_delegate_tool([tuple(t) for t in targets]),
    )


async def _handle(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    handler = harness.delegate_handlers[spec["handler"]]
    answer = await handler.handle(spec["room_id"], spec["calling_agent_id"], spec["arguments"])
    harness.handled_tasks.add(answer["task_id"])
    return answer


DELEGATE_TOOL_CALLS: dict[str, Call] = {
    "delegate_handler": _delegate_handler,
    "setup_delegation": _setup_delegation,
    "delegate.handle": _handle,
}

DELEGATE_TOOL_ARGUMENTS: dict[str, set[str]] = {
    "delegate_handler": {
        "handler",
        "notify",
        "default_share_channels",
        "cache",
        "serialize_per_room",
        "max_conversation",
        "max_result_chars",
    },
    "setup_delegation": {"agent_id", "handler", "targets"},
    "delegate.handle": {"handler", "room_id", "calling_agent_id", "arguments"},
}
