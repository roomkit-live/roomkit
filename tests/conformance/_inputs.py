"""The kit's calls of human input, made on the Python core (roomkit-specs
``conformance/README.md``, "Calls").

``answer_input`` and ``reject_input`` are the kit's doors to a turn's
human-input request (RFC §10.6.5): they answer whether the waiting tool reads
the outcome. A request is named by ``$request`` in an action of a hook on
``on_user_input_required``, and by ``$p1``, ``$p2``... anywhere, in the order
the run first shows them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit
from roomkit.models.pending_input import PendingInputEvent
from tests.conformance._calls import Call

if TYPE_CHECKING:
    from tests.conformance._harness import Harness

# Each call's arguments, beside `call` and `organization_id`.
INPUT_ARGUMENTS: dict[str, set[str]] = {
    "answer_input": {"request_id", "value"},
    "reject_input": {"request_id", "reason"},
}


async def _answer_input(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.answer_input(spec["request_id"], spec["value"])


async def _reject_input(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await kit.reject_input(spec["request_id"], spec.get("reason", ""))


INPUT_CALLS: dict[str, Call] = {
    "answer_input": _answer_input,
    "reject_input": _reject_input,
}


def request_record(payload: PendingInputEvent) -> dict[str, Any]:
    """An ``on_user_input_required`` hook's payload as a scenario reads it
    (README, "Expectations")."""
    return {
        "pending_id": payload.pending_id,
        "tool_name": payload.tool_name,
        "arguments": payload.arguments,
        "room_id": payload.room_id,
        "tool_call_id": payload.tool_call_id,
        "channel_id": payload.channel_id,
        "channel_type": str(payload.channel_type),
        "actor_id": payload.actor_id,
    }
