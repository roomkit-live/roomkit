"""Declared owner callbacks and calls through the public turn host."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from roomkit import current_turn_host
from roomkit.tools import (
    current_response_metadata,
    current_tool_actor_id,
    current_tool_allowed_names,
    current_tool_call,
    current_tool_chain_depth,
    current_tool_requester,
    current_tool_room,
    current_tool_room_id,
)
from roomkit.turns._error_codec import error_record
from tests.conformance._turn_errors import inspect_failure

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness
    from tests.conformance._script import Gates


HOST_ARGUMENTS = {
    "declare_host_verb": {"name", "handler", "between_turns"},
    "host.call": {"verb", "arguments", "record_keys", "inspect_error"},
}
HOST_HANDLER = {
    "returns",
    "returns_context",
    "raises_captured",
    "returns_arguments",
    "raises",
    "as",
    "gate",
    "writes_record",
    "removes_record",
    "updates_record",
    "removes_entries",
    "record_keys",
    "cancelled",
}
HOST_CALLS_SCRIPT = {"calls", "concurrent", "record_keys"}
HOST_CALL = {"verb", "arguments", "returned", "cancel_after"}
_ERRORS: dict[str, type[Exception]] = {
    "ValueError": ValueError,
    "RuntimeError": RuntimeError,
    "PermissionError": PermissionError,
}


def _record(keys: list[str] | None = None) -> dict[str, Any] | None:
    record = current_response_metadata()
    if record is None:
        return None
    return deepcopy({key: record[key] for key in keys or () if key in record})


def _context(arguments: dict[str, Any]) -> dict[str, Any]:
    call = current_tool_call()
    room = current_tool_room()
    names = current_tool_allowed_names()
    return {
        "arguments": deepcopy(arguments),
        "room_id": current_tool_room_id(),
        "organization_id": None if room is None else room.organization_id,
        "room_metadata": None if room is None else deepcopy(room.metadata),
        "channel_id": None if call is None else call.channel_id,
        "tool_call_id": None if call is None else call.tool_call_id,
        "actor_id": current_tool_actor_id(),
        "requester": current_tool_requester(),
        "chain_depth": current_tool_chain_depth(),
        "allowed_names": None if names is None else sorted(names),
        "has_record": current_response_metadata() is not None,
        "has_turn_host": current_turn_host() is not None,
    }


def _change_record(spec: Mapping[str, Any]) -> None:
    record = current_response_metadata()
    changes = {"writes_record", "removes_record", "updates_record", "removes_entries"}
    if not changes.intersection(spec):
        return
    assert record is not None, "a scripted record change needs a running turn"
    record.update(deepcopy(spec.get("writes_record", {})))
    for key in spec.get("removes_record", []):
        record.pop(key, None)
    for key, entries in spec.get("updates_record", {}).items():
        record.setdefault(key, {}).update(deepcopy(entries))
    for key, entries in spec.get("removes_entries", {}).items():
        for entry in entries:
            record[key].pop(entry, None)


async def _declare(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    script = spec["handler"]

    async def handler(arguments: dict[str, Any]) -> Any:
        context = _context(arguments)
        if "record_keys" in script:
            context["record"] = _record(script["record_keys"])
        if "as" in script:
            harness.observed.actions.setdefault(script["as"], []).append(
                {"result": context, "error": None}
            )
        if "gate" in script:
            try:
                await harness.gates.hold(script["gate"])
            except asyncio.CancelledError:
                if "cancelled" in script:
                    harness.gates.reach(script["cancelled"])
                raise
        if "record_keys" in script:
            context["record_after_gate"] = _record(script["record_keys"])
        _change_record(script)
        if "raises_captured" in script:
            raise harness.executors.captured_errors.pop(script["raises_captured"])
        if "raises" in script:
            failure = script["raises"]
            raise _ERRORS[failure["kind"]](failure["message"])
        if script.get("returns_context"):
            return context
        if script.get("returns_arguments"):
            return arguments
        return deepcopy(script.get("returns"))

    options = {"between_turns": spec["between_turns"]} if "between_turns" in spec else {}
    kit.declare_host_verb(spec["name"], handler, **options)


async def _call(spec: Mapping[str, Any]) -> dict[str, Any]:
    host = current_turn_host()
    assert host is not None, "a scripted host call needs a turn's host"
    try:
        options = {"arguments": deepcopy(spec["arguments"])} if "arguments" in spec else {}
        result = await host.call(spec["verb"], **options)
    except Exception as error:
        if spec.get("inspect_error"):
            return {"result": None, "error": inspect_failure(error)}
        recorded = error_record(error)
        return {"result": None, "error": {"kind": recorded.kind, "message": recorded.message}}
    return {"result": result, "error": None}


async def _host_call(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    outcome = await _call(spec)
    if "record_keys" in spec:
        outcome["record"] = _record(spec["record_keys"])
    return outcome


async def _tool_call(spec: Mapping[str, Any], gates: Gates) -> dict[str, Any]:
    if "cancel_after" in spec:
        task = asyncio.create_task(_call(spec))
        try:
            await gates.wait_for(spec["cancel_after"])
            task.cancel()
            result = await task
        except asyncio.CancelledError:
            if (current := asyncio.current_task()) is not None and current.cancelling():
                raise
            result = {"result": None, "error": {"kind": "cancelled", "message": ""}}
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    else:
        result = await _call(spec)
    if "returned" in spec:
        gates.reach(spec["returned"])
    return result


async def call_from_tool(spec: Mapping[str, Any], gates: Gates) -> str:
    """Call the public host, then read the caller's record after every reply.

    Canonical JSON text is a fixture representation; it lets an exact model
    result assert nested values without depending on map iteration order.
    """
    calls = spec["calls"]
    if spec.get("concurrent"):
        results = await asyncio.gather(*(_tool_call(call, gates) for call in calls))
    else:
        results = [await _tool_call(call, gates) for call in calls]
    answer = {"calls": results, "record": _record(spec.get("record_keys"))}
    return json.dumps(answer, ensure_ascii=False, sort_keys=True)


HOST_CALLS = {"declare_host_verb": _declare, "host.call": _host_call}
