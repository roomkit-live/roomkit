"""The members a scenario may hold, so a runner never ignores one it does not know.

roomkit-specs ``conformance/scenario.schema.json`` is the scenario's grammar;
this is its skeleton, checked before a run: a member the runner would not read
(a misspelled expectation, a step of no known kind) fails the scenario rather
than pass unread. Records in the RFC's serialized form (messages, contents,
templates, capabilities) are the implementation's to read, and open here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TOP = {"$schema", "id", "title", "rules", "reentrancy", "notes", "setup", "steps", "expect"}
SETUP = {"kit", "channels", "rooms", "hooks"}
KIT = {"max_chain_depth", "process_timeout_ms"}
CHANNEL = {
    "id",
    "type",
    "category",
    "direction",
    "capabilities",
    "recipient_key",
    "parse",
    "on_event",
    "deliver",
    "streams",
    "deliver_stream",
    "model",
    "tools",
    "tool_policy",
    "max_tool_rounds",
}
MODEL = {"rounds"}
MODEL_ROUND = {"text", "deltas", "calls", "finish_reason", "usage", "fails"}
MODEL_CALL = {"id", "name", "arguments", "cut"}
TOOL = {
    "name",
    "description",
    "parameters",
    "returns",
    "raises",
    "fails",
    "refuses",
    "hangs",
    "declines",
}
RAISES = {"kind", "message"}
POLICY = {"allow", "deny"}
ROOM = {"id", "organization_id", "metadata", "timers", "bindings", "participants", "status"}
BINDING = {"channel_id", "category", "access", "muted", "visibility", "metadata", "participant_id"}
HOOK = {
    "name",
    "trigger",
    "execution",
    "priority",
    "timeout_ms",
    "channel_types",
    "channel_ids",
    "directions",
    "fail_closed",
    "needs_lock",
    "reactions",
}
REACTION = {
    "when",
    "repeat",
    "gate",
    "do",
    "fail",
    "respond",
    "output",
    "result",
    "stop_after",
}
WHEN = {"room_id", "text", "type", "source_channel_id", "chain_depth"}
OUTPUT = {"tasks", "observations", "metadata_updates"}
RESULT = {"action", "event", "reason", "injected_events", "tasks", "observations", "metadata"}
INJECTED = {"event", "target_channel_ids"}
OUTCOME = {"result", "error"}
ERROR = {"name", "reason"}
EXPECT = {
    "timeline",
    "rooms",
    "bindings",
    "framework_events",
    "hooks",
    "hook_order",
    "deliveries",
    "turns",
    "observations",
    "tasks",
    "actions",
    "lifecycle",
    "models",
    "streamed",
}
NEW_ROOM = {"id", "organization_id", "metadata", "timers"}

# Each call's arguments, beside `call` and `organization_id`.
CALLS: dict[str, set[str]] = {
    "process_inbound": {"message", "room_id"},
    "send_event": {
        "room_id",
        "channel_id",
        "content",
        "event_type",
        "participant_id",
        "visibility",
        "response_visibility",
        "addressed_to",
        "parent_event_id",
        "idempotency_key",
        "metadata",
        "chain_depth",
        "standalone",
    },
    "post": {
        "room_id",
        "content",
        "participant_id",
        "addressed_to",
        "metadata",
        "channel_id",
        "idempotency_key",
        "chain_depth",
    },
    "commit_event": {"room_id", "event"},
    "update_event": {"room_id", "event_id", "content", "source", "metadata"},
    "delete_event": {"room_id", "event_id", "cascade_replies"},
    "create_room": {"room"},
    "attach_channel": {
        "room_id",
        "channel_id",
        "category",
        "access",
        "visibility",
        "muted",
        "metadata",
        "participant_id",
    },
    "detach_channel": {"room_id", "channel_id"},
    "mute": {"room_id", "channel_id"},
    "unmute": {"room_id", "channel_id"},
    "set_access": {"room_id", "channel_id", "access"},
    "set_visibility": {"room_id", "channel_id", "visibility"},
    "close_room": {"room_id"},
    "archive_room": {"room_id"},
    "check_timers": {"room_id"},
    "register_channel": {"channel"},
    "unregister_channel": {"channel_id"},
    "register_hook": {"hook"},
}
STEPS = {"await": {"await", "expect"}, "concurrent": {"concurrent"}, "wait_for": {"wait_for"}}
STEPS |= {"open": {"open"}, "idle": {"idle"}}


def unknown_members(scenario: Mapping[str, Any]) -> list[str]:
    """Every member of *scenario* the runner does not know, as ``path: member``."""
    found: list[str] = []
    _members(scenario, TOP, "$", found)
    setup = scenario.get("setup", {})
    _members(setup, SETUP, "setup", found)
    _members(setup.get("kit", {}), KIT, "setup.kit", found)
    for position, channel in enumerate(setup.get("channels", [])):
        _channel(channel, f"setup.channels[{position}]", found)
    for position, room in enumerate(setup.get("rooms", [])):
        path = f"setup.rooms[{position}]"
        _members(room, ROOM, path, found)
        for index, binding in enumerate(room.get("bindings", [])):
            _members(binding, BINDING, f"{path}.bindings[{index}]", found)
    for position, hook in enumerate(setup.get("hooks", [])):
        _hook(hook, f"setup.hooks[{position}]", found)
    for position, step in enumerate(scenario.get("steps", [])):
        _step(step, f"steps[{position}]", found)
    expect = scenario.get("expect", {})
    _members(expect, EXPECT, "expect", found)
    for label, outcomes in expect.get("actions", {}).items():
        listed = outcomes.get("$unordered", []) if isinstance(outcomes, dict) else outcomes
        for index, outcome in enumerate(listed):
            _outcome(outcome, f"expect.actions.{label}[{index}]", found)
    return found


def _members(value: Any, known: set[str], path: str, found: list[str]) -> None:
    if not isinstance(value, Mapping):
        found.append(f"{path}: not an object")
        return
    found.extend(f"{path}: {member}" for member in value if member not in known)


def _channel(channel: Mapping[str, Any], path: str, found: list[str]) -> None:
    _members(channel, CHANNEL, path, found)
    for kind in ("parse", "on_event", "deliver", "deliver_stream"):
        for index, reaction in enumerate(channel.get(kind, [])):
            _reaction(reaction, f"{path}.{kind}[{index}]", found)
    if "model" in channel:
        _model(channel["model"], f"{path}.model", found)
    for index, tool in enumerate(channel.get("tools", [])):
        _members(tool, TOOL, f"{path}.tools[{index}]", found)
        if "raises" in tool:
            _members(tool["raises"], RAISES, f"{path}.tools[{index}].raises", found)
    if "tool_policy" in channel:
        _members(channel["tool_policy"], POLICY, f"{path}.tool_policy", found)


def _model(model: Mapping[str, Any], path: str, found: list[str]) -> None:
    _members(model, MODEL, path, found)
    for index, round_ in enumerate(model.get("rounds", [])):
        _members(round_, MODEL_ROUND, f"{path}.rounds[{index}]", found)
        for position, call in enumerate(round_.get("calls", [])):
            _members(call, MODEL_CALL, f"{path}.rounds[{index}].calls[{position}]", found)


def _hook(hook: Mapping[str, Any], path: str, found: list[str]) -> None:
    _members(hook, HOOK, path, found)
    for index, reaction in enumerate(hook.get("reactions", [])):
        _reaction(reaction, f"{path}.reactions[{index}]", found)


def _reaction(reaction: Mapping[str, Any], path: str, found: list[str]) -> None:
    _members(reaction, REACTION, path, found)
    _members(reaction.get("when", {}), WHEN, f"{path}.when", found)
    _members(reaction.get("output", {}), OUTPUT, f"{path}.output", found)
    if "result" in reaction:
        _members(reaction["result"], RESULT, f"{path}.result", found)
        for index, item in enumerate(reaction["result"].get("injected_events", [])):
            _members(item, INJECTED, f"{path}.result.injected_events[{index}]", found)
    for index, action in enumerate(reaction.get("do", [])):
        _call(action, {"as", "expect"}, f"{path}.do[{index}]", found)


def _step(step: Mapping[str, Any], path: str, found: list[str]) -> None:
    if "call" in step:
        _call(step, {"as", "background", "expect"}, path, found)
        return
    kinds = [kind for kind in STEPS if kind in step]
    if len(kinds) != 1:
        found.append(f"{path}: a step of no known kind: {sorted(step)}")
        return
    _members(step, STEPS[kinds[0]], path, found)
    if kinds[0] == "await" and "expect" in step:
        _outcome(step["expect"], f"{path}.expect", found)
    for index, call in enumerate(step.get("concurrent", [])):
        _call(call, {"as", "expect"}, f"{path}.concurrent[{index}]", found)


def _call(call: Mapping[str, Any], extra: set[str], path: str, found: list[str]) -> None:
    name = call.get("call")
    if name not in CALLS:
        found.append(f"{path}: no call {name!r}")
        return
    _members(call, CALLS[name] | extra | {"call", "organization_id"}, path, found)
    if "expect" in call:
        _outcome(call["expect"], f"{path}.expect", found)
    if name == "create_room":
        _members(call.get("room", {}), NEW_ROOM, f"{path}.room", found)
    if name == "register_channel":
        _channel(call.get("channel", {}), f"{path}.channel", found)
    if name == "register_hook":
        _hook(call.get("hook", {}), f"{path}.hook", found)


def _outcome(outcome: Any, path: str, found: list[str]) -> None:
    _members(outcome, OUTCOME, path, found)
    if isinstance(outcome, Mapping) and isinstance(outcome.get("error"), Mapping):
        _members(outcome["error"], ERROR, f"{path}.error", found)
