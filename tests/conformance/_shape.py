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

from tests.conformance._delegate_tool import DELEGATE_TOOL_ARGUMENTS
from tests.conformance._discussion import DISCUSSION_ARGUMENTS, DISCUSSION_MEMBERS
from tests.conformance._handoff import HANDOFF_ARGUMENTS
from tests.conformance._inputs import INPUT_ARGUMENTS
from tests.conformance._media import ATTACHED_MEDIA, MEDIA_RESOLVER
from tests.conformance._members import MEMBER_ARGUMENTS
from tests.conformance._routing import (
    CONDITIONS,
    ROUTER,
    ROUTING_ARGUMENTS,
    ROUTING_RULE,
    TRANSITION,
)
from tests.conformance._store import READ_MARKER_ARGUMENTS, STORE_ARGUMENTS
from tests.conformance._strategies import (
    LOOP,
    PIPELINE,
    STRATEGY,
    STRATEGY_ARGUMENTS,
    STRATEGY_STEP,
    STRATEGY_UNINSTALL,
    SUPERVISOR,
    SWARM,
)
from tests.conformance._tasks import TASK_ARGUMENTS
from tests.conformance._verification import VERIFICATION_ARGUMENTS

TOP = {"$schema", "id", "title", "rules", "reentrancy", "notes", "setup", "steps", "expect"}
SETUP = {
    "kit",
    "channels",
    "rooms",
    "hooks",
    "identity_hooks",
    "identities",
    "strategies",
    "session_auditor",
}
KIT = {
    "max_chain_depth",
    "process_timeout_ms",
    "router",
    "identity_resolver",
    "identity_channel_types",
    "identity_timeout_ms",
    "delivery_backend",
    "verification_policy",
    "media_resolver",
    "attached_media",
    "orchestration",
    "delivery_gap_timeout_ms",
    "lock_manager",
}
LOCK_MANAGER = {"release_error"}
DELIVERY_BACKEND = {
    "retry_delay_ms",
    "max_retry_delay_ms",
    "max_queue_size",
    "max_dead_letter_size",
}
RESOLVER = {"answers"}
RESOLVER_ANSWER = {"when", "repeat", "gate", "fail", "result"}
RESOLVER_WHEN = {"sender_id", "channel_id", "room_id", "text"}
IDENTITY_HOOK = {"name", "trigger", "channel_types", "channel_ids", "directions", "reactions"}
IDENTITY_REACTION = {"when", "repeat", "gate", "fail", "result"}
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
    "reads_usage",
    "deliver_stream",
    "on_room_attached",
    "on_room_detached",
    "model",
    "tools",
    "tool_policy",
    "max_tool_rounds",
    "max_empty_retries",
    "max_context_events",
    "tool_timeout_ms",
    "tool_timeouts_ms",
    "turn_budget_tokens",
    "reasoning_effort",
    "response_schema",
    "turn_budget_usd",
    "system_prompt",
    "temperature",
    "max_tokens",
    "config_provider",
    "skills",
    "skills_in_prompt",
    "thinking_budget",
    "enable_thinking",
    "thinking_coalesce_ms",
    "thinking_coalesce_chars",
    "publish_thinking_text",
    "text_coalesce_ms",
    "text_coalesce_chars",
    "publish_text_deltas",
    "retry_policy",
    "fallback_model",
    "max_author_images",
    "describe_empty_event",
    "serialize_room_turns",
    "steer_room_turns",
    "human_input",
    "kit_tools",
    "audit",
    "tool_search",
    "tool_search_pinned",
    "tool_search_threshold",
    "tool_search_threshold_pct",
    "tool_search_miss_hint",
    "tool_search_threshold_tokens",
    "addresses",
    "streams_in",
    "recent_events_window",
    "sender_is_participant",
    "memory",
    "agent",
}
ADDRESSES = {"normalizes", "conversation_key", "recipient_at"}
AGENT = {
    "name",
    "role",
    "description",
    "purpose",
    "scope",
    "language",
    "identity_in_prompt",
    "greeting",
    "auto_greet",
}
MODEL = {
    "rounds",
    "vision",
    "pricing",
    "context_window",
    "name",
    "provider",
    "chat_dialect",
    "response_schema",
    "response_schema_with_tools",
}
MODEL_ROUND = {
    "text",
    "deltas",
    "thinking",
    "calls",
    "finish_reason",
    "usage",
    "fails",
    "done",
    "gate",
    "refuses",
    "failure_after_calls",
    "failure_after_done",
    "stream",
    "whole_response",
}
REFUSES = {"status", "message", "context_overflow", "provider", "retryable"}
RETRY_POLICY = {"max_retries", "base_delay_ms", "max_delay_ms", "exponential_base"}
MODEL_CALL = {"id", "name", "arguments", "cut", "metadata"}
TOOL = {
    "name",
    "description",
    "parameters",
    "tags",
    "returns",
    "raises",
    "fails",
    "refuses",
    "hangs",
    "declines",
    "gate",
    "declared",
    "returns_json",
    "writes_record",
    "returns_parts",
}
RAISES = {"kind", "message"}
HUMAN_INPUT = {"tool_names", "timeout_ms", "tools"}
TURN_CONFIG = {
    "system_prompt",
    "thinking_budget",
    "enable_thinking",
    "tools",
    "temperature",
    "max_tokens",
    "reasoning_effort",
    "response_schema",
    "turn_budget_tokens",
    "turn_budget_usd",
    "tool_handler",
    "skills",
}
TURN_TOOL = {"name", "description", "parameters"}
SKILL = {"name", "description", "instructions", "references"}
HUMAN_TOOL = {"name", "description", "parameters"}
POLICY = {"allow", "deny"}
MEMORY = {
    "max_events",
    "messages",
    "notes",
    "speakers",
    "budget_aware",
    "writes_record",
    "compacting",
    "summarizing",
    "summarizer",
    "retrieval",
}
MEMORY_MESSAGE = {"role", "text"}
BUDGET_AWARE = {"max_context_tokens", "safety_margin_ratio", "min_events", "reserved_tokens"}
COMPACTING = {"max_context_tokens", "summary_ratio", "safety_margin_ratio", "min_events"}
RETRIEVAL = {"sources", "max_results", "min_query_length"}
SOURCE = {"name", "results", "fails"}
SOURCE_RESULT = {"content", "score", "source"}
SUMMARIZING = {
    "max_context_tokens",
    "tier1_ratio",
    "tier2_ratio",
    "truncate_chars",
    "summary_max_tokens",
    "min_events",
}
ROOM = {
    "id",
    "organization_id",
    "metadata",
    "store_metadata",
    "timers",
    "bindings",
    "participants",
    "status",
}
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
    "router",
}
# A hook added for one room declares no filter and no timeout.
ROOM_HOOK = HOOK - {"timeout_ms", "channel_types", "channel_ids", "directions"}
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
    "hold_after",
    "hold_gate",
    "writes_record",
    "parsed",
    "refuse",
    "after_turn",
}
REFUSE = {"error", "code", "retryable", "provider_message_id"}
WHEN = {"room_id", "text", "type", "source_channel_id", "chain_depth"}
OUTPUT = {"tasks", "observations", "metadata_updates", "response_metadata", "error"}
RESULT = {"action", "event", "reason", "injected_events", "tasks", "observations", "metadata"}
INJECTED = {"event", "target_channel_ids"}
OUTCOME = {"result", "error"}
ERROR = {"name", "reason"}
EXPECT = {
    "timeline",
    "rooms",
    "bindings",
    "framework_events",
    "ai_events",
    "hooks",
    "hook_order",
    "deliveries",
    "contexts",
    "turns",
    "ingested",
    "observations",
    "tasks",
    "actions",
    "lifecycle",
    "models",
    "thinking_events",
    "text_events",
    "streamed",
    "room_callbacks",
    "participants",
    "resolver",
    "media_resolver",
    "tool_audit",
    "session_audit",
}
NEW_ROOM = {"id", "organization_id", "metadata", "timers"}

# Each call's arguments, beside `call` and `organization_id`.
CALLS: dict[str, set[str]] = {
    "process_inbound": {"message", "room_id", "defer_delivery"},
    "delivery.cancel": {"of", "reason", "timeout_ms"},
    "delivery.wait": {"of"},
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
    "create_room": {"room", "orchestration"},
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
    "update_binding_metadata": {"room_id", "channel_id", "metadata"},
    "set_visibility": {"room_id", "channel_id", "visibility"},
    "close_room": {"room_id"},
    "send_greeting": {"room_id", "channel_id", "agent_id", "greeting"},
    "archive_room": {"room_id"},
    "check_timers": {"room_id"},
    "register_channel": {"channel"},
    "unregister_channel": {"channel_id"},
    "register_hook": {"hook"},
    "add_room_hook": {"room_id", "hook"},
    "remove_room_hook": {"room_id", "name"},
    "mute_output": {"room_id", "channel_id"},
    "unmute_output": {"room_id", "channel_id"},
    "process_delivery_status": {"status"},
    "process_webhook": {"meta", "channel_id"},
    "deliver": {
        "room_id",
        "content",
        "channel_id",
        "strategy",
        "metadata",
        "addressed_to",
        "idempotency_key",
        "session_id",
        "chain_depth",
        "instruction",
    },
    "cancel_delivery": {"key", "reason"},
    "steer": {"channel_id", "directive", "room_id"},
}
CALLS |= STORE_ARGUMENTS | READ_MARKER_ARGUMENTS | TASK_ARGUMENTS | MEMBER_ARGUMENTS
CALLS |= INPUT_ARGUMENTS
CALLS |= VERIFICATION_ARGUMENTS
CALLS |= STRATEGY_ARGUMENTS
CALLS |= ROUTING_ARGUMENTS
CALLS |= HANDOFF_ARGUMENTS
CALLS |= DELEGATE_TOOL_ARGUMENTS
CALLS |= DISCUSSION_ARGUMENTS
STEPS = {"await": {"await", "expect"}, "concurrent": {"concurrent"}, "wait_for": {"wait_for"}}
STEPS |= {"open": {"open"}, "idle": {"idle"}, "wait_for_lock": {"wait_for_lock", "waiters"}}
STEPS |= {"lock_releases": {"lock_releases"}}


def unknown_members(scenario: Mapping[str, Any]) -> list[str]:
    """Every member of *scenario* the runner does not know, as ``path: member``."""
    found: list[str] = []
    _members(scenario, TOP, "$", found)
    setup = scenario.get("setup", {})
    _members(setup, SETUP, "setup", found)
    _members(setup.get("kit", {}), KIT, "setup.kit", found)
    if "delivery_backend" in setup.get("kit", {}):
        backend = setup["kit"]["delivery_backend"]
        _members(backend, DELIVERY_BACKEND, "setup.kit.delivery_backend", found)
    if "lock_manager" in setup.get("kit", {}):
        _members(setup["kit"]["lock_manager"], LOCK_MANAGER, "setup.kit.lock_manager", found)
    readers = {"media_resolver": MEDIA_RESOLVER, "attached_media": ATTACHED_MEDIA}
    for reader, members in readers.items():
        if reader in setup.get("kit", {}):
            _members(setup["kit"][reader], members, f"setup.kit.{reader}", found)
    _identity(setup, found)
    for position, channel in enumerate(setup.get("channels", [])):
        _channel(channel, f"setup.channels[{position}]", found)
    for position, room in enumerate(setup.get("rooms", [])):
        path = f"setup.rooms[{position}]"
        _members(room, ROOM, path, found)
        for index, binding in enumerate(room.get("bindings", [])):
            _members(binding, BINDING, f"{path}.bindings[{index}]", found)
    for position, hook in enumerate(setup.get("hooks", [])):
        _hook(hook, f"setup.hooks[{position}]", found)
    for position, strategy in enumerate(setup.get("strategies", [])):
        _strategy(strategy, f"setup.strategies[{position}]", found)
    for position, step in enumerate(scenario.get("steps", [])):
        _step(step, f"steps[{position}]", found)
    expect = scenario.get("expect", {})
    _members(expect, EXPECT, "expect", found)
    for label, outcomes in expect.get("actions", {}).items():
        listed = outcomes.get("$unordered", []) if isinstance(outcomes, dict) else outcomes
        for index, outcome in enumerate(listed):
            _outcome(outcome, f"expect.actions.{label}[{index}]", found)
    return found


def _strategy(strategy: Mapping[str, Any], path: str, found: list[str]) -> None:
    """The members of a scripted strategy, and of its install and uninstall."""
    _members(strategy, STRATEGY, path, found)
    if "loop" in strategy:
        _members(strategy["loop"], LOOP, f"{path}.loop", found)
    if "supervisor" in strategy:
        _members(strategy["supervisor"], SUPERVISOR, f"{path}.supervisor", found)
    if "swarm" in strategy:
        _members(strategy["swarm"], SWARM, f"{path}.swarm", found)
    if "pipeline" in strategy:
        _members(strategy["pipeline"], PIPELINE, f"{path}.pipeline", found)
    if "discussion" in strategy:
        _members(strategy["discussion"], DISCUSSION_MEMBERS, f"{path}.discussion", found)
    for member, known in (("install", STRATEGY_STEP), ("uninstall", STRATEGY_UNINSTALL)):
        step = strategy.get(member, {})
        _members(step, known, f"{path}.{member}", found)
        for index, action in enumerate(step.get("do", [])):
            _call(action, {"as", "expect"}, f"{path}.{member}.do[{index}]", found)


def _identity(setup: Mapping[str, Any], found: list[str]) -> None:
    """The members of the scripted resolver and the identity hooks (§11)."""
    resolver = setup.get("kit", {}).get("identity_resolver")
    if resolver is not None:
        _members(resolver, RESOLVER, "setup.kit.identity_resolver", found)
        for index, answer in enumerate(resolver.get("answers", [])):
            path = f"setup.kit.identity_resolver.answers[{index}]"
            _members(answer, RESOLVER_ANSWER, path, found)
            _members(answer.get("when", {}), RESOLVER_WHEN, f"{path}.when", found)
    for position, hook in enumerate(setup.get("identity_hooks", [])):
        path = f"setup.identity_hooks[{position}]"
        _members(hook, IDENTITY_HOOK, path, found)
        for index, reaction in enumerate(hook.get("reactions", [])):
            _members(reaction, IDENTITY_REACTION, f"{path}.reactions[{index}]", found)
            _members(reaction.get("when", {}), WHEN, f"{path}.reactions[{index}].when", found)


def _members(value: Any, known: set[str], path: str, found: list[str]) -> None:
    if not isinstance(value, Mapping):
        found.append(f"{path}: not an object")
        return
    found.extend(f"{path}: {member}" for member in value if member not in known)


def _channel(channel: Mapping[str, Any], path: str, found: list[str]) -> None:
    _members(channel, CHANNEL, path, found)
    for kind in (
        "parse",
        "on_event",
        "deliver",
        "deliver_stream",
        "on_room_attached",
        "on_room_detached",
    ):
        for index, reaction in enumerate(channel.get(kind, [])):
            _reaction(reaction, f"{path}.{kind}[{index}]", found)
    if "model" in channel:
        _model(channel["model"], f"{path}.model", found)
    if "fallback_model" in channel:
        _model(channel["fallback_model"], f"{path}.fallback_model", found)
    if "retry_policy" in channel:
        _members(channel["retry_policy"], RETRY_POLICY, f"{path}.retry_policy", found)
    for index, tool in enumerate(channel.get("tools", [])):
        _members(tool, TOOL, f"{path}.tools[{index}]", found)
        if "raises" in tool:
            _members(tool["raises"], RAISES, f"{path}.tools[{index}].raises", found)
    if "tool_policy" in channel:
        _members(channel["tool_policy"], POLICY, f"{path}.tool_policy", found)
    if "addresses" in channel:
        _members(channel["addresses"], ADDRESSES, f"{path}.addresses", found)
    if "agent" in channel:
        _members(channel["agent"], AGENT, f"{path}.agent", found)
    if "memory" in channel:
        _memory(channel["memory"], f"{path}.memory", found)
    for room_id, config in channel.get("config_provider", {}).items():
        path_of = f"{path}.config_provider.{room_id}"
        if isinstance(config, list):
            for turn, given in enumerate(config):
                _turn_config(given or {}, f"{path_of}[{turn}]", found)
        else:
            _turn_config(config or {}, path_of, found)
    _skills(channel.get("skills", []), f"{path}.skills", found)
    if "human_input" in channel:
        human = channel["human_input"]
        _members(human, HUMAN_INPUT, f"{path}.human_input", found)
        for index, tool in enumerate(human.get("tools", [])):
            _members(tool, HUMAN_TOOL, f"{path}.human_input.tools[{index}]", found)


def _turn_config(config: Mapping[str, Any], path: str, found: list[str]) -> None:
    _members(config, TURN_CONFIG, path, found)
    for index, tool in enumerate(config.get("tools") or []):
        _members(tool, TURN_TOOL, f"{path}.tools[{index}]", found)
    for index, tool in enumerate(config.get("tool_handler", [])):
        _members(tool, TOOL, f"{path}.tool_handler[{index}]", found)
    _skills(config.get("skills", []), f"{path}.skills", found)


def _skills(skills: list[Mapping[str, Any]], path: str, found: list[str]) -> None:
    for index, skill in enumerate(skills):
        _members(skill, SKILL, f"{path}[{index}]", found)


def _memory(memory: Mapping[str, Any], path: str, found: list[str]) -> None:
    _members(memory, MEMORY, path, found)
    for index, message in enumerate(memory.get("messages", [])):
        _members(message, MEMORY_MESSAGE, f"{path}.messages[{index}]", found)
    if "budget_aware" in memory:
        _members(memory["budget_aware"], BUDGET_AWARE, f"{path}.budget_aware", found)
    if "compacting" in memory:
        _members(memory["compacting"], COMPACTING, f"{path}.compacting", found)
    if "summarizing" in memory:
        _members(memory["summarizing"], SUMMARIZING, f"{path}.summarizing", found)
    if "summarizer" in memory:
        _model(memory["summarizer"], f"{path}.summarizer", found)
    if "retrieval" in memory:
        _retrieval(memory["retrieval"], f"{path}.retrieval", found)


def _retrieval(retrieval: Mapping[str, Any], path: str, found: list[str]) -> None:
    _members(retrieval, RETRIEVAL, path, found)
    for index, source in enumerate(retrieval.get("sources", [])):
        _members(source, SOURCE, f"{path}.sources[{index}]", found)
        for position, result in enumerate(source.get("results", [])):
            _members(result, SOURCE_RESULT, f"{path}.sources[{index}].results[{position}]", found)


def _model(model: Mapping[str, Any], path: str, found: list[str]) -> None:
    _members(model, MODEL, path, found)
    for index, round_ in enumerate(model.get("rounds", [])):
        _members(round_, MODEL_ROUND, f"{path}.rounds[{index}]", found)
        if "refuses" in round_:
            _members(round_["refuses"], REFUSES, f"{path}.rounds[{index}].refuses", found)
        for position, call in enumerate(round_.get("calls", [])):
            _members(call, MODEL_CALL, f"{path}.rounds[{index}].calls[{position}]", found)


def _hook(hook: Mapping[str, Any], path: str, found: list[str], members: set[str] = HOOK) -> None:
    _members(hook, members, path, found)
    for index, reaction in enumerate(hook.get("reactions", [])):
        _reaction(reaction, f"{path}.reactions[{index}]", found)
    if "router" in hook:
        _router(hook["router"], f"{path}.router", found)


def _router(router: Mapping[str, Any], path: str, found: list[str]) -> None:
    _members(router, ROUTER, path, found)
    for index, rule in enumerate(router.get("rules", [])):
        _members(rule, ROUTING_RULE, f"{path}.rules[{index}]", found)
        _members(
            rule.get("conditions", {}), CONDITIONS, f"{path}.rules[{index}].conditions", found
        )


def _reaction(reaction: Mapping[str, Any], path: str, found: list[str]) -> None:
    _members(reaction, REACTION, path, found)
    _members(reaction.get("when", {}), WHEN, f"{path}.when", found)
    _members(reaction.get("output", {}), OUTPUT, f"{path}.output", found)
    _members(reaction.get("refuse", {}), REFUSE, f"{path}.refuse", found)
    if "result" in reaction:
        _members(reaction["result"], RESULT, f"{path}.result", found)
        for index, item in enumerate(reaction["result"].get("injected_events", [])):
            _members(item, INJECTED, f"{path}.result.injected_events[{index}]", found)
    for member in ("do", "after_turn"):
        for index, action in enumerate(reaction.get(member, [])):
            _call(action, {"as", "expect"}, f"{path}.{member}[{index}]", found)


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
    if name == "add_room_hook":
        _hook(call.get("hook", {}), f"{path}.hook", found, ROOM_HOOK)
    if name in ("handoff_handler", "install_router"):
        _router(call.get("router", {}), f"{path}.router", found)
    if name == "save_conversation_state":
        for index, transition in enumerate(call.get("transitions", [])):
            _members(transition, TRANSITION, f"{path}.transitions[{index}]", found)


def _outcome(outcome: Any, path: str, found: list[str]) -> None:
    _members(outcome, OUTCOME, path, found)
    if isinstance(outcome, Mapping) and isinstance(outcome.get("error"), Mapping):
        _members(outcome["error"], ERROR, f"{path}.error", found)
