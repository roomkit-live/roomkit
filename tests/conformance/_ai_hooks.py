"""The payloads of an AI channel's loop triggers as a scenario reads them, and
a modify's patch on each (roomkit-specs ``conformance/README.md``, "Scripted
agents" and "Expectations").

``BEFORE_AI_GENERATION`` hands an ``AIGenerationEvent``, read as its
``generation``; ``AFTER_TOOL_ROUND`` a ``ToolRoundEvent``, its ``tool_round``;
``ON_AI_RESPONSE`` an ``AIResponseEvent``, its ``response``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.tool_call import AIGenerationEvent, AIResponseEvent, ToolRoundEvent
from roomkit.providers.ai import AIMessage, AITool
from tests.conformance._agent import messages, result_content

_GENERATION_PATCH = frozenset(
    {
        "system_prompt",
        "messages",
        "reasoning_effort",
        "thinking_budget",
        "enable_thinking",
        "response_schema",
        "response_metadata",
        "tools",
        "define",
    }
)
_TOOL_ROUND_PATCH = frozenset({"withdrawn", "messages"})


class UnknownPatchMemberError(Exception):
    """A modify's patch names a member the runner does not know."""


def generation_record(payload: AIGenerationEvent) -> dict[str, Any]:
    """A ``BEFORE_AI_GENERATION`` payload: the context the turn starts from."""
    context = payload.ai_context
    return {
        "channel_id": payload.channel_id,
        "provider_name": payload.provider_name,
        "purpose": payload.purpose,
        "trigger": payload.trigger.id if payload.trigger is not None else None,
        "system_prompt": context.system_prompt,
        "messages": messages(context),
        "tools": [tool.name for tool in context.tools or []],
    }


def patched_generation(payload: AIGenerationEvent, patch: Mapping[str, Any]) -> AIGenerationEvent:
    """A replacement event (``HookResult.modify``): its context's
    ``system_prompt``, ``messages`` (each ``{role, text}``), ``reasoning_effort``,
    ``thinking_budget``, ``enable_thinking``, ``response_schema`` and
    ``response_metadata`` are the patch's,
    its ``tools`` those of the context the patch names, in their order, and
    each tool ``define`` gives in place of the tool of its name, or after
    them."""
    _refuse_unknown(patch, _GENERATION_PATCH)
    context = payload.ai_context
    update: dict[str, Any] = {}
    if "system_prompt" in patch:
        update["system_prompt"] = patch["system_prompt"]
    if "messages" in patch:
        update["messages"] = [
            AIMessage(role=message["role"], content=message["text"])
            for message in patch["messages"]
        ]
    for setting in ("reasoning_effort", "thinking_budget", "enable_thinking", "response_schema"):
        if setting in patch:
            update[setting] = patch[setting]
    if "response_metadata" in patch:
        update["response_metadata"] = ResponseMetadata(patch["response_metadata"])
    if "tools" in patch:
        kept = set(patch["tools"])
        update["tools"] = [tool for tool in context.tools or [] if tool.name in kept]
    if "define" in patch:
        update["tools"] = _defined(update.get("tools", context.tools or []), patch["define"])
    return dataclasses.replace(payload, ai_context=context.model_copy(update=update))


def _defined(tools: list[AITool], definitions: list[Mapping[str, Any]]) -> list[AITool]:
    """*tools* with each of *definitions* in place of the tool of its name,
    or after them."""
    tools = list(tools)
    for definition in definitions:
        tool = AITool(
            name=definition["name"],
            description=definition.get("description", ""),
            parameters=dict(definition.get("parameters", {})),
        )
        at = next((i for i, t in enumerate(tools) if t.name == tool.name), None)
        if at is None:
            tools.append(tool)
        else:
            tools[at] = tool
    return tools


def tool_round_record(payload: ToolRoundEvent) -> dict[str, Any]:
    """An ``AFTER_TOOL_ROUND`` payload: the round the channel ran."""
    return {
        "channel_id": payload.channel_id,
        "round_index": payload.round_index,
        "calls": [
            {
                "tool_call_id": call.id,
                "name": call.name,
                "arguments": call.arguments,
                "metadata": call.metadata,
                "partial": call.partial,
                "garbled": call.garbled,
            }
            for call in payload.calls
        ],
        "results": [
            {
                "tool_call_id": result.tool_call_id,
                "content": result_content(result.result),
                "is_error": result.is_error,
            }
            for result in payload.results
        ],
        "tools": list(payload.tools),
    }


def patched_tool_round(payload: ToolRoundEvent, patch: Mapping[str, Any]) -> ToolRoundEvent:
    """The round's event as the hook acts on it, in place: the names the
    patch's ``withdrawn`` lists withdrawn, its ``messages`` added."""
    _refuse_unknown(patch, _TOOL_ROUND_PATCH)
    payload.withdraw(*patch.get("withdrawn", []))
    for text in patch.get("messages", []):
        payload.add_message(text)
    return payload


def response_record(payload: AIResponseEvent) -> dict[str, Any]:
    """An ``ON_AI_RESPONSE`` payload: the turn as it completed."""
    return {
        "channel_id": payload.channel_id,
        "response_content": payload.response_content,
        "segments": list(payload.segments),
        "tool_calls_count": payload.tool_calls_count,
        "round_count": payload.round_count,
        "loop_end_reason": payload.loop_end_reason,
        "usage": dict(payload.usage),
        "streaming": payload.streaming,
        "thinking": payload.thinking,
        "declared_tools": {
            tool.name: {
                "position": position,
                "description": tool.description,
                "parameters": tool.parameters,
                "origin": tool.origin,
            }
            for position, tool in enumerate(payload.declared_tools)
        },
    }


def _refuse_unknown(patch: Mapping[str, Any], known: frozenset[str]) -> None:
    unknown = set(patch) - known
    if unknown:
        raise UnknownPatchMemberError(f"unknown patch members: {sorted(unknown)}")
