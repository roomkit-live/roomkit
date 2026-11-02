"""The scripted agent of a scenario: the Python core's own AI channel over a
scripted model and scripted tools.

roomkit-specs ``conformance/README.md`` ("Scripted agents") defines it: an
``ai`` channel with a ``model`` is the implementation's AI channel, whose
provider answers its n-th generation with the model's n-th round, and whose
tools each answer every call one way.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

from roomkit.channels.ai import AIChannel
from roomkit.core.exceptions import ToolFailedError, ToolRefusedError, UnservedToolCallError
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai import AIContext, AIProvider, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.base import (
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    response_stream_events,
)
from roomkit.tasks import CancelTaskTool, TaskStatusTool
from roomkit.tools.human_input import HumanInputToolHandler
from roomkit.tools.policy import ToolPolicy
from tests.conformance._memory import scripted_memory

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from tests.conformance._script import Gates


class ModelPastItsScriptError(Exception):
    """The agent asked the scripted model for a generation its script lacks."""


class ModelFailedError(Exception):
    """What a round's ``fails`` raises, once its deltas are written."""


class ScriptedModel(AIProvider):
    """A provider answering each generation with the next round of a script,
    and keeping what each request carried."""

    def __init__(self, rounds: list[Mapping[str, Any]], gates: Gates) -> None:
        self._rounds = list(rounds)
        self._gates = gates
        self.requests: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return "scripted"

    async def generate(self, context: AIContext) -> AIResponse:
        round_ = self._next_round(context)
        await self._hold(round_)
        return _response(round_)

    async def _hold(self, round_: Mapping[str, Any]) -> None:
        """Hold the round at its ``gate``, before it writes anything."""
        if (gate := round_.get("gate")) is not None:
            await self._gates.hold(gate)

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """The round's text as its deltas, then its calls and its end; a round
        that ``fails`` raises once its deltas are written. The model waits
        between two deltas, so each reaches the reader apart: deltas yielded
        without waiting may cross as one (RFC §10.6.4)."""
        round_ = self._next_round(context)
        await self._hold(round_)
        for position, delta in enumerate(_deltas(round_)):
            if position:
                await asyncio.sleep(0)
            yield StreamTextDelta(text=delta)
        if (message := round_.get("fails")) is not None:
            raise ModelFailedError(message)
        # The calls and the end as generate()'s whole response gives them; a
        # round that ends without its done event stops after its calls.
        for event in response_stream_events(_response(round_).model_copy(update={"content": ""})):
            if isinstance(event, StreamDone) and round_.get("done") is False:
                return
            yield event

    def _next_round(self, context: AIContext) -> Mapping[str, Any]:
        self.requests.append(
            {
                "messages": messages(context),
                "tool_results": tool_results(context),
                "system_prompt": context.system_prompt,
                "tools": {
                    tool.name: position for position, tool in enumerate(context.tools or [])
                },
            }
        )
        position = len(self.requests) - 1
        if position >= len(self._rounds):
            raise ModelPastItsScriptError(f"generation {position + 1} is past the script")
        return self._rounds[position]


def _deltas(round_: Mapping[str, Any]) -> list[str]:
    """The pieces a round writes its text in: its ``deltas``, or its whole text."""
    if "deltas" in round_:
        return list(round_["deltas"])
    text = round_.get("text", "")
    return [text] if text else []


def _response(round_: Mapping[str, Any]) -> AIResponse:
    calls = [
        AIToolCall(
            id=call["id"],
            name=call["name"],
            arguments=dict(call.get("arguments", {})),
            partial=bool(call.get("cut", False)),
        )
        for call in round_.get("calls", [])
    ]
    return AIResponse(
        content="".join(_deltas(round_)),
        finish_reason=round_.get("finish_reason", "tool_calls" if calls else "stop"),
        usage=dict(round_.get("usage", {})),
        tool_calls=calls,
    )


def messages(context: AIContext) -> list[dict[str, Any]]:
    """The messages a request carries, in order: each one's role and its text
    parts joined, empty for a message that has none."""
    return [
        {"role": message.role, "text": _text_of(message.content)} for message in context.messages
    ]


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content if getattr(part, "type", None) == "text")


def tool_results(context: AIContext) -> list[dict[str, Any]]:
    """The tool results a request carries, in order, as the model reads them,
    and whether it reads each call as failed."""
    found: list[dict[str, Any]] = []
    for message in context.messages:
        if isinstance(message.content, str):
            continue
        for part in message.content:
            if getattr(part, "type", None) == "tool_result":
                found.append(
                    {
                        "tool_call_id": part.tool_call_id,
                        "content": part.result,
                        "is_error": part.is_error,
                    }
                )
    return found


_KINDS: dict[str, type[Exception]] = {}


def _error_of_kind(kind: str) -> type[Exception]:
    """An exception class named *kind*: what a tool that ``raises`` raises."""
    if kind not in _KINDS:
        _KINDS[kind] = type(kind, (Exception,), {})
    return _KINDS[kind]


def _tool_handler(tools: list[Mapping[str, Any]], gates: Gates) -> Any:
    by_name = {tool["name"]: tool for tool in tools}

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        tool = by_name.get(name)
        if tool is None:
            # A name the scripted tools do not define: the kit's tools the
            # agent is given serve it (README, "Scripted agents").
            raise UnservedToolCallError(f"{name} is not this handler's")
        if (gate := tool.get("gate")) is not None:
            await gates.hold(gate)
        if "returns" in tool:
            return str(tool["returns"])
        if "raises" in tool:
            raise _error_of_kind(tool["raises"]["kind"])(tool["raises"]["message"])
        if "fails" in tool:
            raise ToolFailedError(tool["fails"])
        if "refuses" in tool:
            raise ToolRefusedError(tool["refuses"])
        if tool.get("declines"):
            raise UnservedToolCallError(f"{name} is not this handler's")
        await asyncio.Event().wait()
        raise AssertionError("a tool that hangs never answers")

    return handler


def scripted_agent(
    spec: Mapping[str, Any], gates: Gates, kit: RoomKit
) -> tuple[AIChannel, dict[str, ScriptedModel]]:
    """The AI channel of a scripted agent, and its models by the name
    ``expect.models`` gives them: its own under its id, its memory's
    summarizer under ``<id>/summarizer``; a round or a tool with a ``gate``
    holds at it; the tools *kit* makes that it is given, after its own."""
    model = ScriptedModel(spec["model"]["rounds"], gates)
    models = {spec["id"]: model}
    tools = spec.get("tools", [])
    kwargs: dict[str, Any] = {}
    if "tool_policy" in spec:
        policy = spec["tool_policy"]
        kwargs["tool_policy"] = ToolPolicy(
            allow=list(policy.get("allow", [])), deny=list(policy.get("deny", []))
        )
    if "max_tool_rounds" in spec:
        kwargs["max_tool_rounds"] = spec["max_tool_rounds"]
    if "max_context_events" in spec:
        kwargs["max_context_events"] = spec["max_context_events"]
    for flag in ("serialize_room_turns", "steer_room_turns"):
        if flag in spec:
            kwargs[flag] = bool(spec[flag])
    if "memory" in spec:
        memory = spec["memory"]
        summarizer = None
        if "summarizer" in memory:
            summarizer = ScriptedModel(memory["summarizer"]["rounds"], gates)
            models[f"{spec['id']}/summarizer"] = summarizer
        kwargs["memory"] = scripted_memory(memory, spec.get("max_context_events", 50), summarizer)
    if "human_input" in spec:
        kwargs["human_input_handler"] = _human_input(spec["human_input"])
    channel = AIChannel(
        spec["id"],
        provider=model,
        tools=[
            AITool(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=dict(tool.get("parameters", {})),
            )
            for tool in tools
        ]
        + _kit_tools(spec.get("kit_tools", []), kit),
        tool_handler=_tool_handler(tools, gates),
        **kwargs,
    )
    return channel, models


def _kit_tools(names: list[str], kit: RoomKit) -> list[Any]:
    """The tools the kit makes that a scripted agent is given as its own
    (README, "Scripted agents")."""
    made = {"task_status": TaskStatusTool, "cancel_task": CancelTaskTool}
    return [made[name](kit) for name in names]


def _human_input(spec: Mapping[str, Any]) -> HumanInputToolHandler:
    """A scripted agent's human-input tools (README, "Scripted agents")."""
    return HumanInputToolHandler(
        set(spec["tool_names"]),
        timeout=spec.get("timeout_ms", 300_000) / 1000,
        tool_definitions=[
            AITool(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=dict(tool.get("parameters", {})),
            )
            for tool in spec.get("tools", [])
        ],
    )


def tool_call_record(payload: ToolCallEvent) -> dict[str, Any]:
    """A tool hook's payload as a scenario reads it (README, "Expectations")."""
    return {
        "name": payload.name,
        "tool_call_id": payload.tool_call_id,
        "arguments": payload.arguments,
        "result": payload.result,
        "is_error": payload.is_error,
        "refused": payload.refused,
        "cancelled": payload.cancelled,
        "error_detail": payload.error_detail,
    }


def patched_tool_call(payload: ToolCallEvent, patch: Mapping[str, Any]) -> ToolCallEvent:
    """A modify's ``event`` read as a patch on a tool hook's payload."""
    return dataclasses.replace(payload, **dict(patch))
