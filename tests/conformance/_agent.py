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
import shutil
import tempfile
import weakref
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roomkit import Agent, AIChannelTurnConfig
from roomkit.channels.ai import AIChannel
from roomkit.core.exceptions import ToolFailedError, ToolRefusedError, UnservedToolCallError
from roomkit.models.channel import RetryPolicy
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai import (
    AIContext,
    AIImagePart,
    AIProvider,
    AIResponse,
    AITextPart,
    AIThinkingPart,
    AITool,
    AIToolCall,
    ModelInfo,
    ModelPricing,
)
from roomkit.providers.ai.base import (
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    response_stream_events,
    stream_call_of,
)
from roomkit.skills.registry import SkillRegistry
from roomkit.tasks import CancelTaskTool, TaskStatusTool
from roomkit.tools import current_response_metadata
from roomkit.tools.human_input import HumanInputToolHandler
from roomkit.tools.policy import ToolPolicy
from tests.conformance._memory import scripted_memory

if TYPE_CHECKING:
    from roomkit.channels._turn_config import ConfigProvider
    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from tests.conformance._audits import Auditors
    from tests.conformance._script import Gates


TOOL_SEARCH_SETTINGS = (
    "tool_search",
    "tool_search_pinned",
    "tool_search_threshold",
    "tool_search_threshold_pct",
    "tool_search_miss_hint",
    "tool_search_threshold_tokens",
)


class ModelPastItsScriptError(Exception):
    """The agent asked the scripted model for a generation its script lacks."""


class ModelFailedError(Exception):
    """What a round's ``fails`` raises, once its deltas are written."""


class ScriptedModel(AIProvider):
    """A provider answering each generation with the next round of a script,
    and keeping what each request carried; *labelled* reads a call's
    arguments as it is written, each ``$<label>`` naming what the run made."""

    def __init__(
        self,
        rounds: list[Mapping[str, Any]],
        gates: Gates,
        *,
        vision: bool = False,
        labelled: Callable[[Any], Any] | None = None,
        pricing: Mapping[str, Any] | None = None,
        context_window: int | None = None,
        name: str = "scripted",
        provider: str = "scripted",
    ) -> None:
        self._context_window = context_window
        self._name = name
        self._provider = provider
        self._rounds = list(rounds)
        self._gates = gates
        self._vision = vision
        self._pricing = ModelPricing.model_validate(pricing) if pricing is not None else None
        self._labelled = labelled
        self.requests: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def supports_vision(self) -> bool:
        return self._vision

    def catalog_entry(self) -> ModelInfo | None:
        """The model's catalogue entry, with the price and the context window
        the scenario gives it; none for a model the scenario gives neither."""
        if self._pricing is None and self._context_window is None:
            return None
        return ModelInfo(
            id=self.model_name, pricing=self._pricing, context_window=self._context_window
        )

    async def generate(self, context: AIContext) -> AIResponse:
        round_ = self._next_round(context)
        await self._hold(round_)
        _refuse(round_)
        return _response(round_, self._labelled)

    async def _hold(self, round_: Mapping[str, Any]) -> None:
        """Hold the round at its ``gate``, before it writes anything."""
        if (gate := round_.get("gate")) is not None:
            await self._gates.hold(gate)

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """The round's text as its deltas, then its calls and its end; a round
        that ``fails`` raises once its deltas are written, once its calls are
        with ``failure_after_calls``, or once its end is with
        ``failure_after_done``. The model waits between two deltas,
        so each reaches the reader apart: deltas yielded without waiting may
        cross as one (RFC §10.6.4)."""
        round_ = self._next_round(context)
        await self._hold(round_)
        if "whole_response" in round_:
            for event in response_stream_events(_response(round_, self._labelled)):
                yield event
            return
        if "stream" in round_:
            for event in _stream_events(round_, self._labelled):
                yield event
                await asyncio.sleep(0)
            return
        for piece in round_.get("thinking", []):
            yield StreamThinkingDelta(thinking=piece)
            await asyncio.sleep(0)
        for position, delta in enumerate(_deltas(round_)):
            if position:
                await asyncio.sleep(0)
            yield StreamTextDelta(text=delta)
        # The calls and the end as generate()'s whole response gives them.
        response = _response(round_, self._labelled)
        events = list(response_stream_events(response.model_copy(update={"content": ""})))
        if round_.get("failure_after_calls"):
            for event in events:
                if not isinstance(event, StreamDone):
                    yield event
            events = [event for event in events if isinstance(event, StreamDone)]
        if round_.get("failure_after_done"):
            for event in events:
                yield event
            events = []
        if (message := round_.get("fails")) is not None:
            raise ModelFailedError(message)
        _refuse(round_)
        # A round that ends without its done event stops after its calls.
        for event in events:
            if isinstance(event, StreamDone) and round_.get("done") is False:
                return
            yield event

    def _next_round(self, context: AIContext) -> Mapping[str, Any]:
        self.requests.append(
            {
                "messages": messages(context),
                "tool_results": tool_results(context),
                "system_prompt": context.system_prompt,
                "temperature": context.temperature,
                "max_tokens": context.max_tokens,
                "reasoning_effort": context.reasoning_effort,
                "thinking_budget": context.thinking_budget,
                "enable_thinking": context.enable_thinking,
                "tools": {
                    tool.name: position for position, tool in enumerate(context.tools or [])
                },
            }
        )
        position = len(self.requests) - 1
        if position >= len(self._rounds):
            raise ModelPastItsScriptError(f"generation {position + 1} is past the script")
        return self._rounds[position]


def _refuse(round_: Mapping[str, Any]) -> None:
    """Raise the provider's refusal a round's ``refuses`` names: an error with
    its HTTP status and its provider, worth another try when it says so (RFC
    §6.4)."""
    if (refusal := round_.get("refuses")) is not None:
        raise ProviderError(
            refusal["message"],
            retryable=bool(refusal.get("retryable", False)),
            status_code=refusal["status"],
            context_overflow=refusal.get("context_overflow"),
            provider=refusal.get("provider", ""),
        )


def _deltas(round_: Mapping[str, Any]) -> list[str]:
    """The pieces a round writes its text in: its ``deltas``, or its whole text."""
    if "deltas" in round_:
        return list(round_["deltas"])
    text = round_.get("text", "")
    return [text] if text else []


def _tool_call(call: Mapping[str, Any], labelled: Callable[[Any], Any] | None) -> AIToolCall:
    written = labelled or (lambda arguments: arguments)
    return AIToolCall(
        id=call["id"],
        name=call["name"],
        arguments=written(dict(call.get("arguments", {}))),
        metadata=dict(call.get("metadata", {})),
        partial=bool(call.get("cut", False)),
    )


def _response(
    round_: Mapping[str, Any], labelled: Callable[[Any], Any] | None = None
) -> AIResponse:
    calls = [_tool_call(call, labelled) for call in round_.get("calls", [])]
    whole = round_.get("whole_response", {})
    parts = whole.get("thinking_parts")
    return AIResponse(
        content="".join(_deltas(round_)),
        thinking=whole.get("thinking"),
        thinking_signature=whole.get("thinking_signature"),
        thinking_parts=None if parts is None else [AIThinkingPart(**part) for part in parts],
        finish_reason=round_.get("finish_reason", "tool_calls" if calls else "stop"),
        usage=dict(round_.get("usage", {})),
        tool_calls=calls,
    )


def _stream_events(
    round_: Mapping[str, Any], labelled: Callable[[Any], Any] | None
) -> Iterator[StreamEvent]:
    """A round's ``stream``, in its order, then its end."""
    called = False
    for event in round_["stream"]:
        if "call" in event:
            called = True
            yield stream_call_of(_tool_call(event["call"], labelled))
        elif "text" in event:
            yield StreamTextDelta(text=event["text"])
        else:
            yield StreamThinkingDelta(**event)
    yield StreamDone(
        finish_reason=round_.get("finish_reason", "tool_calls" if called else "stop"),
        usage=dict(round_.get("usage", {})),
    )


def messages(context: AIContext) -> list[dict[str, Any]]:
    """The messages a request carries, in order: each one's role, its text
    parts joined, empty for a message that has none, and its text and image
    parts in order, None for a text content."""
    return [
        {
            "role": message.role,
            "text": _text_of(message.content),
            "parts": _parts_of(message.content),
            "round": _round_of(message.content),
        }
        for message in context.messages
    ]


def _round_of(content: Any) -> list[dict[str, Any]] | None:
    """A message's reasoning, text and call parts in order, None for a text
    content."""
    if isinstance(content, str):
        return None
    read: list[dict[str, Any]] = []
    for part in content:
        kind = getattr(part, "type", None)
        if kind == "thinking":
            read.append(
                {"thinking": part.thinking, "signature": part.signature, "redacted": part.redacted}
            )
        elif kind == "text":
            read.append({"text": part.text})
        elif kind == "tool_call":
            read.append({"call": part.id, "arguments": part.arguments, "metadata": part.metadata})
    return read


def _parts_of(content: Any) -> list[dict[str, Any]] | None:
    if isinstance(content, str):
        return None
    parts: list[dict[str, Any]] = []
    for part in content:
        kind = getattr(part, "type", None)
        if kind == "text":
            parts.append({"text": part.text})
        elif kind == "image":
            parts.append({"image": part.url, "mime_type": part.mime_type})
    return parts


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
                        "content": result_content(part.result),
                        "is_error": part.is_error,
                    }
                )
    return found


def result_content(result: Any) -> Any:
    """A result as the model reads it: its text, or its text and image
    parts in order, as a message's ``parts`` are written; anything else (no
    result yet) as it is."""
    return _parts_of(result) if isinstance(result, list) else result


def _returned_parts(parts: list[Mapping[str, Any]]) -> list[AITextPart | AIImagePart]:
    """The parts a tool that ``returns_parts`` answers, in order."""
    return [
        AITextPart(text=part["text"])
        if "text" in part
        else AIImagePart(url=part["image"], mime_type=part.get("mime_type"))
        for part in parts
    ]


_KINDS: dict[str, type[Exception]] = {}


def _error_of_kind(kind: str) -> type[Exception]:
    """An exception class named *kind*: what a tool that ``raises`` raises."""
    if kind not in _KINDS:
        _KINDS[kind] = type(kind, (Exception,), {})
    return _KINDS[kind]


def _write_record(spec: Mapping[str, Any]) -> None:
    """Writes a tool's ``writes_record`` on the turn's record, when it answers one."""
    record = current_response_metadata()
    if record is not None:
        record.update(spec.get("writes_record", {}))


def _tool_handler(tools: list[Mapping[str, Any]], gates: Gates) -> Any:
    by_name = {tool["name"]: tool for tool in tools}

    async def handler(name: str, arguments: dict[str, Any]) -> Any:
        tool = by_name.get(name)
        if tool is None:
            # A name the scripted tools do not define: the kit's tools the
            # agent is given serve it (README, "Scripted agents").
            raise UnservedToolCallError(f"{name} is not this handler's")
        _write_record(tool)
        if (gate := tool.get("gate")) is not None:
            await gates.hold(gate)
        if "returns" in tool:
            return str(tool["returns"])
        if "returns_json" in tool:
            return tool["returns_json"]
        if "returns_parts" in tool:
            return _returned_parts(tool["returns_parts"])
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
    spec: Mapping[str, Any],
    gates: Gates,
    kit: RoomKit,
    labelled: Callable[[Any], Any] | None = None,
    auditors: Auditors | None = None,
) -> tuple[AIChannel, dict[str, ScriptedModel]]:
    """The AI channel of a scripted agent, and its models by the name
    ``expect.models`` gives them: its own under its id, its memory's
    summarizer under ``<id>/summarizer``, its fallback under
    ``<id>/fallback``; a round or a tool with a ``gate``
    holds at it; the tools *kit* makes that it is given, after its own; its
    calls' arguments read by *labelled* as the model writes them."""
    model = ScriptedModel(
        spec["model"]["rounds"],
        gates,
        vision=bool(spec["model"].get("vision", False)),
        labelled=labelled,
        pricing=spec["model"].get("pricing"),
        context_window=spec["model"].get("context_window"),
        name=spec["model"].get("name", "scripted"),
        provider=spec["model"].get("provider", "scripted"),
    )
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
    if "max_empty_retries" in spec:
        kwargs["max_empty_retries"] = spec["max_empty_retries"]
    if "max_context_events" in spec:
        kwargs["max_context_events"] = spec["max_context_events"]
    if "tool_timeout_ms" in spec:
        kwargs["tool_timeout_seconds"] = _seconds(spec["tool_timeout_ms"])
    if "turn_budget_tokens" in spec:
        kwargs["turn_budget_tokens"] = spec["turn_budget_tokens"]
    for setting in (
        "system_prompt",
        "temperature",
        "max_tokens",
        "reasoning_effort",
        "thinking_budget",
        "enable_thinking",
        "thinking_coalesce_ms",
        "thinking_coalesce_chars",
        "publish_thinking_text",
    ):
        if setting in spec:
            kwargs[setting] = spec[setting]
    if "config_provider" in spec:
        kwargs["config_provider"] = _config_provider(spec["config_provider"], gates)
    if "skills" in spec:
        kwargs["skills"] = _skill_registry(spec["skills"])
    if "skills_in_prompt" in spec:
        kwargs["skills_in_prompt"] = spec["skills_in_prompt"]
    if "turn_budget_usd" in spec:
        kwargs["turn_budget_usd"] = spec["turn_budget_usd"]
    if "retry_policy" in spec:
        kwargs["retry_policy"] = _retry_policy(spec["retry_policy"])
    if "fallback_model" in spec:
        fallback = ScriptedModel(
            spec["fallback_model"]["rounds"],
            gates,
            vision=bool(spec["fallback_model"].get("vision", False)),
            labelled=labelled,
            pricing=spec["fallback_model"].get("pricing"),
            name=spec["fallback_model"].get("name", "scripted"),
            provider=spec["fallback_model"].get("provider", "scripted"),
        )
        models[f"{spec['id']}/fallback"] = fallback
        kwargs["fallback_provider"] = fallback
    if "max_author_images" in spec:
        kwargs["max_author_images"] = spec["max_author_images"]
    if "describe_empty_event" in spec:
        says = spec["describe_empty_event"]["says"]
        kwargs["describe_empty_event"] = lambda event: says
    if "tool_timeouts_ms" in spec:
        kwargs["tool_timeouts"] = {
            name: _seconds(bound) for name, bound in spec["tool_timeouts_ms"].items()
        }
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
    for setting in TOOL_SEARCH_SETTINGS:
        if setting in spec:
            kwargs[setting] = spec[setting]
    # An identity makes the scripted agent an Agent (README, "Scripted agents").
    build: Callable[..., AIChannel] = AIChannel
    if "agent" in spec:
        build = Agent
        kwargs.update(spec["agent"])
    channel = build(
        spec["id"],
        provider=model,
        tools=[
            AITool(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=dict(tool.get("parameters", {})),
                tags=list(tool.get("tags", [])),
            )
            for tool in tools
            if tool.get("declared", True)
        ]
        + _kit_tools(spec.get("kit_tools", []), kit),
        tool_handler=_audited(_tool_handler(tools, gates), spec, auditors),
        **kwargs,
    )
    return channel, models


def _audited(handler: Any, spec: Mapping[str, Any], auditors: Auditors | None) -> Any:
    """*handler* wrapped to record to the auditor the agent's `audit` names."""
    return handler if auditors is None else auditors.audited(handler, spec)


def _config_provider(by_room: Mapping[str, Any], gates: Gates) -> ConfigProvider:
    """A scripted agent's per-turn configuration: the one the scenario gives
    the binding's room, none for a room it does not name (README, "Scripted
    agents")."""

    turns: dict[str, int] = {}

    async def provider(
        binding: ChannelBinding, context: RoomContext
    ) -> AIChannelTurnConfig | None:
        if context.room.id != binding.room_id:
            raise AssertionError(f"called for {binding.room_id} in {context.room.id}")
        spec = by_room.get(binding.room_id)
        if isinstance(spec, list):
            turn = turns.get(binding.room_id, 0)
            turns[binding.room_id] = turn + 1
            spec = spec[min(turn, len(spec) - 1)]
        if spec is None:
            return None
        settings = {
            key: value
            for key, value in spec.items()
            if key not in ("tools", "tool_handler", "skills")
        }
        config = AIChannelTurnConfig(**settings)
        if "skills" in spec:
            config.skills = _skill_registry(spec["skills"])
        if (tools := spec.get("tools")) is not None:
            config.tools = [
                AITool(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=dict(tool.get("parameters", {})),
                )
                for tool in tools
            ]
        if "tool_handler" in spec:
            config.tool_handler = _tool_handler(spec["tool_handler"], gates)
        return config

    return provider


def _skill_registry(skills: list[Mapping[str, Any]]) -> SkillRegistry:
    """A scripted skill registry: each skill written to a directory of its
    own, removed with the registry, and discovered from there (README,
    "Scripted agents")."""
    root = Path(tempfile.mkdtemp(prefix="roomkit-conformance-skills-"))
    for skill in skills:
        directory = root / skill["name"]
        (directory / "references").mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {skill['name']}\ndescription: {skill['description']}\n---\n"
            f"{skill.get('instructions', '')}",
            encoding="utf-8",
        )
        for name, text in skill.get("references", {}).items():
            (directory / "references" / name).write_text(text, encoding="utf-8")
    registry = SkillRegistry()
    registry.discover(root)
    weakref.finalize(registry, shutil.rmtree, root, ignore_errors=True)
    return registry


def _retry_policy(spec: Mapping[str, Any]) -> RetryPolicy:
    """A scripted agent's retry policy, its delays given in milliseconds
    (README, "Scripted agents")."""
    defaults = RetryPolicy()
    return RetryPolicy(
        max_retries=spec.get("max_retries", defaults.max_retries),
        base_delay_seconds=spec.get("base_delay_ms", defaults.base_delay_seconds * 1000) / 1000,
        max_delay_seconds=spec.get("max_delay_ms", defaults.max_delay_seconds * 1000) / 1000,
        exponential_base=spec.get("exponential_base", defaults.exponential_base),
    )


def _seconds(milliseconds: int | None) -> float | None:
    """A bound in milliseconds as the channel takes it, in seconds."""
    return None if milliseconds is None else milliseconds / 1000


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
        "result": result_content(payload.result),
        "is_error": payload.is_error,
        "refused": payload.refused,
        "cancelled": payload.cancelled,
        "error_detail": payload.error_detail,
    }


def patched_tool_call(payload: ToolCallEvent, patch: Mapping[str, Any]) -> ToolCallEvent:
    """A modify's ``event`` read as a patch on a tool hook's payload."""
    return dataclasses.replace(payload, **dict(patch))
