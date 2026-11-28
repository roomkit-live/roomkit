"""A scripted public Thinker and the real contexts its calls receive."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from roomkit.speaking import LLMThinker, Thinker, Thought, ThoughtEvent

if TYPE_CHECKING:
    from roomkit.providers.ai import AIContext, AIProvider
    from tests.conformance._script import Gates


class ThinkingFailedError(Exception):
    """A configured thinker failure, distinct from unreadable controls."""


def thought_problems(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["thought must be an object"]
    found = [
        f"unknown thought member {key}"
        for key in value
        if key not in {"text", "want_to_say", "urgent"}
    ]
    if "text" in value and not isinstance(value["text"], str):
        found.append("thought.text must be a string")
    if "urgent" in value and not isinstance(value["urgent"], bool):
        found.append("thought.urgent must be a boolean")
    wants = value.get("want_to_say", [])
    if not isinstance(wants, list) or any(not isinstance(item, str) for item in wants):
        found.append("thought.want_to_say must be an array of strings")
    return found


def llm_problems(value: Any) -> list[str]:
    """What is wrong with an LLM thinker's controls (README, ``thinker.llm``)."""
    if not isinstance(value, Mapping):
        return ["thinker.llm must be an object"]
    found = [f"unknown thinker.llm member {key}" for key in value if key not in _LLM_MEMBERS]
    if not isinstance(value.get("model"), Mapping):
        found.append("thinker.llm.model must be a scripted model")
    for key, kind in (("instructions", str), ("reasoning_effort", str)):
        if key in value and not isinstance(value[key], kind):
            found.append(f"thinker.llm.{key} must be a string")
    if "max_tokens" in value and (
        not isinstance(value["max_tokens"], int)
        or isinstance(value["max_tokens"], bool)
        or value["max_tokens"] < 0
    ):
        found.append("thinker.llm.max_tokens must be a non-negative integer")
    timeout = value.get("timeout_ms", 0)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        found.append("thinker.llm.timeout_ms must be a number")
    return found


_LLM_MEMBERS = frozenset({"model", "instructions", "max_tokens", "reasoning_effort", "timeout_ms"})


def thinker_problems(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["thinker must be an object"]
    if "llm" in value:
        found = [f"unknown thinker member {key}" for key in value if key != "llm"]
        return found + llm_problems(value["llm"])
    found = [f"unknown thinker member {key}" for key in value if key != "thoughts"]
    entries = value.get("thoughts")
    if not isinstance(entries, list) or not entries:
        return [*found, "thinker.thoughts must be a nonempty array"]
    for index, entry in enumerate(entries):
        path = f"thoughts[{index}]"
        if not isinstance(entry, Mapping):
            found.append(f"{path}: expected an object")
            continue
        found.extend(
            f"{path}: unknown member {key}"
            for key in entry
            if key not in {"thought", "fail", "gate"}
        )
        if ("thought" in entry) == ("fail" in entry):
            found.append(f"{path}: exactly one of thought and fail is required")
        if "thought" in entry:
            found.extend(f"{path}: {problem}" for problem in thought_problems(entry["thought"]))
        if "fail" in entry and not isinstance(entry["fail"], str):
            found.append(f"{path}.fail must be a string")
        if "gate" in entry and not isinstance(entry["gate"], str):
            found.append(f"{path}.gate must be a string")
    return found


def channel_problems(channel: Mapping[str, Any], path: str) -> list[str]:
    found = []
    if "thinker" in channel:
        found.extend(
            f"{path}.thinker: {problem}" for problem in thinker_problems(channel["thinker"])
        )
    if "think_wait_ms" in channel:
        value = channel["think_wait_ms"]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            found.append(f"{path}.think_wait_ms must be a number")
    return found


class ScriptedThinker(Thinker):
    def __init__(
        self,
        spec: Mapping[str, Any],
        gates: Gates,
        calls: list[Any],
        context_record: Callable[[AIContext], dict[str, Any]],
    ) -> None:
        if problems := thinker_problems(spec):
            raise ValueError("thinker: " + "; ".join(problems))
        self._entries = spec["thoughts"]
        self._gates, self._calls, self._context_record = gates, calls, context_record

    async def think(self, previous: Thought, context: AIContext) -> Thought:
        self._calls.append(
            {"previous": thought_value(previous), "context": self._context_record(context)}
        )
        entry = self._entries[min(len(self._calls), len(self._entries)) - 1]
        if "gate" in entry:
            await self._gates.hold(entry["gate"])
        if "fail" in entry:
            raise ThinkingFailedError(entry["fail"])
        thought = entry["thought"]
        return Thought(**{**thought, "want_to_say": tuple(thought.get("want_to_say", ()))})


def llm_thinker(spec: Mapping[str, Any], model: AIProvider) -> LLMThinker:
    """The implementation's LLM thinker over *model* (README, ``thinker.llm``)."""
    options: dict[str, Any] = {
        key: spec[key] for key in ("instructions", "max_tokens", "reasoning_effort") if key in spec
    }
    if "timeout_ms" in spec:
        options["timeout"] = spec["timeout_ms"] / 1000
    return LLMThinker(model, **options)


def thinking_options(
    channel: Mapping[str, Any],
    gates: Gates,
    observations: dict[str, list[Any]] | None,
    context_record: Callable[[AIContext], dict[str, Any]],
    model: AIProvider | None = None,
) -> dict[str, Any]:
    """The thinker and its wait a channel *spec* gives: a scripted thinker, or
    the LLM thinker over *model*, its scripted model."""
    if problems := channel_problems(channel, "channel"):
        raise ValueError("; ".join(problems))
    options: dict[str, Any] = {}
    calls = observations.setdefault(channel["id"], []) if observations is not None else None
    if "llm" in channel.get("thinker", {}):
        if model is None:
            raise AssertionError("thinker.llm: no scripted model")
        options["thinker"] = llm_thinker(channel["thinker"]["llm"], model)
    elif "thinker" in channel:
        if calls is None:
            raise AssertionError("thinker: no observation sink")
        options["thinker"] = ScriptedThinker(channel["thinker"], gates, calls, context_record)
    if "think_wait_ms" in channel:
        options["think_wait"] = channel["think_wait_ms"] / 1000
    return options


def thought_record(report: ThoughtEvent) -> dict[str, Any]:
    return {
        **dataclasses.asdict(report),
        "thought": thought_value(report.thought),
        "previous": thought_value(report.previous),
    }


def thought_value(thought: Thought) -> dict[str, Any]:
    return {**dataclasses.asdict(thought), "want_to_say": list(thought.want_to_say)}
