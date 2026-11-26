"""A realtime channel's tools and reasoning (roomkit-specs
``conformance/README.md``, "Scripted realtime sessions"): its tool policy,
skills, Tool Search and result bound (§12.4); what a tool's handler does
once past its gate, an ending of its own among it; and the integrator-side
backend a delegation runs (§12.4.1), a scripted one or an agent."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit import (
    AgentReasoningBackend,
    ReasoningBackend,
    ReasoningOutput,
    ReasoningRequest,
    ToolPolicy,
)
from roomkit.recorder import ChannelRecordingConfig
from tests.conformance._agent import scripted_agent
from tests.conformance._realtime_reads import session_names
from tests.conformance._script_executor import ScriptedScriptExecutor
from tests.conformance._skills import scripted_skill_registry

if TYPE_CHECKING:
    from roomkit import RoomKit
    from roomkit.channels.realtime_voice import RealtimeVoiceChannel
    from roomkit.voice.base import VoiceSession
    from tests.conformance._harness import Harness
    from tests.conformance._script import Gates

# The members of a realtime channel's spec, of one of its tools, of its
# reasoning backend and of one action of a scripted backend's run.
REALTIME = {
    "tools",
    "reasoning_error",
    "tool_policy",
    "skills",
    "script_executor",
    "tool_search",
    "tool_search_pinned",
    "tool_search_threshold",
    "tool_result_max_length",
    "tool_handler",
    "reasoning",
    "reasoning_timeout_ms",
    "recording",
}
REALTIME_TOOL = {
    "gate",
    "result",
    "reconnect",
    "parameters",
    "fails",
    "action",
    "in_task",
    "after_gate",
    "delay_ms",
}
REASONING = {"runs", "agent"}
BACKEND_ACTION = {
    "call",
    "arguments",
    "via",
    "task",
    "join",
    "wait_for",
    "say",
    "spoken",
    "final",
    "delegate",
}
DELEGATE = {"agent_id", "task"}
_TOOL_SEARCH = ("tool_search", "tool_search_pinned", "tool_search_threshold")


def declaration(name: str, tool: Mapping[str, Any]) -> dict[str, Any]:
    """The declaration of the scripted tool *name*: its schema the one its
    script gives, an empty object's when it gives none."""
    parameters = tool.get("parameters", {"type": "object", "properties": {}})
    return {"name": name, "description": name, "parameters": dict(parameters)}


def declared_names(tools: Any) -> dict[str, int]:
    """The tools of one declaration to the provider, by name: each one's
    position (README, ``realtime.read``'s ``declared``)."""
    names = (tool.get("name") for tool in tools or [] if isinstance(tool, Mapping))
    return {name: position for position, name in enumerate(n for n in names if n)}


@dataclass
class BackendRecord:
    """What a scripted backend was handed and what its calls read."""

    requests: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)


def channel_options(channel_id: str, spec: Mapping[str, Any], harness: Harness) -> dict[str, Any]:
    """The realtime channel's keywords its spec sets beyond its tools and its
    backend, read from the scenario: a caller builds them under ``read``."""
    options: dict[str, Any] = {}
    if "tool_policy" in spec:
        options["tool_policy"] = ToolPolicy.model_validate(spec["tool_policy"])
    if "skills" in spec:
        options["skills"] = scripted_skill_registry(spec["skills"])
    if "script_executor" in spec:
        sink = harness.observed.script_executors
        executor = sink.setdefault(channel_id, {"calls": [], "closes": 0})
        options["script_executor"] = ScriptedScriptExecutor(spec["script_executor"], executor)
    for setting in (*_TOOL_SEARCH, "tool_result_max_length"):
        if setting in spec:
            options[setting] = spec[setting]
    if "reasoning_timeout_ms" in spec:
        options["reasoning_timeout_s"] = spec["reasoning_timeout_ms"] / 1000
    if spec.get("recording"):
        # A room that records records its sessions' audio (README, "Scripted
        # realtime sessions").
        options["recording"] = ChannelRecordingConfig(audio=True)
    return options


def reasoning_backend(
    spec: Mapping[str, Any], harness: Harness, kit: RoomKit, record: BackendRecord
) -> ReasoningBackend:
    """The backend *spec* names: its scripted runs, or an agent of its own,
    built as a setup's scripted agent is and never registered with *kit*.
    The implementation's refusal of the agent is the channel's (README,
    ``realtime.reasoning``); a runner's failure to read the agent is not."""
    if "runs" in spec:
        return ScriptedBackend(spec["runs"], harness.gates, kit, record)
    agent_spec = spec["agent"]
    try:
        agent, models = scripted_agent(
            agent_spec,
            harness.gates,
            kit,
            harness.labelled,
            harness.auditors,
            lambda room_id, channel_id, event: harness.record(
                "ingested", room_id, channel_id, event
            ),
            None,
            harness.record_compacted,
            harness.observed.script_executors,
            harness.mcp.by_channel.get(agent_spec["id"], ()),
            continuation_calls=harness.observed.continuations,
            speak_turns=harness.observed.speak_turns,
            thinker_calls=harness.observed.thinker_calls,
            external_handlers=harness.external_handlers,
            sandbox_executors=harness.observed.sandbox_executors,
            # Its own tools only when its spec gives some, which a backend refuses.
            host_handler="tools" in agent_spec,
        )
    except (ValueError, KeyError, TypeError) as error:
        raise AssertionError(f"reasoning.agent: {type(error).__name__}: {error}") from error
    harness.models.update(models)
    return AgentReasoningBackend(agent)


class ScriptedBackend(ReasoningBackend):
    """A backend whose n-th delegation makes the actions of its n-th run, the
    last one repeating (README, ``realtime.reasoning.runs``)."""

    def __init__(
        self,
        runs: list[list[Mapping[str, Any]]],
        gates: Gates,
        kit: RoomKit,
        record: BackendRecord,
    ) -> None:
        self._runs = runs
        self._gates = gates
        self._kit = kit
        self._record = record
        self._delegations = 0

    async def run(self, request: ReasoningRequest) -> AsyncIterator[ReasoningOutput]:
        self._record.requests.append(_request(request))
        script = self._runs[min(self._delegations, len(self._runs) - 1)]
        self._delegations += 1
        tasks: dict[str, asyncio.Future[None]] = {}
        try:
            for action in script:
                if "say" in action:
                    yield ReasoningOutput(
                        action["say"],
                        spoken=action.get("spoken", True),
                        is_final=action.get("final", False),
                    )
                else:
                    await self._act(request, action, tasks)
        finally:
            # A call never joined ends with the run, its outcome retrieved.
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)

    async def _act(
        self,
        request: ReasoningRequest,
        action: Mapping[str, Any],
        tasks: dict[str, asyncio.Future[None]],
    ) -> None:
        if "call" in action:
            work = self._call(request, action)
            if "task" in action:
                # A task of its own, which inherits the delegation's context.
                tasks[action["task"]] = asyncio.ensure_future(work)
            else:
                await work
        elif "join" in action:
            await tasks.pop(action["join"])
        elif "wait_for" in action:
            await self._gates.wait_for(action["wait_for"])
        elif "delegate" in action:
            spec = action["delegate"]
            room_id = request.session.room_id
            await self._kit.delegate(room_id, spec["agent_id"], spec["task"])
        else:
            raise ValueError(f"Unknown backend action {sorted(action)}")

    async def _call(self, request: ReasoningRequest, action: Mapping[str, Any]) -> None:
        """One call through the request's door, recorded once it returned,
        or as cut when it never did."""
        name = action["call"]
        arguments = dict(action.get("arguments", {}))
        try:
            if action.get("via") == "execute_tool":
                if request.execute_tool is None:
                    raise AssertionError("the request has no execute_tool")
                text = await request.execute_tool(name, arguments)
                self._record.calls.append({"name": name, "text": text})
                return
            if request.execute_tool_call is None:
                raise AssertionError("the request has no execute_tool_call")
            done = await request.execute_tool_call(name, arguments)
        except asyncio.CancelledError:
            self._record.calls.append({"name": name, "cut": True})
            raise
        self._record.calls.append(
            {"name": name, "text": done.text, "is_error": done.is_error, "refused": done.refused}
        )


def _request(request: ReasoningRequest) -> dict[str, Any]:
    """A request as ``backend_requests`` reads it."""
    return {
        "delegation_id": request.delegation_id,
        "first": request.first,
        "session": session_names(request.session),
        "transcript": [{"role": line.role, "text": line.text} for line in request.transcript],
        "participant_role": request.participant_role,
        "tools": declared_names(request.tools),
        "unavailable": dict(request.unavailable),
    }


async def act(
    tool: Mapping[str, Any],
    channel: RealtimeVoiceChannel,
    session: VoiceSession,
    kit: RoomKit | None,
) -> None:
    """What the handler of *tool* does once past its gate: its ``action``,
    in a task of its own it awaits when ``in_task`` is ``awaited``."""
    action = tool.get("action")
    if action is None:
        return
    work = _ending(action, channel, session, kit)
    if tool.get("in_task") == "awaited":
        await asyncio.ensure_future(work)
    else:
        await work


def _ending(
    action: str, channel: RealtimeVoiceChannel, session: VoiceSession, kit: RoomKit | None
) -> Awaitable[Any]:
    if action == "end_session":
        return channel.end_session(session)
    if action == "close":
        return channel.close()
    if kit is None:
        raise AssertionError(f"action {action}: no kit")
    if action == "close_kit":
        return kit.close()
    raise ValueError(f"Unknown handler action {action!r}")
