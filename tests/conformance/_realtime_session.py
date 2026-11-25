"""A scripted speech-to-speech provider behind a real realtime voice channel."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.injection import VoiceInjectionResult
from roomkit.voice.realtime.reasoning import ReasoningBackend, ReasoningOutput, ReasoningRequest
from tests.conformance._realtime_io import (
    IO_ARGUMENTS,
    READ_MEMBERS,
    START_MEMBERS,
    IOProvider,
    channel_options,
    io_call,
)
from tests.conformance._realtime_io import read as io_read
from tests.conformance._realtime_reads import (
    CONFIGURATION,
    READS_ARGUMENTS,
    RecordingTransport,
    configuration,
    on_usage,
    prompt_and_voice,
    reads_call,
)
from tests.conformance._realtime_reads import read as reads_read
from tests.conformance._realtime_tools import (
    BackendRecord,
    act,
    declaration,
    declared_names,
    reasoning_backend,
)
from tests.conformance._voice_pipeline import Trace

if TYPE_CHECKING:
    from tests.conformance._harness import Harness
    from tests.conformance._script import Gates


class _Provider(IOProvider):
    """Expose the provider's outbound wire and its connection-scoped call ids."""

    def __init__(self, spec: Mapping[str, Any], gates: Gates) -> None:
        super().__init__(spec, gates)
        self.injections: list[dict[str, Any]] = []
        self.outstanding: set[str] = set()
        self.declared: list[dict[str, int]] = []
        self.reconfigures: list[dict[str, Any]] = []

    async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
        self.declared.append(declared_names(kwargs.get("tools")))
        await super().connect(session, **kwargs)

    async def inject_text(
        self, session: VoiceSession, text: str, *, role: str = "user", silent: bool = False
    ) -> VoiceInjectionResult | None:
        result = await super().inject_text(session, text, role=role, silent=silent)
        # A not_sent result submitted nothing (README, "Scripted realtime sessions").
        if result is None or result.status != "not_sent":
            self.injections.append({"text": text, "role": role, "silent": silent})
        return result

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        self.outstanding.discard(call_id)
        await super().submit_tool_result(session, call_id, result)

    async def reconfigure(self, session: VoiceSession, **kwargs: Any) -> None:
        self.reconfigures.append(configuration(session, kwargs))
        if kwargs.get("tools") is not None:
            self.declared.append(declared_names(kwargs["tools"]))
        orphaned = sorted(self.outstanding)
        self.outstanding.clear()
        await self.simulate_tool_call_cancellation(session, orphaned)


class _FailingBackend(ReasoningBackend):
    def __init__(self, message: str) -> None:
        self.message = message

    def run(self, request: ReasoningRequest) -> AsyncIterator[ReasoningOutput]:
        raise RuntimeError(self.message)


class ScriptedRealtimeSession:
    """One session per fixture, retained after its end to script late callbacks."""

    def __init__(
        self,
        channel_id: str,
        spec: Mapping[str, Any],
        gates: Gates,
        kit: RoomKit,
        harness: Harness,
        options: Mapping[str, Any],
    ) -> None:
        self.provider = _Provider(spec, gates)
        self.trace: Trace = []
        self._tools = spec.get("tools", {})
        self._gates = gates
        self._kit = kit
        self._session: VoiceSession | None = None
        # Every session started, by room: a channel starts one per room.
        self._rooms: dict[str, VoiceSession] = {}
        self.handlers: list[dict[str, str]] = []
        self.backend = BackendRecord()
        options = dict(options) | channel_options(spec, self.trace) | prompt_and_voice(spec)
        self.usage_reports = on_usage(self.provider, spec, gates)
        self.transport = RecordingTransport()
        if "reasoning" in spec:
            options["reasoning_backend"] = reasoning_backend(
                spec["reasoning"], harness, kit, self.backend
            )
        if "reasoning_error" in spec:
            options["reasoning_backend"] = _FailingBackend(spec["reasoning_error"])
        self.channel = RealtimeVoiceChannel(
            channel_id,
            provider=self.provider,
            transport=self.transport,
            tools=[declaration(name, tool) for name, tool in self._tools.items()],
            tool_handler=None if spec.get("tool_handler") is False else self._handle,
            **options,
        )

    @property
    def session(self) -> VoiceSession:
        """The session started, or the one a held start is creating."""
        session = self._session or self.provider.connecting
        if session is None:
            raise ValueError("start the realtime session first")
        return session

    async def start(self, spec: Mapping[str, Any]) -> None:
        held = self._session is None and self.provider.connecting is not None
        if held or spec["room_id"] in self._rooms:
            raise ValueError("this scripted channel already started a session in that room")
        self.provider.hold = {key: spec[key] for key in START_MEMBERS if key in spec}
        # The session's own prompt, voice and tools, given as it opens
        # (README, ``realtime.start``).
        metadata = prompt_and_voice(spec)
        if (tools := spec.get("tools")) is not None:
            metadata["tools"] = [declaration(name, self._tools.get(name, {})) for name in tools]
        organization = {k: spec[k] for k in ("organization_id",) if k in spec}
        session = await self.channel.start_session(
            spec["room_id"],
            spec["participant_id"],
            object(),
            metadata=metadata or None,
            **organization,
        )
        self._session = self._session or session
        self._rooms[spec["room_id"]] = session

    async def reconfigure(self, spec: Mapping[str, Any]) -> None:
        """``realtime.reconfigure``: the session of the room it names, the
        channel's first when it names none."""
        session = self.session
        if "room_id" in spec:
            if spec["room_id"] not in self._rooms:
                raise ValueError(f"no session of this channel in room {spec['room_id']}")
            session = self._rooms[spec["room_id"]]
        tools = spec.get("tools")
        await self.channel.reconfigure_session(
            session,
            system_prompt=spec.get("system_prompt"),
            voice=spec.get("voice"),
            tools=None if tools is None else [self._declare(name) for name in tools],
        )

    def _declare(self, name: str) -> dict[str, Any]:
        return declaration(name, self._tools.get(name, {}))

    def configure(self, spec: Mapping[str, Any]) -> None:
        self.channel.configure(
            tools=[self._declare(name) for name in spec["tools"]],
            system_prompt=spec.get("system_prompt"),
            voice=spec.get("voice"),
        )

    async def _handle(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools[name]
        try:
            if "gate" in tool:
                await self._gates.hold(tool["gate"])
            if mode := tool.get("reconnect"):
                change = self.channel.reconfigure_session(
                    self.session, system_prompt="Reconfigured"
                )
                if mode == "child":
                    await asyncio.create_task(change)
                else:
                    await change
            await act(tool, self.channel, self.session, self._kit)
            if "after_gate" in tool:
                await self._gates.hold(tool["after_gate"])
            if "delay_ms" in tool:
                # Still working a while past its ending (README, a tool's ``delay_ms``).
                await asyncio.sleep(tool["delay_ms"] / 1000)
        except asyncio.CancelledError:
            self.handlers.append({"name": name, "outcome": "interrupted"})
            raise
        if "fails" in tool:
            self.handlers.append({"name": name, "outcome": "raised"})
            raise RuntimeError(tool["fails"])
        self.handlers.append({"name": name, "outcome": "returned"})
        return tool["result"]

    def read(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "injections": list(self.provider.injections),
            "results": [
                {"call_id": key, "result": result} for _, key, result in self.provider.tool_results
            ],
            "handlers": list(self.handlers),
            "declared": list(self.provider.declared),
            "backend_requests": list(self.backend.requests),
            "backend_calls": list(self.backend.calls),
            **io_read(self.provider, self.channel, self.trace, spec),
            **reads_read(
                self.provider,
                self.transport,
                self._session,
                self.provider.reconfigures,
                self.usage_reports,
            ),
        }


REALTIME_ARGUMENTS = {
    "realtime.start": {"channel_id", "room_id", "participant_id", "tools"}
    | START_MEMBERS
    | set(CONFIGURATION),
    "realtime.end": {"channel_id"},
    "realtime.read": {"channel_id"} | READ_MEMBERS,
    "realtime.inject": {"channel_id", "text", "role"},
    "realtime.tool_call": {"channel_id", "call_id", "name", "arguments", "arguments_text"},
    "realtime.cancel": {"channel_id", "call_ids"},
    "realtime.delegate": {"channel_id", "delegation_id", "target"},
    "realtime.error": {"channel_id", "code", "message"},
    "realtime.configure": {"channel_id", "tools", "system_prompt", "voice"},
    **IO_ARGUMENTS,
    **READS_ARGUMENTS,
}


async def realtime_call(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    fixture = harness.realtime_sessions[spec["channel_id"]]
    provider = fixture.provider
    match spec["call"]:
        case "realtime.start":
            await fixture.start(spec)
        case "realtime.end":
            await fixture.channel.end_session(fixture.session)
        case "realtime.read":
            return fixture.read(spec)
        case "realtime.inject":
            await fixture.channel.inject_text(fixture.session, spec["text"], role=spec["role"])
        case "realtime.tool_call":
            provider.outstanding.add(spec["call_id"])
            # Text in place of an object: arguments as the model wrote them.
            arguments = spec.get("arguments_text", spec.get("arguments", {}))
            await provider.simulate_tool_call(
                fixture.session, spec["call_id"], spec["name"], arguments
            )
        case "realtime.cancel":
            provider.outstanding.difference_update(spec["call_ids"])
            await provider.simulate_tool_call_cancellation(fixture.session, spec["call_ids"])
        case "realtime.delegate":
            await provider.simulate_delegation(
                fixture.session, spec["delegation_id"], spec["target"]
            )
        case "realtime.error":
            await provider.simulate_error(fixture.session, spec["code"], spec["message"])
        case "realtime.configure":
            fixture.configure(spec)
        case _:
            raise ValueError(f"Unknown realtime call {spec['call']}")
    return None


REALTIME_CALLS = (
    dict.fromkeys(REALTIME_ARGUMENTS, realtime_call)
    | dict.fromkeys(IO_ARGUMENTS, io_call)
    | dict.fromkeys(READS_ARGUMENTS, reads_call)
)
