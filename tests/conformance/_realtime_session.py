"""A scripted speech-to-speech provider behind a real realtime voice channel."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.injection import VoiceInjectionResult
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import ReasoningBackend, ReasoningOutput, ReasoningRequest
from tests.conformance._realtime_tools import (
    BackendRecord,
    act,
    declaration,
    declared_names,
    reasoning_backend,
)

if TYPE_CHECKING:
    from tests.conformance._harness import Harness
    from tests.conformance._script import Gates


class _Provider(MockRealtimeProvider):
    """Expose the provider's outbound wire and its connection-scoped call ids."""

    def __init__(self) -> None:
        super().__init__()
        self.injections: list[dict[str, Any]] = []
        self.outstanding: set[str] = set()
        self.declared: list[dict[str, int]] = []

    async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
        self.declared.append(declared_names(kwargs.get("tools")))
        await super().connect(session, **kwargs)

    async def inject_text(
        self, session: VoiceSession, text: str, *, role: str = "user", silent: bool = False
    ) -> VoiceInjectionResult:
        self.injections.append({"text": text, "role": role, "silent": silent})
        return await super().inject_text(session, text, role=role, silent=silent)

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        self.outstanding.discard(call_id)
        await super().submit_tool_result(session, call_id, result)

    async def reconfigure(self, session: VoiceSession, **kwargs: Any) -> None:
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
        self.provider = _Provider()
        self._tools = spec.get("tools", {})
        self._gates = gates
        self._kit = kit
        self._session: VoiceSession | None = None
        self.handlers: list[dict[str, str]] = []
        self.backend = BackendRecord()
        options = dict(options)
        if "reasoning" in spec:
            options["reasoning_backend"] = reasoning_backend(
                spec["reasoning"], harness, kit, self.backend
            )
        if "reasoning_error" in spec:
            options["reasoning_backend"] = _FailingBackend(spec["reasoning_error"])
        self.channel = RealtimeVoiceChannel(
            channel_id,
            provider=self.provider,
            transport=MockRealtimeTransport(),
            tools=[declaration(name, tool) for name, tool in self._tools.items()],
            tool_handler=None if spec.get("tool_handler") is False else self._handle,
            **options,
        )

    @property
    def session(self) -> VoiceSession:
        if self._session is None:
            raise ValueError("start the realtime session first")
        return self._session

    async def start(
        self, room_id: str, participant_id: str, tools: list[str] | None = None
    ) -> None:
        if self._session is not None:
            raise ValueError("this scripted channel already started its session")
        metadata = None
        if tools is not None:
            # The session's own tools, given as it opens (README, ``realtime.start``).
            metadata = {"tools": [declaration(name, self._tools.get(name, {})) for name in tools]}
        self._session = await self.channel.start_session(
            room_id, participant_id, object(), metadata=metadata
        )

    def configure(self, tools: list[str]) -> None:
        self.channel.configure(
            tools=[declaration(name, self._tools.get(name, {})) for name in tools]
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
        except asyncio.CancelledError:
            self.handlers.append({"name": name, "outcome": "interrupted"})
            raise
        if "fails" in tool:
            self.handlers.append({"name": name, "outcome": "raised"})
            raise RuntimeError(tool["fails"])
        self.handlers.append({"name": name, "outcome": "returned"})
        return tool["result"]

    def read(self) -> dict[str, Any]:
        return {
            "injections": list(self.provider.injections),
            "results": [
                {"call_id": key, "result": result} for _, key, result in self.provider.tool_results
            ],
            "handlers": list(self.handlers),
            "declared": list(self.provider.declared),
            "backend_requests": list(self.backend.requests),
            "backend_calls": list(self.backend.calls),
        }


REALTIME_ARGUMENTS = {
    "realtime.start": {"channel_id", "room_id", "participant_id", "tools"},
    "realtime.end": {"channel_id"},
    "realtime.read": {"channel_id"},
    "realtime.inject": {"channel_id", "text", "role"},
    "realtime.tool_call": {"channel_id", "call_id", "name", "arguments", "arguments_text"},
    "realtime.cancel": {"channel_id", "call_ids"},
    "realtime.delegate": {"channel_id", "delegation_id", "target"},
    "realtime.error": {"channel_id", "code", "message"},
    "realtime.configure": {"channel_id", "tools"},
}


async def realtime_call(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    fixture = harness.realtime_sessions[spec["channel_id"]]
    provider = fixture.provider
    match spec["call"]:
        case "realtime.start":
            await fixture.start(spec["room_id"], spec["participant_id"], spec.get("tools"))
        case "realtime.end":
            await fixture.channel.end_session(fixture.session)
        case "realtime.read":
            return fixture.read()
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
            fixture.configure(spec["tools"])
        case _:
            raise ValueError(f"Unknown realtime call {spec['call']}")
    return None


REALTIME_CALLS = dict.fromkeys(REALTIME_ARGUMENTS, realtime_call)
