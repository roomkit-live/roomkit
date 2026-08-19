"""Tool calls a realtime model spoke instead of issuing (``call:name{...}``).

Gemini Live occasionally emits a tool call as assistant text rather than
through the function calling API. RoomKit recovers those, so they are a real
tool execution path — and its arguments are rebuilt from free text, which
makes them the least trustworthy the channel handles. They run behind the
same pre-execution gate as an API call, and their outcome travels back as
silent context: the model has no pending FunctionResponse to answer.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomContext, RoomKit, ToolCallEvent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

LOOKUP_TOOL: list[dict[str, Any]] = [
    {
        "name": "lookup",
        "description": "Look up a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    }
]


@pytest.fixture
def provider() -> MockRealtimeProvider:
    return MockRealtimeProvider()


async def _session(
    provider: MockRealtimeProvider,
    channel_id: str,
    *,
    handler: Any = None,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[RoomKit, RealtimeVoiceChannel, VoiceSession]:
    channel = RealtimeVoiceChannel(
        channel_id,
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=tools if tools is not None else LOOKUP_TOOL,
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, channel_id)
    session = await channel.start_session(room.id, "u1", "ws")
    return kit, channel, session


def _injected(provider: MockRealtimeProvider) -> list[str]:
    return [text for _sid, text, _role in provider.injected_texts]


class TestRecoveredCallsAreGated:
    async def test_a_blocking_hook_prevents_the_side_effect(
        self, provider: MockRealtimeProvider
    ) -> None:
        called: list[dict[str, Any]] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            called.append(args)
            return "sunny"

        kit, _channel, session = await _session(provider, "rt-rec-block", handler=handler)

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="deny")
        async def deny(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("lookup is off limits")

        await provider.simulate_transcription(
            session, "call:lookup{city:Paris}", "assistant", True
        )
        await asyncio.sleep(0.1)

        assert called == []
        assert any("denied" in t and "lookup is off limits" in t for t in _injected(provider))

    async def test_a_missing_required_argument_is_refused_before_the_handler(
        self, provider: MockRealtimeProvider
    ) -> None:
        called: list[dict[str, Any]] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            called.append(args)
            return "sunny"

        _kit, _channel, session = await _session(provider, "rt-rec-missing", handler=handler)

        await provider.simulate_transcription(session, "call:lookup{limit:3}", "assistant", True)
        await asyncio.sleep(0.1)

        assert called == []
        assert any("missing required argument 'city'" in t for t in _injected(provider))

    async def test_a_value_that_does_not_coerce_is_refused_before_the_handler(
        self, provider: MockRealtimeProvider
    ) -> None:
        """Spoken text carries no types, and 'many' is not an integer."""
        called: list[dict[str, Any]] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            called.append(args)
            return "sunny"

        _kit, _channel, session = await _session(provider, "rt-rec-args", handler=handler)

        await provider.simulate_transcription(
            session, "call:lookup{city:Paris,limit:many}", "assistant", True
        )
        await asyncio.sleep(0.1)

        assert called == []
        assert any("must be of type integer" in t for t in _injected(provider))

    async def test_a_hook_rewrite_reaches_the_handler(
        self, provider: MockRealtimeProvider
    ) -> None:
        called: list[dict[str, Any]] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            called.append(args)
            return "sunny"

        kit, _channel, session = await _session(provider, "rt-rec-rewrite", handler=handler)

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="restore")
        async def restore(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult(action="allow", metadata={"arguments": {"city": "Montreal"}})

        await provider.simulate_transcription(session, "call:lookup{city:[CITY_1]}", "assistant")
        await asyncio.sleep(0.1)

        assert called == [{"city": "Montreal"}]

    async def test_the_gate_also_runs_without_a_tool_handler(
        self, provider: MockRealtimeProvider
    ) -> None:
        """Hook-only mode: ON_TOOL_CALL serves the tool, the gate still runs."""
        kit, _channel, session = await _session(provider, "rt-rec-hookonly")
        served: list[dict[str, Any]] = []

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="deny")
        async def deny(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("no")

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="serve")
        async def serve(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            served.append(event.arguments)
            return HookResult.allow()

        await provider.simulate_transcription(session, "call:lookup{city:Paris}", "assistant")
        await asyncio.sleep(0.1)

        assert served == []

    async def test_a_denial_never_submits_a_tool_result(
        self, provider: MockRealtimeProvider
    ) -> None:
        """The model spoke the call, so it has no FunctionResponse waiting."""
        kit, _channel, session = await _session(provider, "rt-rec-nosubmit")

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="deny")
        async def deny(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("no")

        await provider.simulate_transcription(session, "call:lookup{city:Paris}", "assistant")
        await asyncio.sleep(0.1)

        assert provider.tool_results == []


class TestRecoveryStillWorks:
    async def test_a_valid_call_runs_and_its_result_is_injected(
        self, provider: MockRealtimeProvider
    ) -> None:
        called: list[dict[str, Any]] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            called.append(args)
            return "22 degrees"

        _kit, _channel, session = await _session(provider, "rt-rec-ok", handler=handler)

        await provider.simulate_transcription(session, "call:lookup{city:Paris}", "assistant")
        await asyncio.sleep(0.1)

        assert called == [{"city": "Paris"}]
        assert any("completed" in t and "22 degrees" in t for t in _injected(provider))
        assert provider.tool_results == []

    async def test_declared_types_are_coerced_before_validation(
        self, provider: MockRealtimeProvider
    ) -> None:
        """Text carries no types: 'limit:3' must reach the handler as an int."""
        called: list[dict[str, Any]] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            called.append(args)
            return "ok"

        _kit, _channel, session = await _session(provider, "rt-rec-coerce", handler=handler)

        await provider.simulate_transcription(
            session, "call:lookup{city:Paris,limit:3}", "assistant"
        )
        await asyncio.sleep(0.1)

        assert called == [{"city": "Paris", "limit": 3}]

    async def test_an_undeclared_name_is_not_recovered_at_all(
        self, provider: MockRealtimeProvider
    ) -> None:
        """The upstream name check still short-circuits before any dispatch."""
        called: list[dict[str, Any]] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            called.append(args)
            return "ok"

        _kit, _channel, session = await _session(provider, "rt-rec-unknown", handler=handler)

        await provider.simulate_transcription(session, "call:drop_table{name:users}", "assistant")
        await asyncio.sleep(0.1)

        assert called == []
        assert _injected(provider) == []
