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
from roomkit.channels._realtime_tool_recovery import _coerce_types, _parse_args
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.event import TextContent
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

        await provider.simulate_transcription(session, "call:lookup{city:Paris}", "assistant")
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

        await provider.simulate_transcription(session, "call:lookup{limit:3}", "assistant")
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
            session, "call:lookup{city:Paris,limit:many}", "assistant"
        )
        await asyncio.sleep(0.1)

        assert called == []
        assert any("must be of type integer" in t for t in _injected(provider))

    async def test_an_undeclared_argument_is_refused_before_the_handler(
        self, provider: MockRealtimeProvider
    ) -> None:
        """The schema closed itself, and a spoken call cannot reopen it."""
        called: list[dict[str, Any]] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            called.append(args)
            return "sunny"

        _kit, _channel, session = await _session(provider, "rt-rec-unknown-arg", handler=handler)

        await provider.simulate_transcription(
            session, "call:lookup{city:Paris,country:FR}", "assistant"
        )
        await asyncio.sleep(0.1)

        assert called == []
        assert any("unknown argument 'country'" in t for t in _injected(provider))

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


class TestRecoveredResultsFollowChannelPolicy:
    async def test_an_oversized_result_is_truncated_with_a_notice(
        self, provider: MockRealtimeProvider
    ) -> None:
        """The host's tool_result_max_length governs this path too."""

        async def handler(name: str, args: dict[str, Any]) -> str:
            return "x" * 5000

        channel = RealtimeVoiceChannel(
            "rt-rec-big",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=LOOKUP_TOOL,
            tool_handler=handler,
            tool_result_max_length=500,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-rec-big")
        session = await channel.start_session(room.id, "u1", "ws")

        await provider.simulate_transcription(session, "call:lookup{city:Paris}", "assistant")
        await asyncio.sleep(0.1)

        injected = _injected(provider)[-1]
        assert "truncated" in injected
        assert "5000 chars" in injected

    async def test_a_serving_hook_that_raises_is_reported_to_the_model(
        self, provider: MockRealtimeProvider
    ) -> None:
        """Nothing served the call, so the model hears the failure, not "ok"."""
        kit, _channel, session = await _session(provider, "rt-rec-hookboom")

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="boom")
        async def boom(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            raise RuntimeError("hook is broken")

        await provider.simulate_transcription(session, "call:lookup{city:Paris}", "assistant")
        await asyncio.sleep(0.1)

        injected = _injected(provider)[-1]
        assert "Tool call failed" in injected
        assert "hook is broken" in injected

    async def test_a_handler_result_outranks_a_broken_hook(
        self, provider: MockRealtimeProvider
    ) -> None:
        """The tool did run; a broken observer does not erase its result."""

        async def handler(name: str, args: dict[str, Any]) -> str:
            return "sunny"

        kit, _channel, session = await _session(provider, "rt-rec-hookboom2", handler=handler)

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="boom")
        async def boom(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            raise RuntimeError("hook is broken")

        await provider.simulate_transcription(session, "call:lookup{city:Paris}", "assistant")
        await asyncio.sleep(0.1)

        assert any("sunny" in t for t in _injected(provider))


class TestSpokenCallParsing:
    """The text-to-arguments half: what the model said becomes what runs."""

    def test_an_undeclared_key_becomes_an_argument_the_schema_can_refuse(self) -> None:
        """Swallowed into the previous value, it would pass a string schema."""
        assert _parse_args("city:Paris,country:FR", ["city", "limit"]) == {
            "city": "Paris",
            "country": "FR",
        }

    def test_an_undeclared_key_ending_with_a_declared_name_does_not_open_it(self) -> None:
        """The ``name`` inside ``username:`` is not where ``name``'s value starts."""
        assert _parse_args("username:bob,name:alice", ["name"]) == {
            "username": "bob",
            "name": "alice",
        }

    def test_a_time_after_a_comma_is_a_value_not_a_key(self) -> None:
        assert _parse_args("note:ok, 3:30 pm", ["note"]) == {"note": "ok, 3:30 pm"}

    def test_keys_separated_by_whitespace_still_split(self) -> None:
        assert _parse_args("city:Paris limit:3", ["city", "limit"]) == {
            "city": "Paris",
            "limit": "3",
        }

    def test_a_colon_inside_a_value_is_not_a_boundary(self) -> None:
        assert _parse_args("note:see you at 3:30", ["note"]) == {"note": "see you at 3:30"}

    def test_a_declared_name_inside_a_value_is_not_a_second_boundary(self) -> None:
        """Only the first occurrence of a declared name opens its value."""
        assert _parse_args("task:review the task:2 ticket", ["task"]) == {
            "task": "review the task:2 ticket"
        }

    def test_a_trailing_brace_is_not_part_of_the_value(self) -> None:
        assert _parse_args("city:Paris}", ["city"]) == {"city": "Paris"}

    def test_an_empty_value_is_dropped_rather_than_passed_as_empty(self) -> None:
        assert _parse_args("city:,limit:3", ["city", "limit"]) == {"limit": "3"}

    def test_a_tool_with_no_declared_parameters_parses_nothing(self) -> None:
        assert _parse_args("anything:at all", []) == {}

    def test_declared_types_win_over_the_text_that_carried_them(self) -> None:
        types = {"flag": "boolean", "count": "integer", "ratio": "number", "note": "string"}
        args = {"flag": "yes", "count": "3", "ratio": "1.5", "note": "3"}
        assert _coerce_types(args, types) == {
            "flag": True,
            "count": 3,
            "ratio": 1.5,
            "note": "3",
        }

    def test_a_value_that_cannot_be_coerced_is_left_for_the_schema_to_refuse(self) -> None:
        """Guessing an integer out of "many" would be worse than a refusal."""
        assert _coerce_types({"count": "many"}, {"count": "integer"}) == {"count": "many"}

    async def test_speech_before_the_call_still_reaches_the_room(
        self, provider: MockRealtimeProvider
    ) -> None:
        """The model often narrates, then calls: the narration is real speech."""
        called: list[dict[str, Any]] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            called.append(args)
            return "ok"

        kit, _channel, session = await _session(provider, "rt-rec-prefix", handler=handler)
        room_id = session.room_id

        await provider.simulate_transcription(
            session, "Bien sur, je regarde. call:lookup{city:Paris}", "assistant"
        )
        await asyncio.sleep(0.1)

        assert called == [{"city": "Paris"}]
        timeline = await kit.get_timeline(room_id)
        spoken = [e.content.body for e in timeline if isinstance(e.content, TextContent)]
        assert spoken == ["Bien sur, je regarde."]

    async def test_a_call_with_no_speech_around_it_is_not_spoken(
        self, provider: MockRealtimeProvider
    ) -> None:
        kit, _channel, session = await _session(provider, "rt-rec-silent")
        room_id = session.room_id

        await provider.simulate_transcription(session, "call:lookup{city:Paris}", "assistant")
        await asyncio.sleep(0.1)

        timeline = await kit.get_timeline(room_id)
        assert [e for e in timeline if isinstance(e.content, TextContent)] == []
