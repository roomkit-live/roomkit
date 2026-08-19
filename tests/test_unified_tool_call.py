"""Tests for the unified ON_TOOL_CALL hook across channel types."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    RoomContext,
    RoomKit,
    ToolCallEvent,
)
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.enums import ChannelType
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.registry import SkillRegistry
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rt_provider() -> MockRealtimeProvider:
    return MockRealtimeProvider()


@pytest.fixture
def rt_transport() -> MockRealtimeTransport:
    return MockRealtimeTransport()


@pytest.fixture
def rt_channel(
    rt_provider: MockRealtimeProvider, rt_transport: MockRealtimeTransport
) -> RealtimeVoiceChannel:
    return RealtimeVoiceChannel(
        "rt-voice",
        provider=rt_provider,
        transport=rt_transport,
        system_prompt="Test.",
        voice="alloy",
    )


@pytest.fixture
def ai_provider() -> MockAIProvider:
    return MockAIProvider()


@pytest.fixture
def ai_channel(ai_provider: MockAIProvider) -> AIChannel:
    async def handler(name: str, args: dict[str, Any]) -> str:
        if name == "get_weather":
            return json.dumps({"temp": 22, "city": args.get("city", "unknown")})
        return json.dumps({"error": f"unknown tool: {name}"})

    from roomkit.providers.ai.base import AITool

    return AIChannel(
        "ai-1",
        provider=ai_provider,
        system_prompt="Test AI.",
        tool_handler=handler,
        tools=[
            AITool(
                name="get_weather",
                description="Get weather for a city.",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        ],
    )


@pytest.fixture
async def kit_with_rt(rt_channel: RealtimeVoiceChannel) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(rt_channel)
    return kit


@pytest.fixture
async def kit_with_ai(ai_channel: AIChannel) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(ai_channel)
    return kit


@pytest.fixture
async def kit_with_both(rt_channel: RealtimeVoiceChannel, ai_channel: AIChannel) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(rt_channel)
    kit.register_channel(ai_channel)
    return kit


# ---------------------------------------------------------------------------
# RealtimeVoiceChannel: ON_TOOL_CALL hook fires
# ---------------------------------------------------------------------------


class TestRealtimeVoiceToolCallHook:
    async def test_hook_provides_result(
        self,
        kit_with_rt: RoomKit,
        rt_channel: RealtimeVoiceChannel,
        rt_provider: MockRealtimeProvider,
    ) -> None:
        room = await kit_with_rt.create_room()
        room_id = room.id
        await kit_with_rt.attach_channel(room_id, "rt-voice")
        session = await rt_channel.start_session(room_id, "u1", "ws")

        @kit_with_rt.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="provide")
        async def provide(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            assert event.channel_type == ChannelType.REALTIME_VOICE
            assert event.name == "get_weather"
            assert event.session is not None
            return HookResult(action="allow", metadata={"result": '{"temp": 22}'})

        await rt_provider.simulate_tool_call(session, "c1", "get_weather", {"city": "NYC"})
        await asyncio.sleep(0.1)

        assert len(rt_provider.tool_results) == 1
        _, call_id, result_str = rt_provider.tool_results[0]
        assert call_id == "c1"
        assert json.loads(result_str) == {"temp": 22}

    async def test_handler_and_hook_coexist(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
    ) -> None:
        """tool_handler runs first, hook observes and can override."""

        async def handler(name: str, args: dict[str, Any]) -> str:
            return json.dumps({"handler_ran": True})

        ch = RealtimeVoiceChannel(
            "rt-both",
            provider=rt_provider,
            transport=rt_transport,
            tool_handler=handler,
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-both")
        session = await ch.start_session(room.id, "u1", "ws")

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="observe")
        async def observe(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            # Handler already ran — event.result contains its output
            assert event.result is not None
            parsed = json.loads(event.result)
            assert parsed["handler_ran"] is True
            # Override the result
            return HookResult(action="allow", metadata={"result": '{"overridden": true}'})

        await rt_provider.simulate_tool_call(session, "c2", "do_thing", {})
        await asyncio.sleep(0.1)

        _, _, result_str = rt_provider.tool_results[0]
        assert json.loads(result_str) == {"overridden": True}

    async def test_hook_blocks_tool_call(
        self,
        kit_with_rt: RoomKit,
        rt_channel: RealtimeVoiceChannel,
        rt_provider: MockRealtimeProvider,
    ) -> None:
        room = await kit_with_rt.create_room()
        await kit_with_rt.attach_channel(room.id, "rt-voice")
        session = await rt_channel.start_session(room.id, "u1", "ws")

        @kit_with_rt.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="block")
        async def block(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("not allowed")

        await rt_provider.simulate_tool_call(session, "c3", "dangerous", {})
        await asyncio.sleep(0.1)

        _, _, result_str = rt_provider.tool_results[0]
        result = json.loads(result_str)
        assert "error" in result
        assert "not allowed" in result["error"]

    async def test_no_hook_no_handler_returns_error(
        self,
        kit_with_rt: RoomKit,
        rt_channel: RealtimeVoiceChannel,
        rt_provider: MockRealtimeProvider,
    ) -> None:
        """Without handler or hook result, channel returns a 'no handler' error."""
        room = await kit_with_rt.create_room()
        await kit_with_rt.attach_channel(room.id, "rt-voice")
        session = await rt_channel.start_session(room.id, "u1", "ws")

        # Register a hook that just allows (no result override)
        @kit_with_rt.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="noop")
        async def noop(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult.allow()

        await rt_provider.simulate_tool_call(session, "c4", "unknown", {})
        await asyncio.sleep(0.1)

        _, _, result_str = rt_provider.tool_results[0]
        result = json.loads(result_str)
        assert result == {"status": "ok"}


# ---------------------------------------------------------------------------
# AIChannel: ON_TOOL_CALL hook fires via callback
# ---------------------------------------------------------------------------


class TestAIChannelToolCallHook:
    async def test_callback_injected_on_register(
        self,
        kit_with_ai: RoomKit,
        ai_channel: AIChannel,
    ) -> None:
        """Framework injects _tool_call_hook when registering an AIChannel."""
        assert ai_channel._tool_call_hook is not None

    async def test_hook_observes_ai_tool_call(
        self,
        kit_with_ai: RoomKit,
        ai_channel: AIChannel,
    ) -> None:
        """ON_TOOL_CALL hook fires with channel_type=AI after handler runs."""
        room = await kit_with_ai.create_room()
        await kit_with_ai.attach_channel(room.id, "ai-1")

        observed: list[ToolCallEvent] = []

        @kit_with_ai.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="spy")
        async def spy(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            observed.append(event)
            return HookResult.allow()

        # Trigger tool call by sending an event that causes tool use
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelDirection
        from roomkit.models.event import EventSource, RoomEvent, TextContent
        from roomkit.providers.ai.base import AIResponse, AIToolCall

        # Configure mock to return a tool call then a text response
        ai_channel._provider._ai_responses = [
            AIResponse(
                content="",
                tool_calls=[
                    AIToolCall(id="tc-1", name="get_weather", arguments={"city": "Paris"}),
                ],
            ),
            AIResponse(content="The weather in Paris is 22°C."),
        ]

        event = RoomEvent(
            room_id=room.id,
            source=EventSource(
                channel_id="sms-1",
                channel_type=ChannelType.SMS,
                direction=ChannelDirection.INBOUND,
            ),
            content=TextContent(body="What's the weather in Paris?"),
        )
        binding = ChannelBinding(
            channel_id="ai-1",
            room_id=room.id,
            channel_type=ChannelType.AI,
        )
        context = await kit_with_ai._build_context(room.id)

        await ai_channel.on_event(event, binding, context)

        assert len(observed) == 1
        assert observed[0].channel_type == ChannelType.AI
        assert observed[0].name == "get_weather"
        assert observed[0].arguments == {"city": "Paris"}
        # Result was set by the handler
        assert observed[0].result is not None
        parsed = json.loads(observed[0].result)
        assert parsed["city"] == "Paris"

    async def test_hook_overrides_ai_tool_result(
        self,
        kit_with_ai: RoomKit,
        ai_channel: AIChannel,
    ) -> None:
        """ON_TOOL_CALL hook can override the tool handler's result."""
        room = await kit_with_ai.create_room()
        await kit_with_ai.attach_channel(room.id, "ai-1")

        @kit_with_ai.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="override")
        async def override(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            result = '{"temp": 99, "override": true}'
            return HookResult(action="allow", metadata={"result": result})

        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelDirection
        from roomkit.models.event import EventSource, RoomEvent, TextContent
        from roomkit.providers.ai.base import AIResponse, AIToolCall

        # The provider will see the overridden result in the tool result message
        seen_results: list[str] = []
        original_generate = ai_channel._provider.generate

        async def spy_generate(ctx: Any) -> AIResponse:
            # Check tool results in messages
            for msg in ctx.messages:
                for part in msg.content:
                    if hasattr(part, "result"):
                        seen_results.append(part.result)
            return await original_generate(ctx)

        ai_channel._provider.generate = spy_generate  # type: ignore[assignment]
        ai_channel._provider._ai_responses = [
            AIResponse(
                content="",
                tool_calls=[
                    AIToolCall(id="tc-1", name="get_weather", arguments={"city": "X"}),
                ],
            ),
            AIResponse(content="Done."),
        ]

        event = RoomEvent(
            room_id=room.id,
            source=EventSource(
                channel_id="sms-1",
                channel_type=ChannelType.SMS,
                direction=ChannelDirection.INBOUND,
            ),
            content=TextContent(body="Weather?"),
        )
        binding = ChannelBinding(
            channel_id="ai-1",
            room_id=room.id,
            channel_type=ChannelType.AI,
        )
        context = await kit_with_ai._build_context(room.id)
        await ai_channel.on_event(event, binding, context)

        # The overridden result should have been passed to the provider
        assert any("override" in r for r in seen_results)


# ---------------------------------------------------------------------------
# ToolCallEvent — basic construction
# ---------------------------------------------------------------------------


class TestToolCallEvent:
    def test_construction(self) -> None:
        event = ToolCallEvent(
            channel_id="ch-1",
            channel_type=ChannelType.AI,
            tool_call_id="tc-1",
            name="get_weather",
            arguments={"city": "NYC"},
        )
        assert event.channel_type == ChannelType.AI
        assert event.result is None
        assert event.session is None

    def test_with_result(self) -> None:
        event = ToolCallEvent(
            channel_id="ch-1",
            channel_type=ChannelType.REALTIME_VOICE,
            tool_call_id="tc-2",
            name="search",
            arguments={"q": "test"},
            result='{"found": true}',
        )
        assert event.result == '{"found": true}'

    def test_frozen(self) -> None:
        event = ToolCallEvent(
            channel_id="ch-1",
            channel_type=ChannelType.AI,
            tool_call_id="tc-1",
            name="x",
            arguments={},
        )
        with pytest.raises(AttributeError):
            event.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# H1: fail-closed tool authorization
# ---------------------------------------------------------------------------


class TestToolAuthorizationH1:
    async def test_before_tool_use_fails_closed_on_context_error(self) -> None:
        """If building context for BEFORE_TOOL_USE fails, the tool call is denied
        (fail-closed), never allowed by default."""
        kit = RoomKit()
        callback = kit._build_before_tool_call_hook("ch-1")

        async def boom(room_id: str) -> Any:
            raise RuntimeError("context build failed")

        kit._build_context = boom  # type: ignore[assignment]

        event = ToolCallEvent(
            channel_id="ch-1",
            channel_type=ChannelType.AI,
            tool_call_id="tc-1",
            name="x",
            arguments={},
            room_id="r1",
        )
        assert (await callback(event)).allowed is False

    async def test_ai_invalid_tool_args_rejected_before_handler(
        self, ai_provider: MockAIProvider
    ) -> None:
        """Malformed tool arguments are rejected before the handler runs."""
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelDirection
        from roomkit.models.event import EventSource, RoomEvent, TextContent
        from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall

        called: list[dict[str, Any]] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            called.append(args)
            return json.dumps({"ok": True})

        ch = AIChannel(
            "ai-x",
            provider=ai_provider,
            system_prompt="Test.",
            tool_handler=handler,
            tools=[
                AITool(
                    name="get_weather",
                    description="Weather.",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ],
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "ai-x")

        # Tool call is missing the required `city` argument.
        ai_provider._ai_responses = [
            AIResponse(
                content="",
                tool_calls=[AIToolCall(id="tc-1", name="get_weather", arguments={})],
            ),
            AIResponse(content="done"),
        ]

        event = RoomEvent(
            room_id=room.id,
            source=EventSource(
                channel_id="sms-1",
                channel_type=ChannelType.SMS,
                direction=ChannelDirection.INBOUND,
            ),
            content=TextContent(body="weather?"),
        )
        binding = ChannelBinding(channel_id="ai-x", room_id=room.id, channel_type=ChannelType.AI)
        context = await kit._build_context(room.id)
        await ch.on_event(event, binding, context)

        # The malformed call never reached the handler.
        assert called == []

    async def test_ai_binding_tool_schema_is_enforced(self, ai_provider: MockAIProvider) -> None:
        """Room-bound tools use the same validation as constructor tools."""
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelDirection
        from roomkit.models.event import EventSource, RoomEvent, TextContent
        from roomkit.providers.ai.base import AIResponse, AIToolCall

        handler = AsyncMock(return_value="should not run")
        ch = AIChannel("ai-binding", provider=ai_provider, tool_handler=handler)
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "ai-binding")
        ai_provider._ai_responses = [
            AIResponse(
                content="",
                tool_calls=[AIToolCall(id="tc-binding", name="lookup", arguments={})],
            ),
            AIResponse(content="done"),
        ]
        event = RoomEvent(
            room_id=room.id,
            source=EventSource(
                channel_id="sms-1",
                channel_type=ChannelType.SMS,
                direction=ChannelDirection.INBOUND,
            ),
            content=TextContent(body="lookup"),
        )
        binding = ChannelBinding(
            channel_id="ai-binding",
            room_id=room.id,
            channel_type=ChannelType.AI,
            metadata={
                "tools": [
                    {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ]
            },
        )

        await ch.on_event(event, binding, await kit._build_context(room.id))

        handler.assert_not_awaited()

    async def test_ai_provider_cannot_invent_a_tool_name(
        self, ai_provider: MockAIProvider
    ) -> None:
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelDirection
        from roomkit.models.event import EventSource, RoomEvent, TextContent
        from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall

        handler = AsyncMock(return_value="should not run")
        ch = AIChannel(
            "ai-declared",
            provider=ai_provider,
            tools=[AITool(name="lookup", description="safe", parameters={})],
            tool_handler=handler,
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "ai-declared")
        ai_provider._ai_responses = [
            AIResponse(
                content="",
                tool_calls=[AIToolCall(id="tc-unknown", name="delete_all", arguments={})],
            ),
            AIResponse(content="done"),
        ]
        event = RoomEvent(
            room_id=room.id,
            source=EventSource(
                channel_id="sms-1",
                channel_type=ChannelType.SMS,
                direction=ChannelDirection.INBOUND,
            ),
            content=TextContent(body="do it"),
        )
        binding = ChannelBinding(
            channel_id="ai-declared", room_id=room.id, channel_type=ChannelType.AI
        )

        await ch.on_event(event, binding, await kit._build_context(room.id))

        handler.assert_not_awaited()

    async def test_realtime_before_tool_use_blocks_before_handler(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
    ) -> None:
        """A BEFORE_TOOL_USE block in realtime prevents the handler side effect,
        not just the returned result."""
        called: list[str] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            called.append(name)
            return json.dumps({"handler_ran": True})

        ch = RealtimeVoiceChannel(
            "rt-gate",
            provider=rt_provider,
            transport=rt_transport,
            tool_handler=handler,
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-gate")
        session = await ch.start_session(room.id, "u1", "ws")

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="deny")
        async def deny(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("not allowed")

        await rt_provider.simulate_tool_call(session, "c9", "dangerous", {})
        await asyncio.sleep(0.1)

        # Handler never ran — the side effect was prevented, not merely hidden.
        assert called == []
        _, _, result_str = rt_provider.tool_results[0]
        result = json.loads(result_str)
        assert "error" in result
        assert "not allowed" in result["error"]

    async def test_realtime_before_tool_use_rewrites_arguments_before_handler(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
    ) -> None:
        """Realtime tools execute and report the hook's effective arguments."""
        received: list[dict[str, Any]] = []
        observed: list[dict[str, Any]] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            received.append(args)
            return "ok"

        ch = RealtimeVoiceChannel(
            "rt-rewrite",
            provider=rt_provider,
            transport=rt_transport,
            tools=[
                {
                    "name": "lookup",
                    "description": "Look up a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            tool_handler=handler,
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-rewrite")
        session = await ch.start_session(room.id, "u1", "ws")

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="restore")
        async def restore(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult(action="allow", metadata={"arguments": {"city": "Montreal"}})

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="observe")
        async def observe(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            observed.append(event.arguments)
            return HookResult.allow()

        await rt_provider.simulate_tool_call(session, "c10", "lookup", {"city": "[CITY_1]"})
        await asyncio.sleep(0.1)

        assert received == [{"city": "Montreal"}]
        assert observed == [{"city": "Montreal"}]

    async def test_realtime_before_tool_use_rejects_invalid_rewrite(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
    ) -> None:
        """A hook cannot bypass a realtime tool's declared input schema."""
        handler = AsyncMock(return_value="should not run")
        ch = RealtimeVoiceChannel(
            "rt-invalid-rewrite",
            provider=rt_provider,
            transport=rt_transport,
            tools=[
                {
                    "name": "lookup",
                    "description": "Look up a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            tool_handler=handler,
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-invalid-rewrite")
        session = await ch.start_session(room.id, "u1", "ws")

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="bad")
        async def bad_rewrite(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult(action="allow", metadata={"arguments": {"city": 42}})

        await rt_provider.simulate_tool_call(session, "c11", "lookup", {"city": "Paris"})
        await asyncio.sleep(0.1)

        handler.assert_not_awaited()
        result = json.loads(rt_provider.tool_results[0][2])
        assert "Invalid rewritten arguments" in result["error"]

    async def test_realtime_provider_cannot_invent_a_tool_name(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
    ) -> None:
        handler = AsyncMock(return_value="should not run")
        ch = RealtimeVoiceChannel(
            "rt-undeclared",
            provider=rt_provider,
            transport=rt_transport,
            tools=[{"name": "lookup", "description": "safe", "parameters": {}}],
            tool_handler=handler,
        )
        session = await ch.start_session("room-1", "u1", "ws")

        await rt_provider.simulate_tool_call(session, "c12", "delete_everything", {})
        await asyncio.sleep(0.1)

        handler.assert_not_awaited()
        result = json.loads(rt_provider.tool_results[0][2])
        assert result == {"error": "Tool 'delete_everything' is not declared"}


# ---------------------------------------------------------------------------
# RMK-131: the realtime gate is a property of the channel, not of tool_handler
# ---------------------------------------------------------------------------


class TestRealtimeGateWithoutToolHandler:
    """Hook-only mode — the tool is served by ON_TOOL_CALL, not by a handler.

    The pre-execution gate is a property of the channel: the catalogue check,
    argument validation and BEFORE_TOOL_USE run whether or not a handler
    serves the call.
    """

    @staticmethod
    async def _hook_only_channel(
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
        channel_id: str,
        tools: list[dict[str, Any]] | None,
    ) -> tuple[RoomKit, RealtimeVoiceChannel, VoiceSession]:
        ch = RealtimeVoiceChannel(
            channel_id,
            provider=rt_provider,
            transport=rt_transport,
            tools=tools,
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, channel_id)
        session = await ch.start_session(room.id, "u1", "ws")
        return kit, ch, session

    @staticmethod
    def _lookup_tool() -> list[dict[str, Any]]:
        return [
            {
                "name": "lookup",
                "description": "Look up a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            }
        ]

    async def test_invalid_arguments_are_refused_before_the_hook(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
    ) -> None:
        kit, _ch, session = await self._hook_only_channel(
            rt_provider, rt_transport, "rt-hookonly-args", self._lookup_tool()
        )
        served: list[dict[str, Any]] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="serve")
        async def serve(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            served.append(event.arguments)
            return HookResult(action="allow", metadata={"result": "sunny"})

        await rt_provider.simulate_tool_call(session, "c1", "lookup", {"city": 42})
        await asyncio.sleep(0.1)

        assert served == []
        result = json.loads(rt_provider.tool_results[0][2])
        assert "Invalid arguments" in result["error"]

    async def test_an_unknown_argument_on_a_closed_schema_is_refused(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
    ) -> None:
        kit, _ch, session = await self._hook_only_channel(
            rt_provider, rt_transport, "rt-hookonly-unknown", self._lookup_tool()
        )
        served: list[dict[str, Any]] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="serve")
        async def serve(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            served.append(event.arguments)
            return HookResult.allow()

        await rt_provider.simulate_tool_call(
            session, "c2", "lookup", {"city": "Paris", "country": "FR"}
        )
        await asyncio.sleep(0.1)

        assert served == []
        result = json.loads(rt_provider.tool_results[0][2])
        assert "unknown argument 'country'" in result["error"]

    async def test_a_blocking_before_tool_use_hook_stops_the_call(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
    ) -> None:
        kit, _ch, session = await self._hook_only_channel(
            rt_provider, rt_transport, "rt-hookonly-block", self._lookup_tool()
        )
        served: list[dict[str, Any]] = []
        gated: list[str | None] = []

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="deny")
        async def deny(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            gated.append(event.session.id if event.session else None)
            return HookResult.block("not allowed")

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="serve")
        async def serve(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            served.append(event.arguments)
            return HookResult(action="allow", metadata={"result": "sunny"})

        await rt_provider.simulate_tool_call(session, "c3", "lookup", {"city": "Paris"})
        await asyncio.sleep(0.1)

        # The tool was never served — the block prevented it, not merely hid it.
        assert served == []
        # The gate event names the voice session the call came from, so a hook
        # can decide per call rather than per channel.
        assert gated == [session.id]
        result = json.loads(rt_provider.tool_results[0][2])
        assert "not allowed" in result["error"]

    async def test_a_hook_rewrite_reaches_the_serving_hook(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
    ) -> None:
        kit, _ch, session = await self._hook_only_channel(
            rt_provider, rt_transport, "rt-hookonly-rewrite", self._lookup_tool()
        )
        served: list[dict[str, Any]] = []

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="restore")
        async def restore(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult(action="allow", metadata={"arguments": {"city": "Montreal"}})

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="serve")
        async def serve(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            served.append(event.arguments)
            return HookResult(action="allow", metadata={"result": "sunny"})

        await rt_provider.simulate_tool_call(session, "c4", "lookup", {"city": "[CITY_1]"})
        await asyncio.sleep(0.1)

        assert served == [{"city": "Montreal"}]

    async def test_a_valid_call_is_still_served_by_the_hook(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
    ) -> None:
        """A valid call passes the gate and reaches the serving hook."""
        kit, _ch, session = await self._hook_only_channel(
            rt_provider, rt_transport, "rt-hookonly-ok", self._lookup_tool()
        )
        served: list[dict[str, Any]] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="serve")
        async def serve(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            served.append(event.arguments)
            return HookResult(action="allow", metadata={"result": "sunny"})

        await rt_provider.simulate_tool_call(session, "c5", "lookup", {"city": "Paris"})
        await asyncio.sleep(0.1)

        assert served == [{"city": "Paris"}]
        assert rt_provider.tool_results[0][2] == "sunny"

    async def test_an_empty_catalogue_still_accepts_an_undeclared_name(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
    ) -> None:
        """Dynamic mode: no declarations means the channel cannot judge a name."""
        kit, _ch, session = await self._hook_only_channel(
            rt_provider, rt_transport, "rt-hookonly-dynamic", None
        )
        served: list[str] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="serve")
        async def serve(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            served.append(event.name)
            return HookResult(action="allow", metadata={"result": "done"})

        await rt_provider.simulate_tool_call(session, "c6", "anything_at_all", {"x": 1})
        await asyncio.sleep(0.1)

        assert served == ["anything_at_all"]
        assert rt_provider.tool_results[0][2] == "done"


class TestRealtimeGateSharesItsContext:
    async def test_one_tool_call_reads_the_history_once(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
    ) -> None:
        """The gate and ON_TOOL_CALL both need a context; the room has one history."""

        async def handler(name: str, args: dict[str, Any]) -> str:
            return "ok"

        ch = RealtimeVoiceChannel(
            "rt-carry", provider=rt_provider, transport=rt_transport, tool_handler=handler
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-carry")
        session = await ch.start_session(room.id, "u1", "ws")

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="gate")
        async def gate(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult.allow()

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="serve")
        async def serve(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult.allow()

        reads: list[str] = []
        real_get_conversation = kit._store.get_conversation

        async def counting(room_id: str, **kwargs: Any) -> Any:
            reads.append(room_id)
            return await real_get_conversation(room_id, **kwargs)

        kit._store.get_conversation = counting  # type: ignore[method-assign]

        await rt_provider.simulate_tool_call(session, "c1", "anything", {})
        await asyncio.sleep(0.1)

        assert len(reads) == 1


class TestRealtimeGateParityWithTheAIPath:
    """Three guards the classic AI path applies and the realtime one skipped."""

    @staticmethod
    def _registry_gating(tmp_path: Path, tool: str) -> SkillRegistry:
        skill_dir = tmp_path / "billing"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: billing\ndescription: Billing operations\n"
            f"allowed_tools: {tool}\n---\nUse the billing tools.",
            encoding="utf-8",
        )
        registry = SkillRegistry()
        registry.discover(tmp_path)
        return registry

    async def test_a_skill_gated_tool_is_refused_at_execution_not_only_hidden(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
        tmp_path: Path,
    ) -> None:
        """A model can name a tool it saw before the skill was deactivated."""
        called: list[str] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            called.append(name)
            return "ran"

        ch = RealtimeVoiceChannel(
            "rt-gated",
            provider=rt_provider,
            transport=rt_transport,
            tools=[{"name": "refund", "description": "Refund an order", "parameters": {}}],
            tool_handler=handler,
            skills=self._registry_gating(tmp_path, "refund"),
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-gated")
        session = await ch.start_session(room.id, "u1", "ws")

        await rt_provider.simulate_tool_call(session, "c1", "refund", {})
        await asyncio.sleep(0.1)

        assert called == []
        result = json.loads(rt_provider.tool_results[0][2])
        assert "gated by a skill" in result["error"]

    async def test_an_infrastructure_tool_passes_the_gate_too(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
        tmp_path: Path,
    ) -> None:
        """A host auditing tool use must see the channel's own tools as well."""
        ch = RealtimeVoiceChannel(
            "rt-infra-gate",
            provider=rt_provider,
            transport=rt_transport,
            skills=self._registry_gating(tmp_path, "refund"),
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-infra-gate")
        session = await ch.start_session(room.id, "u1", "ws")

        seen: list[str] = []

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="audit")
        async def audit(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            seen.append(event.name)
            return HookResult.allow()

        await rt_provider.simulate_tool_call(session, "c2", "activate_skill", {"name": "billing"})
        await asyncio.sleep(0.1)

        assert seen == ["activate_skill"]

    async def test_the_gate_announces_its_decision_as_a_framework_event(
        self,
        rt_provider: MockRealtimeProvider,
        rt_transport: MockRealtimeTransport,
    ) -> None:
        handler = AsyncMock(return_value="ok")
        ch = RealtimeVoiceChannel(
            "rt-fw-event", provider=rt_provider, transport=rt_transport, tool_handler=handler
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-fw-event")
        session = await ch.start_session(room.id, "u1", "ws")

        emitted: list[tuple[str, dict[str, Any]]] = []
        real_emit = kit._emit_framework_event

        async def capture(event_type: str, **kwargs: Any) -> Any:
            emitted.append((event_type, kwargs.get("data", {})))
            return await real_emit(event_type, **kwargs)

        kit._emit_framework_event = capture  # type: ignore[method-assign]

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="deny")
        async def deny(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("nope")

        await rt_provider.simulate_tool_call(session, "c3", "anything", {})
        await asyncio.sleep(0.1)

        gate_events = [data for kind, data in emitted if kind == "before_tool_use"]
        assert len(gate_events) == 1
        assert gate_events[0]["tool_name"] == "anything"
        assert gate_events[0]["allowed"] is False
        assert gate_events[0]["reason"] == "nope"
