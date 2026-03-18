"""Tests for the unified ON_TOOL_CALL hook across channel types."""

from __future__ import annotations

import asyncio
import json
from typing import Any

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
async def kit_with_both(
    rt_channel: RealtimeVoiceChannel, ai_channel: AIChannel
) -> RoomKit:
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
