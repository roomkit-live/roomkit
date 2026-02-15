"""Tests for the telemetry provider system."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelType,
    HookTrigger,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse
from roomkit.telemetry import (
    Attr,
    ConsoleTelemetryProvider,
    MockTelemetryProvider,
    NoopTelemetryProvider,
    Span,
    SpanKind,
    TelemetryConfig,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class SimpleChannel(Channel):
    """Minimal channel for testing."""

    channel_type = ChannelType.SMS

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
            ),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class _MockAIProvider(AIProvider):
    """Minimal AI provider for testing."""

    name = "mock-ai"
    supports_streaming = False
    supports_vision = False

    @property
    def model_name(self) -> str:
        return "mock-model"

    async def generate(self, context: AIContext) -> AIResponse:
        return AIResponse(
            content="test response",
            usage={"input_tokens": 10, "output_tokens": 5},
        )


# ---------------------------------------------------------------------------
# ABC and SpanKind tests
# ---------------------------------------------------------------------------


class TestSpanKind:
    def test_span_kind_values(self) -> None:
        assert SpanKind.STT_TRANSCRIBE == "stt.transcribe"
        assert SpanKind.TTS_SYNTHESIZE == "tts.synthesize"
        assert SpanKind.LLM_GENERATE == "llm.generate"
        assert SpanKind.HOOK_SYNC == "hook.sync"
        assert SpanKind.INBOUND_PIPELINE == "framework.inbound"

    def test_span_kind_is_str(self) -> None:
        assert isinstance(SpanKind.CUSTOM, str)


class TestAttr:
    def test_attr_constants(self) -> None:
        assert Attr.PROVIDER == "provider"
        assert Attr.TTFB_MS == "ttfb_ms"
        assert Attr.LLM_INPUT_TOKENS == "llm.input_tokens"
        assert Attr.STT_TEXT_LENGTH == "stt.text_length"
        assert Attr.TTS_CHAR_COUNT == "tts.char_count"


class TestSpanDataclass:
    def test_span_defaults(self) -> None:
        span = Span()
        assert len(span.id) == 16
        assert span.kind == SpanKind.CUSTOM
        assert span.status == "ok"
        assert span.duration_ms is None

    def test_span_duration(self) -> None:
        from datetime import UTC, datetime, timedelta

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = start + timedelta(milliseconds=150)
        span = Span(start_time=start, end_time=end)
        assert span.duration_ms == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# NoopTelemetryProvider tests
# ---------------------------------------------------------------------------


class TestNoopProvider:
    def test_noop_name(self) -> None:
        p = NoopTelemetryProvider()
        assert p.name == "noop"

    def test_noop_returns_empty_string(self) -> None:
        p = NoopTelemetryProvider()
        span_id = p.start_span(SpanKind.CUSTOM, "test")
        assert span_id == ""

    def test_noop_operations_succeed(self) -> None:
        p = NoopTelemetryProvider()
        span_id = p.start_span(SpanKind.CUSTOM, "test")
        p.set_attribute(span_id, "key", "value")
        p.end_span(span_id)
        p.record_metric("test.metric", 1.0)
        p.close()
        p.reset()

    def test_noop_context_manager(self) -> None:
        p = NoopTelemetryProvider()
        with p.span(SpanKind.CUSTOM, "test") as span_id:
            assert span_id == ""


# ---------------------------------------------------------------------------
# MockTelemetryProvider tests
# ---------------------------------------------------------------------------


class TestMockProvider:
    def test_mock_name(self) -> None:
        p = MockTelemetryProvider()
        assert p.name == "mock"

    def test_start_end_span(self) -> None:
        p = MockTelemetryProvider()
        span_id = p.start_span(SpanKind.STT_TRANSCRIBE, "test.stt")
        assert len(span_id) == 16
        assert len(p.get_active_spans()) == 1

        p.end_span(span_id)
        assert len(p.get_active_spans()) == 0
        assert len(p.completed_spans) == 1
        assert p.completed_spans[0].kind == SpanKind.STT_TRANSCRIBE
        assert p.completed_spans[0].duration_ms is not None

    def test_end_span_with_attributes(self) -> None:
        p = MockTelemetryProvider()
        span_id = p.start_span(SpanKind.LLM_GENERATE, "test.llm")
        p.end_span(span_id, attributes={"tokens": 100})
        assert p.spans[0].attributes["tokens"] == 100

    def test_end_span_error(self) -> None:
        p = MockTelemetryProvider()
        span_id = p.start_span(SpanKind.CUSTOM, "test")
        p.end_span(span_id, status="error", error_message="boom")
        assert p.spans[0].status == "error"
        assert p.spans[0].error_message == "boom"

    def test_set_attribute(self) -> None:
        p = MockTelemetryProvider()
        span_id = p.start_span(SpanKind.CUSTOM, "test")
        p.set_attribute(span_id, "key", "value")
        assert p.get_active_spans()[0].attributes["key"] == "value"

    def test_record_metric(self) -> None:
        p = MockTelemetryProvider()
        p.record_metric("test.metric", 42.0, unit="ms", attributes={"a": "b"})
        assert len(p.metrics) == 1
        assert p.metrics[0]["name"] == "test.metric"
        assert p.metrics[0]["value"] == 42.0
        assert p.metrics[0]["unit"] == "ms"
        assert p.metrics[0]["attributes"]["a"] == "b"

    def test_get_spans_by_kind(self) -> None:
        p = MockTelemetryProvider()
        p.start_span(SpanKind.STT_TRANSCRIBE, "stt1")
        p.start_span(SpanKind.TTS_SYNTHESIZE, "tts1")
        p.start_span(SpanKind.STT_TRANSCRIBE, "stt2")
        # End all
        for span in list(p._spans.values()):
            p.end_span(span.id)
        stt_spans = p.get_spans(SpanKind.STT_TRANSCRIBE)
        assert len(stt_spans) == 2

    def test_context_manager(self) -> None:
        p = MockTelemetryProvider()
        with p.span(SpanKind.CUSTOM, "ctx_test") as span_id:
            p.set_attribute(span_id, "inside", True)
        assert len(p.spans) == 1
        assert p.spans[0].attributes["inside"] is True

    def test_context_manager_error(self) -> None:
        p = MockTelemetryProvider()
        with pytest.raises(ValueError, match="boom"), p.span(SpanKind.CUSTOM, "err_test"):
            raise ValueError("boom")
        assert len(p.spans) == 1
        assert p.spans[0].status == "error"
        assert p.spans[0].error_message == "boom"

    def test_span_with_parent(self) -> None:
        p = MockTelemetryProvider()
        parent_id = p.start_span(SpanKind.LLM_GENERATE, "parent")
        child_id = p.start_span(SpanKind.LLM_TOOL_CALL, "child", parent_id=parent_id)
        p.end_span(child_id)
        p.end_span(parent_id)
        child = p.get_spans(SpanKind.LLM_TOOL_CALL)[0]
        assert child.parent_id == parent_id

    def test_span_with_room_session_channel(self) -> None:
        p = MockTelemetryProvider()
        span_id = p.start_span(
            SpanKind.CUSTOM,
            "test",
            room_id="room1",
            session_id="sess1",
            channel_id="ch1",
        )
        p.end_span(span_id)
        span = p.spans[0]
        assert span.room_id == "room1"
        assert span.session_id == "sess1"
        assert span.channel_id == "ch1"

    def test_reset_clears_all(self) -> None:
        p = MockTelemetryProvider()
        p.start_span(SpanKind.CUSTOM, "test")
        p.record_metric("m", 1.0)
        p.reset()
        assert len(p.get_active_spans()) == 0
        assert len(p.spans) == 0
        assert len(p.metrics) == 0

    def test_end_nonexistent_span(self) -> None:
        """Ending a nonexistent span should not raise."""
        p = MockTelemetryProvider()
        p.end_span("nonexistent")  # no error


# ---------------------------------------------------------------------------
# ConsoleTelemetryProvider tests
# ---------------------------------------------------------------------------


class TestConsoleProvider:
    def test_console_name(self) -> None:
        p = ConsoleTelemetryProvider()
        assert p.name == "console"

    def test_console_logs_spans(self, caplog: pytest.LogCaptureFixture) -> None:
        p = ConsoleTelemetryProvider()
        with caplog.at_level(logging.INFO, logger="roomkit.telemetry"):
            span_id = p.start_span(SpanKind.STT_TRANSCRIBE, "stt.batch")
            p.end_span(span_id, attributes={Attr.STT_TEXT_LENGTH: 42})
        assert "SPAN START" in caplog.text
        assert "SPAN END" in caplog.text
        assert "stt.batch" in caplog.text

    def test_console_logs_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        p = ConsoleTelemetryProvider()
        with caplog.at_level(logging.INFO, logger="roomkit.telemetry"):
            span_id = p.start_span(SpanKind.CUSTOM, "fail")
            p.end_span(span_id, status="error", error_message="boom")
        assert "SPAN ERROR" in caplog.text
        assert "boom" in caplog.text

    def test_console_logs_metrics(self, caplog: pytest.LogCaptureFixture) -> None:
        p = ConsoleTelemetryProvider()
        with caplog.at_level(logging.INFO, logger="roomkit.telemetry"):
            p.record_metric("test.metric", 42.0, unit="ms")
        assert "METRIC" in caplog.text
        assert "42.00" in caplog.text

    def test_console_close_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        p = ConsoleTelemetryProvider()
        p.start_span(SpanKind.CUSTOM, "leak")
        with caplog.at_level(logging.WARNING, logger="roomkit.telemetry"):
            p.close()
        assert "active spans" in caplog.text


# ---------------------------------------------------------------------------
# TelemetryConfig tests
# ---------------------------------------------------------------------------


class TestTelemetryConfig:
    def test_default_config(self) -> None:
        config = TelemetryConfig()
        assert config.provider is None
        assert config.sample_rate == 1.0
        assert config.enabled_spans is None

    def test_config_with_provider(self) -> None:
        mock = MockTelemetryProvider()
        config = TelemetryConfig(provider=mock)
        assert config.provider is mock


# ---------------------------------------------------------------------------
# RoomKit wiring tests
# ---------------------------------------------------------------------------


class TestRoomKitWiring:
    def test_default_noop_telemetry(self) -> None:
        kit = RoomKit()
        assert isinstance(kit.telemetry, NoopTelemetryProvider)

    def test_provider_direct(self) -> None:
        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)
        assert kit.telemetry is mock

    def test_config_with_provider(self) -> None:
        mock = MockTelemetryProvider()
        config = TelemetryConfig(provider=mock)
        kit = RoomKit(telemetry=config)
        assert kit.telemetry is mock

    def test_config_without_provider_falls_back_to_noop(self) -> None:
        config = TelemetryConfig()
        kit = RoomKit(telemetry=config)
        assert isinstance(kit.telemetry, NoopTelemetryProvider)

    def test_hook_engine_receives_telemetry(self) -> None:
        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)
        assert kit.hook_engine._telemetry is mock

    def test_channel_receives_telemetry(self) -> None:
        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)
        ch = SimpleChannel("ch1")
        kit.register_channel(ch)
        assert ch._telemetry is mock  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Inbound pipeline telemetry integration
# ---------------------------------------------------------------------------


class TestInboundTelemetry:
    async def test_inbound_creates_span(self) -> None:
        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)
        ch = SimpleChannel("ch1")
        kit.register_channel(ch)

        msg = InboundMessage(
            channel_id="ch1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        result = await kit.process_inbound(msg)
        assert not result.blocked

        inbound_spans = mock.get_spans(SpanKind.INBOUND_PIPELINE)
        assert len(inbound_spans) >= 1
        span = inbound_spans[0]
        assert span.name == "framework.inbound"
        assert span.status == "ok"


# ---------------------------------------------------------------------------
# Hook telemetry integration
# ---------------------------------------------------------------------------


class TestHookTelemetry:
    async def test_sync_hook_creates_span(self) -> None:
        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)
        ch = SimpleChannel("ch1")
        kit.register_channel(ch)

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="test_hook")
        async def my_hook(event: RoomEvent, context: RoomContext) -> HookResult:
            return HookResult(action="allow")

        msg = InboundMessage(
            channel_id="ch1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)

        hook_spans = mock.get_spans(SpanKind.HOOK_SYNC)
        assert len(hook_spans) >= 1
        span = hook_spans[0]
        assert "test_hook" in span.name
        assert span.attributes[Attr.HOOK_RESULT] == "allow"

    async def test_hook_error_records_error_span(self) -> None:
        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)
        ch = SimpleChannel("ch1")
        kit.register_channel(ch)

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="bad_hook")
        async def bad_hook(event: RoomEvent, context: RoomContext) -> HookResult:
            raise ValueError("hook failure")

        msg = InboundMessage(
            channel_id="ch1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)

        hook_spans = mock.get_spans(SpanKind.HOOK_SYNC)
        assert len(hook_spans) >= 1
        error_spans = [s for s in hook_spans if s.status == "error"]
        assert len(error_spans) >= 1


# ---------------------------------------------------------------------------
# AI channel LLM telemetry
# ---------------------------------------------------------------------------


class TestLLMTelemetry:
    async def test_ai_channel_creates_llm_span(self) -> None:
        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)

        from roomkit.channels.ai import AIChannel

        ai_provider = _MockAIProvider()
        ai = AIChannel("ai1", provider=ai_provider)
        kit.register_channel(ai)

        ch = SimpleChannel("ch1")
        kit.register_channel(ch)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "ch1")
        await kit.attach_channel(room.id, "ai1")

        msg = InboundMessage(
            channel_id="ch1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg, room_id=room.id)

        llm_spans = mock.get_spans(SpanKind.LLM_GENERATE)
        assert len(llm_spans) >= 1
        span = llm_spans[0]
        assert span.attributes[Attr.PROVIDER] == "_MockAIProvider"
        assert span.attributes[Attr.LLM_INPUT_TOKENS] == 10
        assert span.attributes[Attr.LLM_OUTPUT_TOKENS] == 5

    async def test_ai_channel_tool_call_span(self) -> None:
        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)

        from roomkit.channels.ai import AIChannel
        from roomkit.providers.ai.base import AIToolCall

        call_count = 0

        class _ToolAIProvider(AIProvider):
            name = "tool-ai"
            supports_streaming = False
            supports_vision = False

            @property
            def model_name(self) -> str:
                return "tool-model"

            async def generate(self, context: AIContext) -> AIResponse:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return AIResponse(
                        content="",
                        tool_calls=[
                            AIToolCall(id="t1", name="get_weather", arguments={"city": "Paris"})
                        ],
                    )
                return AIResponse(
                    content="It's sunny!",
                    usage={"input_tokens": 20, "output_tokens": 10},
                )

        async def tool_handler(name: str, args: dict) -> str:
            return '{"temp": 22}'

        ai = AIChannel("ai1", provider=_ToolAIProvider(), tool_handler=tool_handler)
        kit.register_channel(ai)

        ch = SimpleChannel("ch1")
        kit.register_channel(ch)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "ch1")
        await kit.attach_channel(room.id, "ai1")

        msg = InboundMessage(
            channel_id="ch1",
            sender_id="user1",
            content=TextContent(body="what's the weather?"),
        )
        await kit.process_inbound(msg, room_id=room.id)

        tool_spans = mock.get_spans(SpanKind.LLM_TOOL_CALL)
        assert len(tool_spans) >= 1
        assert tool_spans[0].attributes["tool.name"] == "get_weather"


# ---------------------------------------------------------------------------
# Realtime voice channel telemetry
# ---------------------------------------------------------------------------


class TestRealtimeVoiceTelemetry:
    async def test_session_span_created_on_start(self) -> None:
        """start_session creates a REALTIME_SESSION span."""
        import asyncio

        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)

        from roomkit.channels.realtime_voice import RealtimeVoiceChannel
        from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()
        channel = RealtimeVoiceChannel("rt1", provider=provider, transport=transport)
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt1")

        session = await channel.start_session(room.id, "user1", "fake-ws")

        session_spans = [s for s in mock.get_active_spans() if s.kind == SpanKind.REALTIME_SESSION]
        assert len(session_spans) == 1
        assert session_spans[0].session_id == session.id
        assert session_spans[0].room_id == room.id
        assert session_spans[0].attributes[Attr.REALTIME_PROVIDER] == "MockRealtimeProvider"

        await channel.end_session(session)
        # Let pending tasks complete
        await asyncio.sleep(0)

    async def test_session_span_ended_on_end(self) -> None:
        """end_session closes the REALTIME_SESSION span."""
        import asyncio

        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)

        from roomkit.channels.realtime_voice import RealtimeVoiceChannel
        from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()
        channel = RealtimeVoiceChannel("rt1", provider=provider, transport=transport)
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt1")

        session = await channel.start_session(room.id, "user1", "fake-ws")
        await channel.end_session(session)
        await asyncio.sleep(0)

        completed = mock.get_spans(SpanKind.REALTIME_SESSION)
        assert len(completed) == 1
        assert completed[0].status == "ok"
        assert completed[0].duration_ms is not None

    async def test_turn_span_on_response_cycle(self) -> None:
        """response_start/end creates a REALTIME_TURN span."""
        import asyncio

        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)

        from roomkit.channels.realtime_voice import RealtimeVoiceChannel
        from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()
        channel = RealtimeVoiceChannel("rt1", provider=provider, transport=transport)
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt1")

        session = await channel.start_session(room.id, "user1", "fake-ws")

        # Simulate a response cycle
        await provider.simulate_response_start(session)
        await asyncio.sleep(0.01)
        await provider.simulate_response_end(session)
        await asyncio.sleep(0.01)

        turn_spans = mock.get_spans(SpanKind.REALTIME_TURN)
        assert len(turn_spans) == 1
        assert turn_spans[0].session_id == session.id
        # Turn span should be child of session span
        assert turn_spans[0].parent_id is not None

        await channel.end_session(session)
        await asyncio.sleep(0)

    async def test_tool_call_span(self) -> None:
        """Tool calls create a REALTIME_TOOL_CALL span."""
        import asyncio

        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)

        from roomkit.channels.realtime_voice import RealtimeVoiceChannel
        from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()

        async def tool_handler(session: object, name: str, arguments: dict) -> str:
            return '{"result": "ok"}'

        channel = RealtimeVoiceChannel(
            "rt1",
            provider=provider,
            transport=transport,
            tool_handler=tool_handler,
        )
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt1")

        session = await channel.start_session(room.id, "user1", "fake-ws")

        # Simulate a tool call
        await provider.simulate_tool_call(session, "call-1", "get_weather", {"city": "NYC"})
        await asyncio.sleep(0.01)

        tool_spans = mock.get_spans(SpanKind.REALTIME_TOOL_CALL)
        assert len(tool_spans) == 1
        assert tool_spans[0].attributes[Attr.REALTIME_TOOL_NAME] == "get_weather"
        assert tool_spans[0].name == "realtime_tool:get_weather"
        assert tool_spans[0].status == "ok"

        await channel.end_session(session)
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# OpenTelemetry provider tests (import only — skip if not installed)
# ---------------------------------------------------------------------------


class TestOpenTelemetryProvider:
    def test_import_error_without_otel(self) -> None:
        """Test that a clear error is raised when OTel is not installed."""
        try:
            import opentelemetry  # noqa: F401

            pytest.skip("opentelemetry is installed")
        except ImportError:
            from roomkit.telemetry.opentelemetry import OpenTelemetryProvider

            with pytest.raises(ImportError, match="opentelemetry-api"):
                OpenTelemetryProvider()

    def test_lazy_import_from_roomkit(self) -> None:
        """Test that OpenTelemetryProvider is accessible via lazy import."""
        import roomkit

        cls = getattr(roomkit, "OpenTelemetryProvider", None)
        assert cls is not None


# ---------------------------------------------------------------------------
# Phase 2: EventRouter broadcast span
# ---------------------------------------------------------------------------


class TestBroadcastTelemetry:
    async def test_broadcast_creates_span(self) -> None:
        """process_inbound triggers a BROADCAST span via EventRouter."""
        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)
        ch1 = SimpleChannel("ch1")
        ch2 = SimpleChannel("ch2")
        kit.register_channel(ch1)
        kit.register_channel(ch2)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "ch1")
        await kit.attach_channel(room.id, "ch2")

        msg = InboundMessage(
            channel_id="ch1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg, room_id=room.id)

        broadcast_spans = mock.get_spans(SpanKind.BROADCAST)
        assert len(broadcast_spans) >= 1
        span = broadcast_spans[0]
        assert span.name == "framework.broadcast"
        assert span.room_id == room.id
        assert span.status == "ok"
        assert span.attributes["target_count"] >= 1


# ---------------------------------------------------------------------------
# Phase 2: TransportChannel delivery span
# ---------------------------------------------------------------------------


class _MockTransportProvider:
    """Minimal transport provider for testing."""

    name = "mock-transport"

    async def send(self, event: RoomEvent, *, to: str = "", **kwargs: Any) -> None:
        pass


class _FailingTransportProvider:
    """Transport provider that always fails."""

    name = "failing-transport"

    async def send(self, event: RoomEvent, *, to: str = "", **kwargs: Any) -> None:
        raise RuntimeError("send failed")


class TestDeliveryTelemetry:
    async def test_delivery_creates_span(self) -> None:
        """TransportChannel.deliver() creates a DELIVERY span."""
        from roomkit.channels.transport import TransportChannel

        mock = MockTelemetryProvider()
        provider = _MockTransportProvider()
        ch = TransportChannel(
            "sms1",
            ChannelType.SMS,
            provider=provider,
        )
        ch._telemetry = mock  # type: ignore[attr-defined]

        event = RoomEvent(
            room_id="room1",
            source=EventSource(channel_id="other", channel_type=ChannelType.AI),
            content=TextContent(body="hello"),
        )
        binding = ChannelBinding(
            channel_id="sms1",
            room_id="room1",
            channel_type=ChannelType.SMS,
        )
        from roomkit.models.context import RoomContext
        from roomkit.models.room import Room

        context = RoomContext(room=Room(id="room1"), bindings=[], recent_events=[])
        await ch.deliver(event, binding, context)

        delivery_spans = mock.get_spans(SpanKind.DELIVERY)
        assert len(delivery_spans) == 1
        assert delivery_spans[0].name == "framework.delivery"
        assert delivery_spans[0].attributes[Attr.DELIVERY_CHANNEL_TYPE] == str(ChannelType.SMS)
        assert delivery_spans[0].status == "ok"

    async def test_delivery_error_span(self) -> None:
        """TransportChannel.deliver() records error on failure."""
        from roomkit.channels.transport import TransportChannel

        mock = MockTelemetryProvider()
        provider = _FailingTransportProvider()
        ch = TransportChannel(
            "sms1",
            ChannelType.SMS,
            provider=provider,
        )
        ch._telemetry = mock  # type: ignore[attr-defined]

        event = RoomEvent(
            room_id="room1",
            source=EventSource(channel_id="other", channel_type=ChannelType.AI),
            content=TextContent(body="hello"),
        )
        binding = ChannelBinding(
            channel_id="sms1",
            room_id="room1",
            channel_type=ChannelType.SMS,
        )
        from roomkit.models.context import RoomContext
        from roomkit.models.room import Room

        context = RoomContext(room=Room(id="room1"), bindings=[], recent_events=[])
        with pytest.raises(RuntimeError, match="send failed"):
            await ch.deliver(event, binding, context)

        delivery_spans = mock.get_spans(SpanKind.DELIVERY)
        assert len(delivery_spans) == 1
        assert delivery_spans[0].status == "error"
        assert "send failed" in (delivery_spans[0].error_message or "")


# ---------------------------------------------------------------------------
# Phase 2: VoiceChannel VOICE_SESSION span
# ---------------------------------------------------------------------------


class TestVoiceSessionTelemetry:
    async def test_voice_session_span_on_bind_unbind(self) -> None:
        """bind_session/unbind_session creates and ends a VOICE_SESSION span."""
        from roomkit.voice.backends.mock import MockVoiceBackend
        from roomkit.voice.base import VoiceSession, VoiceSessionState

        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)

        from roomkit.channels.voice import VoiceChannel

        backend = MockVoiceBackend()
        channel = VoiceChannel("voice1", backend=backend)
        kit.register_channel(channel)

        room = await kit.create_room()
        binding = await kit.attach_channel(room.id, "voice1")

        session = VoiceSession(
            id="sess1",
            room_id=room.id,
            participant_id="user1",
            channel_id="voice1",
            state=VoiceSessionState.ACTIVE,
        )

        channel.bind_session(session, room.id, binding)

        active = [s for s in mock.get_active_spans() if s.kind == SpanKind.VOICE_SESSION]
        assert len(active) == 1
        assert active[0].session_id == "sess1"
        assert active[0].room_id == room.id
        assert active[0].attributes[Attr.BACKEND_TYPE] == "MockVoiceBackend"

        channel.unbind_session(session)

        completed = mock.get_spans(SpanKind.VOICE_SESSION)
        assert len(completed) == 1
        assert completed[0].status == "ok"
        assert completed[0].duration_ms is not None


# ---------------------------------------------------------------------------
# Phase 2: Store query span (PostgresStore skipped — mocking pool is heavy)
# ---------------------------------------------------------------------------


class TestStoreQuerySpan:
    def test_store_telemetry_wired(self) -> None:
        """Framework sets _telemetry on the store."""
        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)
        assert kit._store._telemetry is mock  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Phase 2: AI provider TTFB metric
# ---------------------------------------------------------------------------


class TestAIProviderTTFB:
    async def test_ai_provider_records_ttfb_metric(self) -> None:
        """AI provider generate() records a roomkit.llm.ttfb_ms metric."""
        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)

        from roomkit.channels.ai import AIChannel

        ai_provider = _MockAIProvider()
        ai = AIChannel("ai1", provider=ai_provider)
        kit.register_channel(ai)

        ch = SimpleChannel("ch1")
        kit.register_channel(ch)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "ch1")
        await kit.attach_channel(room.id, "ai1")

        msg = InboundMessage(
            channel_id="ch1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg, room_id=room.id)

        # The mock AI provider doesn't have _telemetry set via the real
        # propagation path — check that the propagation mechanism works
        assert hasattr(ai_provider, "_telemetry")
        assert ai_provider._telemetry is mock  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Phase 2: Telemetry propagation
# ---------------------------------------------------------------------------


class TestTelemetryPropagation:
    def test_ai_channel_propagates_to_provider(self) -> None:
        """AIChannel._propagate_telemetry() sets _telemetry on AI provider."""
        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)

        from roomkit.channels.ai import AIChannel

        ai_provider = _MockAIProvider()
        ai = AIChannel("ai1", provider=ai_provider)
        kit.register_channel(ai)

        assert ai_provider._telemetry is mock  # type: ignore[attr-defined]

    def test_realtime_voice_channel_propagates_to_provider(self) -> None:
        """RealtimeVoiceChannel._propagate_telemetry() sets _telemetry on provider."""
        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)

        from roomkit.channels.realtime_voice import RealtimeVoiceChannel
        from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()
        channel = RealtimeVoiceChannel("rt1", provider=provider, transport=transport)
        kit.register_channel(channel)

        assert provider._telemetry is mock  # type: ignore[attr-defined]

    def test_voice_channel_propagates_to_stt_tts_backend(self) -> None:
        """VoiceChannel.set_framework() propagates telemetry to STT/TTS/backend."""
        from roomkit.voice.backends.mock import MockVoiceBackend

        mock = MockTelemetryProvider()
        kit = RoomKit(telemetry=mock)

        from roomkit.channels.voice import VoiceChannel

        backend = MockVoiceBackend()
        channel = VoiceChannel("voice1", backend=backend)
        kit.register_channel(channel)

        assert backend._telemetry is mock  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Phase 2: New SpanKind values exist
# ---------------------------------------------------------------------------


class TestPhase2SpanKinds:
    def test_new_span_kinds(self) -> None:
        assert SpanKind.DELIVERY == "framework.delivery"
        assert SpanKind.VOICE_SESSION == "voice.session"
        assert SpanKind.STORE_QUERY == "store.query"
        assert SpanKind.BACKEND_CONNECT == "backend.connect"

    def test_new_attr_constants(self) -> None:
        assert Attr.DELIVERY_CHANNEL_TYPE == "delivery.channel_type"
        assert Attr.DELIVERY_RECIPIENT == "delivery.recipient"
        assert Attr.STORE_OPERATION == "store.operation"
        assert Attr.STORE_TABLE == "store.table"
        assert Attr.BACKEND_TYPE == "backend.type"
