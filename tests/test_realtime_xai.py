"""Unit tests for XAIRealtimeProvider."""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from roomkit.providers.xai.config import XAIRealtimeConfig
from roomkit.providers.xai.realtime import XAI_VOICES, XAIRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState


@pytest.fixture
def config() -> XAIRealtimeConfig:
    return XAIRealtimeConfig(api_key=SecretStr("xai-test-key"))


@pytest.fixture
def provider(config: XAIRealtimeConfig) -> XAIRealtimeProvider:
    return XAIRealtimeProvider(config)


@pytest.fixture
def session() -> VoiceSession:
    return VoiceSession(
        id="test-session-1",
        room_id="room-1",
        participant_id="user-1",
        channel_id="voice-1",
        state=VoiceSessionState.CONNECTING,
    )


def _mock_ws() -> AsyncMock:
    """Create a mock WebSocket that yields no messages."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    ws.__aiter__ = MagicMock(return_value=iter([]))
    return ws


def _inject_ws(provider: XAIRealtimeProvider, session: VoiceSession, ws: AsyncMock) -> None:
    """Wire a mock WebSocket into provider internals without calling connect()."""
    provider._connections[session.id] = ws
    provider._sessions[session.id] = session
    session.state = VoiceSessionState.ACTIVE
    # Create a no-op receive task
    provider._receive_tasks[session.id] = asyncio.ensure_future(asyncio.sleep(999))


class TestConfig:
    def test_defaults(self) -> None:
        cfg = XAIRealtimeConfig(api_key=SecretStr("xai-key"))
        assert cfg.model == "grok-3-fast"
        assert cfg.base_url == "wss://api.x.ai/v1/realtime"
        assert cfg.voice == "eve"
        assert cfg.transcription_model == "grok-2-audio"

    def test_custom_values(self) -> None:
        cfg = XAIRealtimeConfig(
            api_key=SecretStr("xai-key"),
            model="grok-2-audio",
            voice="ara",
            transcription_model="grok-2-audio",
        )
        assert cfg.model == "grok-2-audio"
        assert cfg.voice == "ara"


class TestProviderBasics:
    def test_name(self, provider: XAIRealtimeProvider) -> None:
        assert provider.name == "XAIRealtimeProvider"

    def test_voices_constant(self) -> None:
        assert "eve" in XAI_VOICES
        assert "ara" in XAI_VOICES
        assert "rex" in XAI_VOICES
        assert "sal" in XAI_VOICES
        assert "leo" in XAI_VOICES

    def test_init_with_api_key(self) -> None:
        p = XAIRealtimeProvider(api_key="xai-key-direct")
        assert p.name == "XAIRealtimeProvider"

    def test_init_requires_key_or_config(self) -> None:
        with pytest.raises(ValueError, match="Either config or api_key"):
            XAIRealtimeProvider()


class TestCallbackRegistration:
    def test_on_audio(self, provider: XAIRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_audio(cb)
        assert cb in provider._audio_callbacks

    def test_on_transcription(self, provider: XAIRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_transcription(cb)
        assert cb in provider._transcription_callbacks

    def test_on_speech_start(self, provider: XAIRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_speech_start(cb)
        assert cb in provider._speech_start_callbacks

    def test_on_speech_end(self, provider: XAIRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_speech_end(cb)
        assert cb in provider._speech_end_callbacks

    def test_on_tool_call(self, provider: XAIRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_tool_call(cb)
        assert cb in provider._tool_call_callbacks

    def test_on_response_start(self, provider: XAIRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_response_start(cb)
        assert cb in provider._response_start_callbacks

    def test_on_response_end(self, provider: XAIRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_response_end(cb)
        assert cb in provider._response_end_callbacks

    def test_on_error(self, provider: XAIRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_error(cb)
        assert cb in provider._error_callbacks


class TestConnect:
    async def test_connect_sends_session_update(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        mock_ws_connect = AsyncMock(return_value=ws)

        with patch("websockets.connect", mock_ws_connect):
            await provider.connect(
                session,
                system_prompt="Be helpful",
                voice="ara",
                temperature=0.8,
            )

        assert ws.send.call_count >= 1
        sent = json.loads(ws.send.call_args_list[0][0][0])
        assert sent["type"] == "session.update"

        sc = sent["session"]
        assert sc["voice"] == "ara"
        assert sc["instructions"] == "Be helpful"
        assert sc["temperature"] == 0.8
        assert sc["turn_detection"]["type"] == "server_vad"
        assert sc["input_audio_transcription"]["model"] == "grok-2-audio"
        assert sc["modalities"] == ["text", "audio"]

        assert session.state == VoiceSessionState.ACTIVE

    async def test_connect_url_includes_model(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        mock_ws_connect = AsyncMock(return_value=ws)

        with patch("websockets.connect", mock_ws_connect):
            await provider.connect(session)

        url = mock_ws_connect.call_args[0][0]
        assert "model=grok-3-fast" in url
        assert url.startswith("wss://api.x.ai/v1/realtime")

    async def test_connect_with_tools(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        mock_ws_connect = AsyncMock(return_value=ws)

        tools = [
            {"type": "web_search"},
            {"type": "x_search"},
            {"name": "get_weather", "description": "Get weather"},
        ]
        with patch("websockets.connect", mock_ws_connect):
            await provider.connect(session, tools=tools)

        sent = json.loads(ws.send.call_args_list[0][0][0])
        sc = sent["session"]
        assert len(sc["tools"]) == 3
        assert sc["tools"][0]["type"] == "web_search"
        assert sc["tools"][1]["type"] == "x_search"
        assert sc["tools"][2]["type"] == "function"


class TestSendAudio:
    async def test_send_audio(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        audio = b"\x00\x01\x02\x03"
        await provider.send_audio(session, audio)

        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "input_audio_buffer.append"
        assert base64.b64decode(sent["audio"]) == audio

    async def test_send_audio_no_connection(
        self, provider: XAIRealtimeProvider, session: VoiceSession
    ) -> None:
        await provider.send_audio(session, b"\x00")


class TestInjectText:
    async def test_inject_text(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        await provider.inject_text(session, "Hello from text")

        calls = [json.loads(c[0][0]) for c in ws.send.call_args_list]
        assert calls[0]["type"] == "conversation.item.create"
        assert calls[0]["item"]["content"][0]["text"] == "Hello from text"
        assert calls[1]["type"] == "response.create"

    async def test_inject_text_silent(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        await provider.inject_text(session, "Context only", silent=True)

        calls = [json.loads(c[0][0]) for c in ws.send.call_args_list]
        assert len(calls) == 1
        assert calls[0]["type"] == "conversation.item.create"


class TestInterrupt:
    async def test_interrupt(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        await provider.interrupt(session)

        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "response.cancel"


class TestSubmitToolResult:
    async def test_submit_tool_result(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        await provider.submit_tool_result(session, "call-1", '{"temp": 72}')

        calls = [json.loads(c[0][0]) for c in ws.send.call_args_list]
        assert calls[0]["type"] == "conversation.item.create"
        assert calls[0]["item"]["type"] == "function_call_output"
        assert calls[0]["item"]["call_id"] == "call-1"
        assert calls[0]["item"]["output"] == '{"temp": 72}'
        assert calls[1]["type"] == "response.create"


class TestDisconnect:
    async def test_disconnect(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        await provider.disconnect(session)

        assert session.state == VoiceSessionState.ENDED
        ws.close.assert_awaited_once()

    async def test_close_disconnects_all(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        await provider.close()

        assert session.state == VoiceSessionState.ENDED


class TestServerEvents:
    """Test _handle_server_event dispatches correctly."""

    async def test_speech_started(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        cb = AsyncMock()
        provider.on_speech_start(cb)

        await provider._handle_server_event(session, {"type": "input_audio_buffer.speech_started"})
        cb.assert_awaited_once_with(session)

    async def test_speech_stopped(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        cb = AsyncMock()
        provider.on_speech_end(cb)

        await provider._handle_server_event(session, {"type": "input_audio_buffer.speech_stopped"})
        cb.assert_awaited_once_with(session)

    async def test_audio_delta(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        cb = AsyncMock()
        provider.on_audio(cb)

        audio = b"\x00\x01\x02"
        await provider._handle_server_event(
            session,
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(audio).decode(),
            },
        )
        cb.assert_awaited_once_with(session, audio)

    async def test_transcription_delta(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        cb = AsyncMock()
        provider.on_transcription(cb)

        await provider._handle_server_event(
            session,
            {"type": "response.output_audio_transcript.delta", "delta": "Hello"},
        )
        cb.assert_awaited_once_with(session, "Hello", "assistant", False)

    async def test_user_transcription_completed(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        cb = AsyncMock()
        provider.on_transcription(cb)

        await provider._handle_server_event(
            session,
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "Hi there",
            },
        )
        cb.assert_awaited_once_with(session, "Hi there", "user", True)

    async def test_tool_call(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        cb = AsyncMock()
        provider.on_tool_call(cb)

        await provider._handle_server_event(
            session,
            {
                "type": "response.function_call_arguments.done",
                "call_id": "call-42",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            },
        )
        cb.assert_awaited_once_with(session, "call-42", "get_weather", {"city": "Paris"})

    async def test_response_created(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        cb = AsyncMock()
        provider.on_response_start(cb)

        await provider._handle_server_event(session, {"type": "response.created"})
        cb.assert_awaited_once_with(session)
        assert session.id in provider._responding

    async def test_response_done_with_usage(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        cb = AsyncMock()
        provider.on_response_end(cb)
        provider._responding.add(session.id)

        await provider._handle_server_event(
            session,
            {
                "type": "response.done",
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                    },
                },
            },
        )
        cb.assert_awaited_once_with(session)
        assert session.id not in provider._responding
        assert session._last_usage["input_tokens"] == 100
        assert session._last_usage["output_tokens"] == 50

    async def test_response_done_failed(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        err_cb = AsyncMock()
        provider.on_error(err_cb)

        await provider._handle_server_event(
            session,
            {
                "type": "response.done",
                "response": {
                    "status": "failed",
                    "status_details": {
                        "error": {
                            "type": "server_error",
                            "code": "rate_limit",
                            "message": "Too many requests",
                        }
                    },
                },
            },
        )
        err_cb.assert_awaited_once_with(session, "rate_limit", "Too many requests")

    async def test_error_event(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        err_cb = AsyncMock()
        provider.on_error(err_cb)

        await provider._handle_server_event(
            session,
            {
                "type": "error",
                "error": {"code": "invalid_request", "message": "Bad input"},
            },
        )
        err_cb.assert_awaited_once_with(session, "invalid_request", "Bad input")


class TestSendEvent:
    async def test_send_raw_event(
        self,
        provider: XAIRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        raw = {"type": "input_audio_buffer.commit"}
        await provider.send_event(session, raw)

        sent = json.loads(ws.send.call_args[0][0])
        assert sent == raw


class TestLazyLoaders:
    def test_get_xai_realtime_provider(self) -> None:
        from roomkit.voice import get_xai_realtime_provider

        cls = get_xai_realtime_provider()
        assert cls is XAIRealtimeProvider

    def test_get_xai_realtime_config(self) -> None:
        from roomkit.voice import get_xai_realtime_config

        cls = get_xai_realtime_config()
        assert cls is XAIRealtimeConfig
