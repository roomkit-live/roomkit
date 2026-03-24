"""Unit tests for ElevenLabsRealtimeProvider."""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState


@pytest.fixture
def config() -> ElevenLabsRealtimeConfig:
    return ElevenLabsRealtimeConfig(api_key="xi-test-key", agent_id="agent-123")


@pytest.fixture
def provider(config: ElevenLabsRealtimeConfig) -> ElevenLabsRealtimeProvider:
    return ElevenLabsRealtimeProvider(config)


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


def _inject_ws(provider: ElevenLabsRealtimeProvider, session: VoiceSession, ws: AsyncMock) -> None:
    """Wire a mock WebSocket into provider internals without calling connect()."""
    provider._connections[session.id] = ws
    provider._sessions[session.id] = session
    provider._last_interrupt_id[session.id] = 0
    session.state = VoiceSessionState.ACTIVE
    provider._receive_tasks[session.id] = asyncio.ensure_future(asyncio.sleep(999))


class TestConfig:
    def test_defaults(self) -> None:
        cfg = ElevenLabsRealtimeConfig(api_key="key", agent_id="agent-1")
        assert cfg.base_url == "wss://api.elevenlabs.io"
        assert cfg.requires_auth is False

    def test_custom_values(self) -> None:
        cfg = ElevenLabsRealtimeConfig(
            api_key="key",
            agent_id="agent-1",
            requires_auth=True,
            base_url="wss://api.eu.residency.elevenlabs.io",
        )
        assert cfg.requires_auth is True
        assert "eu.residency" in cfg.base_url


class TestProviderBasics:
    def test_name(self, provider: ElevenLabsRealtimeProvider) -> None:
        assert provider.name == "ElevenLabsRealtimeProvider"

    def test_is_responding_default(self, provider: ElevenLabsRealtimeProvider) -> None:
        assert provider.is_responding("nonexistent") is False


class TestCallbackRegistration:
    def test_on_audio(self, provider: ElevenLabsRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_audio(cb)
        assert cb in provider._audio_callbacks

    def test_on_transcription(self, provider: ElevenLabsRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_transcription(cb)
        assert cb in provider._transcription_callbacks

    def test_on_speech_start(self, provider: ElevenLabsRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_speech_start(cb)
        assert cb in provider._speech_start_callbacks

    def test_on_speech_end(self, provider: ElevenLabsRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_speech_end(cb)
        assert cb in provider._speech_end_callbacks

    def test_on_tool_call(self, provider: ElevenLabsRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_tool_call(cb)
        assert cb in provider._tool_call_callbacks

    def test_on_response_start(self, provider: ElevenLabsRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_response_start(cb)
        assert cb in provider._response_start_callbacks

    def test_on_response_end(self, provider: ElevenLabsRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_response_end(cb)
        assert cb in provider._response_end_callbacks

    def test_on_error(self, provider: ElevenLabsRealtimeProvider) -> None:
        cb = MagicMock()
        provider.on_error(cb)
        assert cb in provider._error_callbacks


class TestConnect:
    async def test_connect_sends_init_data(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        # Simulate server sending conversation_initiation_metadata
        ws.recv = AsyncMock(
            return_value=json.dumps(
                {
                    "type": "conversation_initiation_metadata",
                    "conversation_initiation_metadata_event": {
                        "conversation_id": "conv-abc",
                        "user_input_audio_format": "pcm_16000",
                        "agent_output_audio_format": "pcm_16000",
                    },
                }
            )
        )
        mock_ws_connect = AsyncMock(return_value=ws)

        with patch("websockets.connect", mock_ws_connect):
            await provider.connect(
                session,
                system_prompt="Be helpful",
                voice="voice-id-123",
                temperature=0.7,
            )

        # Verify WebSocket URL
        url = mock_ws_connect.call_args[0][0]
        assert "agent-123" in url
        assert "convai/conversation" in url

        # Verify init data
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "conversation_initiation_client_data"
        assert sent["conversation_config_override"]["agent"]["prompt"]["prompt"] == "Be helpful"
        assert sent["conversation_config_override"]["tts"]["voice_id"] == "voice-id-123"
        assert sent["custom_llm_extra_body"]["temperature"] == 0.7

        assert session.state == VoiceSessionState.ACTIVE
        assert provider._conversation_ids[session.id] == "conv-abc"

    async def test_connect_without_auth_sends_header(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        ws.recv = AsyncMock(
            return_value=json.dumps(
                {
                    "type": "conversation_initiation_metadata",
                    "conversation_initiation_metadata_event": {
                        "conversation_id": "conv-1",
                    },
                }
            )
        )
        mock_ws_connect = AsyncMock(return_value=ws)

        with patch("websockets.connect", mock_ws_connect):
            await provider.connect(session)

        headers = mock_ws_connect.call_args[1]["additional_headers"]
        assert headers["xi-api-key"] == "xi-test-key"

    async def test_connect_with_language_override(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        ws.recv = AsyncMock(
            return_value=json.dumps(
                {
                    "type": "conversation_initiation_metadata",
                    "conversation_initiation_metadata_event": {
                        "conversation_id": "conv-1",
                    },
                }
            )
        )
        mock_ws_connect = AsyncMock(return_value=ws)

        with patch("websockets.connect", mock_ws_connect):
            await provider.connect(
                session, provider_config={"language": "fr", "first_message": "Bonjour!"}
            )

        sent = json.loads(ws.send.call_args[0][0])
        agent = sent["conversation_config_override"]["agent"]
        assert agent["language"] == "fr"
        assert agent["first_message"] == "Bonjour!"

    async def test_connect_with_tools_logs_warning(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ws = _mock_ws()
        ws.recv = AsyncMock(
            return_value=json.dumps(
                {
                    "type": "conversation_initiation_metadata",
                    "conversation_initiation_metadata_event": {
                        "conversation_id": "conv-1",
                    },
                }
            )
        )
        mock_ws_connect = AsyncMock(return_value=ws)

        with patch("websockets.connect", mock_ws_connect):
            await provider.connect(
                session, tools=[{"name": "get_weather", "description": "Get weather"}]
            )

        assert any(
            "tools" in r.message.lower() and "dashboard" in r.message.lower()
            for r in caplog.records
        )


class TestSendAudio:
    async def test_send_audio(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        audio = b"\x00\x01\x02\x03"
        await provider.send_audio(session, audio)

        sent = json.loads(ws.send.call_args[0][0])
        # ElevenLabs uses "user_audio_chunk" with no "type" field
        assert "type" not in sent
        assert base64.b64decode(sent["user_audio_chunk"]) == audio

    async def test_send_audio_no_connection(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        # Should not raise
        await provider.send_audio(session, b"\x00")


class TestInjectText:
    async def test_inject_text_normal(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        await provider.inject_text(session, "Hello from text")

        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "user_message"
        assert sent["user_message"]["text"] == "Hello from text"

    async def test_inject_text_silent(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        await provider.inject_text(session, "Context only", silent=True)

        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "contextual_update"
        assert sent["text"] == "Context only"


class TestSubmitToolResult:
    async def test_submit_tool_result(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        await provider.submit_tool_result(session, "tc-123", '{"temp": 72}')

        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "client_tool_result"
        assert sent["tool_call_id"] == "tc-123"
        assert sent["result"] == '{"temp": 72}'
        assert sent["is_error"] is False


class TestInterrupt:
    async def test_interrupt_sends_user_activity(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        await provider.interrupt(session)

        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "user_activity"


class TestDisconnect:
    async def test_disconnect(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        await provider.disconnect(session)

        assert session.state == VoiceSessionState.ENDED
        ws.close.assert_awaited_once()
        assert session.id not in provider._connections
        assert session.id not in provider._sessions

    async def test_close_disconnects_all(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        await provider.close()

        assert session.state == VoiceSessionState.ENDED


class TestServerEvents:
    """Test _handle_server_event dispatches correctly."""

    async def test_audio_event(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        provider._last_interrupt_id[session.id] = 0
        audio_cb = AsyncMock()
        start_cb = AsyncMock()
        provider.on_audio(audio_cb)
        provider.on_response_start(start_cb)

        audio = b"\x00\x01\x02"
        await provider._handle_server_event(
            session,
            {
                "type": "audio",
                "audio_event": {
                    "event_id": 5,
                    "audio_base_64": base64.b64encode(audio).decode(),
                },
            },
        )
        audio_cb.assert_awaited_once_with(session, audio)
        # First audio chunk triggers response_start
        start_cb.assert_awaited_once_with(session)
        assert session.id in provider._responding

    async def test_audio_event_response_start_fires_only_once(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        """response_start should fire on the first audio chunk only."""
        provider._last_interrupt_id[session.id] = 0
        start_cb = AsyncMock()
        provider.on_response_start(start_cb)

        audio_event = {
            "type": "audio",
            "audio_event": {
                "event_id": 1,
                "audio_base_64": base64.b64encode(b"\x00").decode(),
            },
        }
        await provider._handle_server_event(session, audio_event)
        audio_event["audio_event"]["event_id"] = 2
        await provider._handle_server_event(session, audio_event)

        # Only fired once despite two audio chunks
        assert start_cb.await_count == 1

    async def test_audio_event_filtered_by_interrupt_id(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        provider._last_interrupt_id[session.id] = 10
        cb = AsyncMock()
        provider.on_audio(cb)

        await provider._handle_server_event(
            session,
            {
                "type": "audio",
                "audio_event": {
                    "event_id": 5,
                    "audio_base_64": base64.b64encode(b"\x00").decode(),
                },
            },
        )
        # Stale audio — should not fire callback
        cb.assert_not_awaited()

    async def test_user_transcript(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        cb = AsyncMock()
        provider.on_transcription(cb)

        await provider._handle_server_event(
            session,
            {
                "type": "user_transcript",
                "user_transcription_event": {"user_transcript": "Hello world"},
            },
        )
        cb.assert_awaited_once_with(session, "Hello world", "user", True)

    async def test_tentative_user_transcript(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        cb = AsyncMock()
        provider.on_transcription(cb)

        await provider._handle_server_event(
            session,
            {
                "type": "tentative_user_transcript",
                "tentative_user_transcript": {"tentative_user_transcript": "Hell"},
            },
        )
        cb.assert_awaited_once_with(session, "Hell", "user", False)

    async def test_agent_response(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        tx_cb = AsyncMock()
        end_cb = AsyncMock()
        provider.on_transcription(tx_cb)
        provider.on_response_end(end_cb)
        provider._responding.add(session.id)

        await provider._handle_server_event(
            session,
            {
                "type": "agent_response",
                "agent_response_event": {"agent_response": "I can help with that."},
            },
        )
        tx_cb.assert_awaited_once_with(session, "I can help with that.", "assistant", True)
        end_cb.assert_awaited_once_with(session)
        assert session.id not in provider._responding

    async def test_agent_response_correction(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        cb = AsyncMock()
        provider.on_transcription(cb)

        await provider._handle_server_event(
            session,
            {
                "type": "agent_response_correction",
                "agent_response_correction_event": {
                    "original_agent_response": "Original text",
                    "corrected_agent_response": "Corrected text",
                },
            },
        )
        cb.assert_awaited_once_with(session, "Corrected text", "assistant", True)

    async def test_client_tool_call(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        cb = AsyncMock()
        provider.on_tool_call(cb)

        await provider._handle_server_event(
            session,
            {
                "type": "client_tool_call",
                "client_tool_call": {
                    "tool_name": "get_weather",
                    "tool_call_id": "tc-42",
                    "parameters": {"city": "Paris"},
                },
            },
        )
        cb.assert_awaited_once_with(session, "tc-42", "get_weather", {"city": "Paris"})

    async def test_interruption_updates_last_interrupt_id(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        provider._last_interrupt_id[session.id] = 0
        provider._responding.add(session.id)
        cb = AsyncMock()
        provider.on_speech_start(cb)

        await provider._handle_server_event(
            session,
            {"type": "interruption", "interruption_event": {"event_id": 15}},
        )
        assert provider._last_interrupt_id[session.id] == 15
        assert session.id not in provider._responding
        cb.assert_awaited_once_with(session)

    async def test_ping_pong(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        await provider._handle_server_event(
            session,
            {"type": "ping", "ping_event": {"event_id": 7, "ping_ms": 120}},
        )

        sent = json.loads(ws.send.call_args[0][0])
        assert sent == {"type": "pong", "event_id": 7}

    async def test_error_event(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        cb = AsyncMock()
        provider.on_error(cb)

        await provider._handle_server_event(
            session,
            {
                "type": "error",
                "error": {"code": "invalid_message", "message": "Bad format"},
            },
        )
        cb.assert_awaited_once_with(session, "invalid_message", "Bad format")


class TestSendEvent:
    async def test_send_raw_event(
        self,
        provider: ElevenLabsRealtimeProvider,
        session: VoiceSession,
    ) -> None:
        ws = _mock_ws()
        _inject_ws(provider, session, ws)

        raw = {"type": "user_activity"}
        await provider.send_event(session, raw)

        sent = json.loads(ws.send.call_args[0][0])
        assert sent == raw


class TestBuildInitData:
    def test_minimal(self, provider: ElevenLabsRealtimeProvider) -> None:
        data = provider._build_init_data(
            system_prompt=None,
            voice=None,
            tools=None,
            temperature=None,
            provider_config={},
        )
        assert data["type"] == "conversation_initiation_client_data"
        assert "agent" not in data["conversation_config_override"]
        assert "tts" not in data["conversation_config_override"]
        assert data["custom_llm_extra_body"] == {}

    def test_full(self, provider: ElevenLabsRealtimeProvider) -> None:
        data = provider._build_init_data(
            system_prompt="You are a bot",
            voice="voice-abc",
            tools=[{"name": "search"}],
            temperature=0.9,
            provider_config={
                "language": "ja",
                "first_message": "Konnichiwa",
                "dynamic_variables": {"name": "Test"},
            },
        )
        agent = data["conversation_config_override"]["agent"]
        assert agent["prompt"]["prompt"] == "You are a bot"
        assert agent["language"] == "ja"
        assert agent["first_message"] == "Konnichiwa"
        assert data["conversation_config_override"]["tts"]["voice_id"] == "voice-abc"
        assert data["custom_llm_extra_body"]["temperature"] == 0.9
        assert data["dynamic_variables"] == {"name": "Test"}
