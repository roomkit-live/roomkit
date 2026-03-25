"""Unit tests for ElevenLabsRealtimeProvider."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider, _BridgeAudioInterface
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


class TestSendAudio:
    async def test_send_audio_calls_input_callback(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        cb = MagicMock()
        provider._input_callbacks[session.id] = cb

        await provider.send_audio(session, b"\x00\x01\x02")

        cb.assert_called_once_with(b"\x00\x01\x02")

    async def test_send_audio_no_callback(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        # Should not raise
        await provider.send_audio(session, b"\x00")


class TestDisconnect:
    async def test_disconnect(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        mock_conversation = MagicMock()
        mock_conversation.end_session = MagicMock()
        mock_conversation.wait_for_session_end = MagicMock(return_value="conv-123")

        provider._sessions[session.id] = session
        provider._conversations[session.id] = mock_conversation
        provider._loops[session.id] = asyncio.get_running_loop()
        session.state = VoiceSessionState.ACTIVE

        await provider.disconnect(session)

        assert session.state == VoiceSessionState.ENDED
        mock_conversation.end_session.assert_called_once()
        assert session.id not in provider._sessions
        assert session.id not in provider._conversations

    async def test_close_disconnects_all(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        mock_conversation = MagicMock()
        mock_conversation.end_session = MagicMock()
        mock_conversation.wait_for_session_end = MagicMock(return_value=None)

        provider._sessions[session.id] = session
        provider._conversations[session.id] = mock_conversation
        provider._loops[session.id] = asyncio.get_running_loop()
        session.state = VoiceSessionState.ACTIVE

        await provider.close()

        assert session.state == VoiceSessionState.ENDED


class TestBridgeAudioInterface:
    """Test the _BridgeAudioInterface that connects SDK to RoomKit."""

    async def test_start_stores_input_callback(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        loop = asyncio.get_running_loop()
        bridge = _BridgeAudioInterface(provider, session, loop)

        mock_cb = MagicMock()
        bridge.start(mock_cb)

        assert provider._input_callbacks[session.id] is mock_cb

    async def test_stop_removes_input_callback(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        loop = asyncio.get_running_loop()
        bridge = _BridgeAudioInterface(provider, session, loop)

        provider._input_callbacks[session.id] = MagicMock()
        bridge.stop()

        assert session.id not in provider._input_callbacks

    async def test_output_fires_audio_callbacks(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        loop = asyncio.get_running_loop()
        bridge = _BridgeAudioInterface(provider, session, loop)

        audio_cb = AsyncMock()
        provider.on_audio(audio_cb)

        bridge.output(b"\x00\x01\x02")
        # Allow the coroutine scheduled via run_coroutine_threadsafe to execute
        await asyncio.sleep(0.05)

        audio_cb.assert_awaited_once_with(session, b"\x00\x01\x02")

    async def test_output_fires_response_start_once(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        loop = asyncio.get_running_loop()
        bridge = _BridgeAudioInterface(provider, session, loop)

        start_cb = AsyncMock()
        provider.on_response_start(start_cb)

        bridge.output(b"\x00")
        bridge.output(b"\x01")
        await asyncio.sleep(0.05)

        # response_start should fire only once
        assert start_cb.await_count == 1
        assert session.id in provider._responding

    async def test_interrupt_fires_speech_start(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        loop = asyncio.get_running_loop()
        bridge = _BridgeAudioInterface(provider, session, loop)
        provider._responding.add(session.id)

        speech_cb = AsyncMock()
        provider.on_speech_start(speech_cb)

        bridge.interrupt()
        await asyncio.sleep(0.05)

        speech_cb.assert_awaited_once_with(session)
        assert session.id not in provider._responding


class TestSDKCallbackHandlers:
    """Test the _on_* methods called by SDK from its thread."""

    async def test_on_agent_response(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        loop = asyncio.get_running_loop()
        provider._loops[session.id] = loop
        provider._responding.add(session.id)

        tx_cb = AsyncMock()
        end_cb = AsyncMock()
        provider.on_transcription(tx_cb)
        provider.on_response_end(end_cb)

        provider._on_agent_response(session, "Hello there!")
        await asyncio.sleep(0.05)

        tx_cb.assert_awaited_once_with(session, "Hello there!", "assistant", True)
        end_cb.assert_awaited_once_with(session)
        assert session.id not in provider._responding

    async def test_on_agent_response_correction(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        loop = asyncio.get_running_loop()
        provider._loops[session.id] = loop

        tx_cb = AsyncMock()
        provider.on_transcription(tx_cb)

        provider._on_agent_response_correction(session, "Original", "Corrected")
        await asyncio.sleep(0.05)

        tx_cb.assert_awaited_once_with(session, "Corrected", "assistant", True)

    async def test_on_user_transcript(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        loop = asyncio.get_running_loop()
        provider._loops[session.id] = loop

        tx_cb = AsyncMock()
        provider.on_transcription(tx_cb)

        provider._on_user_transcript(session, "Hello world")
        await asyncio.sleep(0.05)

        tx_cb.assert_awaited_once_with(session, "Hello world", "user", True)
