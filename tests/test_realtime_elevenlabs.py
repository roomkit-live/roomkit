"""Unit tests for ElevenLabsRealtimeProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import (
    ElevenLabsRealtimeProvider,
    _AsyncBridgeAudioInterface,
)
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
        cb = AsyncMock()
        provider._input_callbacks[session.id] = cb

        await provider.send_audio(session, b"\x00\x01\x02")

        cb.assert_awaited_once_with(b"\x00\x01\x02")

    async def test_send_audio_no_callback(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        # Should not raise
        await provider.send_audio(session, b"\x00")


class TestDisconnect:
    async def test_disconnect(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        mock_conversation = AsyncMock()

        provider._sessions[session.id] = session
        provider._conversations[session.id] = mock_conversation
        session.state = VoiceSessionState.ACTIVE

        await provider.disconnect(session)

        assert session.state == VoiceSessionState.ENDED
        mock_conversation.end_session.assert_awaited_once()
        mock_conversation.wait_for_session_end.assert_awaited_once()
        assert session.id not in provider._sessions
        assert session.id not in provider._conversations

    async def test_close_disconnects_all(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        mock_conversation = AsyncMock()

        provider._sessions[session.id] = session
        provider._conversations[session.id] = mock_conversation
        session.state = VoiceSessionState.ACTIVE

        await provider.close()

        assert session.state == VoiceSessionState.ENDED


class TestAsyncBridgeAudioInterface:
    """Test the _AsyncBridgeAudioInterface that connects SDK to RoomKit."""

    async def test_start_stores_input_callback(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        bridge = _AsyncBridgeAudioInterface(provider, session)

        mock_cb = AsyncMock()
        await bridge.start(mock_cb)

        assert provider._input_callbacks[session.id] is mock_cb

    async def test_stop_removes_input_callback(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        bridge = _AsyncBridgeAudioInterface(provider, session)

        provider._input_callbacks[session.id] = AsyncMock()
        await bridge.stop()

        assert session.id not in provider._input_callbacks

    async def test_output_fires_audio_callbacks(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        bridge = _AsyncBridgeAudioInterface(provider, session)

        audio_cb = AsyncMock()
        provider.on_audio(audio_cb)

        await bridge.output(b"\x00\x01\x02")

        audio_cb.assert_awaited_once_with(session, b"\x00\x01\x02")

    async def test_output_fires_response_start_once(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        bridge = _AsyncBridgeAudioInterface(provider, session)

        start_cb = AsyncMock()
        provider.on_response_start(start_cb)

        await bridge.output(b"\x00")
        await bridge.output(b"\x01")

        # response_start should fire only once
        assert start_cb.await_count == 1
        assert session.id in provider._responding

    async def test_interrupt_clears_responding(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        bridge = _AsyncBridgeAudioInterface(provider, session)
        provider._responding.add(session.id)

        await bridge.interrupt()

        assert session.id not in provider._responding


class TestSDKCallbackHandlers:
    """Test the async callback factories."""

    async def test_agent_response_callback(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        provider._responding.add(session.id)

        tx_cb = AsyncMock()
        end_cb = AsyncMock()
        provider.on_transcription(tx_cb)
        provider.on_response_end(end_cb)

        cb = provider._make_agent_response_cb(session)
        await cb("Hello there!")

        tx_cb.assert_awaited_once_with(session, "Hello there!", "assistant", True)
        end_cb.assert_awaited_once_with(session)
        assert session.id not in provider._responding

    async def test_correction_callback(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        tx_cb = AsyncMock()
        provider.on_transcription(tx_cb)

        cb = provider._make_correction_cb(session)
        await cb("Original", "Corrected")

        tx_cb.assert_awaited_once_with(session, "Corrected", "assistant", True)

    async def test_user_transcript_callback(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        tx_cb = AsyncMock()
        provider.on_transcription(tx_cb)

        end_cb = AsyncMock()
        provider.on_speech_end(end_cb)

        cb = provider._make_user_transcript_cb(session)
        await cb("Hello world")

        tx_cb.assert_awaited_once_with(session, "Hello world", "user", True)
        # User transcript arrival also fires speech_end
        end_cb.assert_awaited_once_with(session)
