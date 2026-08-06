"""Unit tests for ElevenLabsRealtimeProvider."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import (
    ElevenLabsRealtimeProvider,
    _AsyncBridgeAudioInterface,
)
from roomkit.voice.base import VoiceSession, VoiceSessionState


class _FakeClientTools:
    """Stand-in for the SDK registry — records what the provider registers."""

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}
        self.is_async: dict[str, bool] = {}

    def register(self, tool_name: str, handler: Any, is_async: bool = False) -> None:
        if tool_name in self.handlers:
            raise ValueError(f"Tool '{tool_name}' is already registered")
        self.handlers[tool_name] = handler
        self.is_async[tool_name] = is_async


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
        """Agent text opens the turn; it must not close the response.

        ConvAI sends the text before the synthesis, so ending the response
        here would fire ``response_end`` ahead of the first audio chunk.
        """
        provider._responding.add(session.id)

        tx_cb = AsyncMock()
        end_cb = AsyncMock()
        provider.on_transcription(tx_cb)
        provider.on_response_end(end_cb)

        cb = provider._make_agent_response_cb(session)
        await cb("Hello there!")

        tx_cb.assert_awaited_once_with(session, "Hello there!", "assistant", True)
        end_cb.assert_not_awaited()
        assert session.id in provider._responding

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


class TestReconfigure:
    def test_mid_session_reconfigure_is_refused(
        self, provider: ElevenLabsRealtimeProvider
    ) -> None:
        """The base reconfigure reconnects, which on ConvAI is a new conversation."""
        assert provider.supports_mid_session_reconfigure is False


class TestClientToolBridge:
    """Tools reach the agent through the SDK's ClientTools registry."""

    def test_registers_one_handler_per_declared_tool(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        registry = _FakeClientTools()

        provider._register_client_tools(
            registry,
            session,
            [
                {"name": "get_weather", "description": "Weather", "parameters": {}},
                {"name": "get_weather", "description": "Duplicate"},
                {"description": "nameless"},
            ],
        )

        # The SDK refuses a name twice, and a definition without a name has
        # nothing to register under.
        assert list(registry.handlers) == ["get_weather"]
        assert registry.is_async["get_weather"] is True

    def test_no_tools_registers_nothing(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        registry = _FakeClientTools()
        provider._register_client_tools(registry, session, None)
        assert registry.handlers == {}

    async def test_call_reaches_on_tool_call_and_returns_the_submitted_result(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        seen: dict[str, Any] = {}
        tasks: list[asyncio.Task[None]] = []

        def on_tool_call(
            s: VoiceSession, call_id: str, name: str, arguments: dict[str, Any]
        ) -> None:
            seen.update(session=s, call_id=call_id, name=name, arguments=arguments)
            tasks.append(
                asyncio.create_task(provider.submit_tool_result(s, call_id, '{"temp": 22}'))
            )

        provider.on_tool_call(on_tool_call)
        handler = provider._make_tool_handler(session, "get_weather")

        result = await handler({"tool_call_id": "call-1", "city": "Montreal"})

        assert result == '{"temp": 22}'
        assert seen["session"] is session
        assert seen["call_id"] == "call-1"
        assert seen["name"] == "get_weather"
        # tool_call_id is SDK plumbing, not a declared parameter of the tool.
        assert seen["arguments"] == {"city": "Montreal"}
        assert provider._pending_tools[session.id] == {}
        await asyncio.gather(*tasks)

    async def test_call_without_an_id_still_correlates(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        tasks: list[asyncio.Task[None]] = []

        def on_tool_call(
            s: VoiceSession, call_id: str, name: str, arguments: dict[str, Any]
        ) -> None:
            assert call_id
            tasks.append(asyncio.create_task(provider.submit_tool_result(s, call_id, "ok")))

        provider.on_tool_call(on_tool_call)
        handler = provider._make_tool_handler(session, "ping")

        assert await handler({}) == "ok"
        await asyncio.gather(*tasks)

    async def test_a_call_nobody_answers_times_out_as_an_error(
        self, session: VoiceSession
    ) -> None:
        provider = ElevenLabsRealtimeProvider(
            ElevenLabsRealtimeConfig(api_key="k", agent_id="a", tool_timeout_s=0.01)
        )
        provider.on_tool_call(lambda *_: None)
        handler = provider._make_tool_handler(session, "slow")

        # Raising is what the SDK turns into is_error=True; a returned string
        # would reach the agent as a successful call.
        with pytest.raises(RuntimeError, match="did not return within"):
            await handler({"tool_call_id": "call-1"})

        assert provider._pending_tools[session.id] == {}

    async def test_result_for_an_unknown_call_is_dropped(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        # A late result from a call that already timed out must not raise.
        await provider.submit_tool_result(session, "gone", "late")

    async def test_disconnect_fails_the_calls_still_in_flight(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        provider._sessions[session.id] = session
        provider._pending_tools[session.id] = {"call-1": future}

        await provider.disconnect(session)

        assert future.done()
        with pytest.raises(RuntimeError, match="abandoned"):
            future.result()


class TestResponseLifecycle:
    """ConvAI sends no end-of-audio marker, so silence closes the turn."""

    async def test_response_ends_once_the_audio_goes_quiet(self, session: VoiceSession) -> None:
        provider = ElevenLabsRealtimeProvider(
            ElevenLabsRealtimeConfig(api_key="k", agent_id="a", response_idle_ms=20)
        )
        end_cb = AsyncMock()
        provider.on_response_end(end_cb)
        bridge = _AsyncBridgeAudioInterface(provider, session)

        await bridge.output(b"\x00")
        await asyncio.sleep(0.15)

        end_cb.assert_awaited_once_with(session)
        assert session.id not in provider._responding

    async def test_a_pending_tool_call_holds_the_turn_open(self, session: VoiceSession) -> None:
        provider = ElevenLabsRealtimeProvider(
            ElevenLabsRealtimeConfig(api_key="k", agent_id="a", response_idle_ms=20)
        )
        end_cb = AsyncMock()
        provider.on_response_end(end_cb)
        bridge = _AsyncBridgeAudioInterface(provider, session)

        await bridge.output(b"\x00")
        provider._pending_tools[session.id] = {
            "call-1": asyncio.get_running_loop().create_future()
        }
        await asyncio.sleep(0.1)

        # The agent resumes speaking on this same turn once it has the result.
        end_cb.assert_not_awaited()

        provider._pending_tools[session.id].clear()
        await asyncio.sleep(0.1)

        end_cb.assert_awaited_once_with(session)

    async def test_interrupted_turn_still_ends(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        end_cb = AsyncMock()
        provider.on_response_end(end_cb)
        bridge = _AsyncBridgeAudioInterface(provider, session)

        await bridge.output(b"\x00")
        await bridge.interrupt()

        end_cb.assert_awaited_once_with(session)
        assert session.id not in provider._responding


class TestSessionSupervision:
    """A session that dies on its own must not look healthy."""

    async def test_connect_failure_inside_the_sdk_task_surfaces(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        err_cb = AsyncMock()
        provider.on_error(err_cb)
        provider._sessions[session.id] = session
        session.state = VoiceSessionState.ACTIVE

        conversation = AsyncMock()
        conversation.wait_for_session_end.side_effect = RuntimeError("401 Unauthorized")

        await provider._supervise_session(session, conversation)

        err_cb.assert_awaited_once_with(session, "connection_failed", "401 Unauthorized")
        assert session.state == VoiceSessionState.ENDED
        assert session.id not in provider._sessions

    async def test_service_closed_session_surfaces(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        err_cb = AsyncMock()
        provider.on_error(err_cb)
        provider._sessions[session.id] = session

        await provider._make_end_session_cb(session)()

        err_cb.assert_awaited_once()
        assert err_cb.await_args is not None
        assert err_cb.await_args.args[1] == "session_ended"

    async def test_our_own_disconnect_is_not_an_error(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        err_cb = AsyncMock()
        provider.on_error(err_cb)
        provider._sessions[session.id] = session
        provider._closing.add(session.id)

        await provider._make_end_session_cb(session)()

        err_cb.assert_not_awaited()

    async def test_failure_is_reported_once(
        self, provider: ElevenLabsRealtimeProvider, session: VoiceSession
    ) -> None:
        err_cb = AsyncMock()
        provider.on_error(err_cb)
        provider._sessions[session.id] = session

        await provider._fail_session(session, "session_ended", "gone")
        await provider._fail_session(session, "connection_failed", "gone again")

        assert err_cb.await_count == 1
