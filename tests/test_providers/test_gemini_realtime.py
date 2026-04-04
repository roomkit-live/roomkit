"""Tests for GeminiLiveProvider."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from roomkit.voice.base import VoiceSession, VoiceSessionState


def _make_session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(
        id=sid,
        room_id="room1",
        participant_id="p1",
        channel_id="ch1",
    )


def _build_fake_genai():
    """Build a fake google.genai module tree."""

    # SessionResumptionConfig and SlidingWindow need mutable attributes
    class FakeSessionResumption:
        def __init__(self, handle=None):
            self.handle = handle

    class FakeSlidingWindow:
        pass

    class FakeContextWindowCompression:
        def __init__(self, sliding_window=None):
            self.sliding_window = sliding_window

    types = SimpleNamespace(
        HttpOptions=lambda **kw: SimpleNamespace(**kw),
        AudioTranscriptionConfig=lambda **kw: SimpleNamespace(**kw),
        SpeechConfig=lambda **kw: SimpleNamespace(**kw),
        VoiceConfig=lambda **kw: SimpleNamespace(**kw),
        PrebuiltVoiceConfig=lambda **kw: SimpleNamespace(**kw),
        LiveConnectConfig=lambda **kw: SimpleNamespace(**kw),
        AutomaticActivityDetection=lambda **kw: SimpleNamespace(**kw),
        RealtimeInputConfig=lambda **kw: SimpleNamespace(**kw),
        ThinkingConfig=lambda **kw: SimpleNamespace(**kw),
        ProactivityConfig=lambda **kw: SimpleNamespace(**kw),
        Tool=lambda **kw: SimpleNamespace(**kw),
        FunctionDeclaration=lambda **kw: SimpleNamespace(**kw),
        SessionResumptionConfig=FakeSessionResumption,
        SlidingWindow=FakeSlidingWindow,
        ContextWindowCompressionConfig=FakeContextWindowCompression,
        Blob=lambda **kw: SimpleNamespace(**kw),
        Content=lambda **kw: SimpleNamespace(**kw),
        Part=lambda **kw: SimpleNamespace(**kw),
        FunctionResponse=lambda **kw: SimpleNamespace(**kw),
    )

    genai = SimpleNamespace(
        Client=lambda **kw: SimpleNamespace(
            aio=SimpleNamespace(live=SimpleNamespace(connect=lambda **k: None))
        ),
        types=types,
    )

    google = SimpleNamespace(genai=genai)

    return {
        "google": google,
        "google.genai": genai,
        "google.genai.types": types,
    }


def _load_provider():
    """Import the provider module with google.genai mocked."""
    mods = _build_fake_genai()
    with patch.dict(sys.modules, mods):
        import roomkit.providers.gemini.realtime as mod

        importlib.reload(mod)
        return mod


def _make_mock_live_session():
    """Create a mock Gemini live session."""
    ls = AsyncMock()
    ls.send_realtime_input = AsyncMock()
    ls.send_client_content = AsyncMock()
    ls.send_tool_response = AsyncMock()
    ls.close = AsyncMock()
    # receive() returns an async iterator
    ls.receive = MagicMock(return_value=_async_iter([]))
    return ls


async def _async_iter(items):
    """Create an async iterator from a list."""
    for item in items:
        yield item


class TestGeminiLiveProvider:
    def test_constructor_and_name(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        assert provider.name == "GeminiLiveProvider"

    def test_constructor_with_custom_model(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(
            api_key="test-key",
            model="gemini-2.0-flash-live",
        )
        assert provider._model == "gemini-2.0-flash-live"

    def test_callback_registration(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")

        audio_cb = lambda session, audio: None  # noqa: E731
        transcription_cb = lambda session, text, role, final: None  # noqa: E731
        speech_start_cb = lambda session: None  # noqa: E731
        speech_end_cb = lambda session: None  # noqa: E731
        tool_call_cb = lambda session, cid, name, args: None  # noqa: E731
        response_start_cb = lambda session: None  # noqa: E731
        response_end_cb = lambda session: None  # noqa: E731
        error_cb = lambda session, code, msg: None  # noqa: E731

        provider.on_audio(audio_cb)
        provider.on_transcription(transcription_cb)
        provider.on_speech_start(speech_start_cb)
        provider.on_speech_end(speech_end_cb)
        provider.on_tool_call(tool_call_cb)
        provider.on_response_start(response_start_cb)
        provider.on_response_end(response_end_cb)
        provider.on_error(error_cb)

        assert audio_cb in provider._audio_callbacks
        assert transcription_cb in provider._transcription_callbacks
        assert speech_start_cb in provider._speech_start_callbacks
        assert speech_end_cb in provider._speech_end_callbacks
        assert tool_call_cb in provider._tool_call_callbacks
        assert response_start_cb in provider._response_start_callbacks
        assert response_end_cb in provider._response_end_callbacks
        assert error_cb in provider._error_callbacks

    async def test_disconnect_unknown_session_is_noop(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session("unknown")
        # disconnect on unknown session should not raise
        await provider.disconnect(session)
        assert session.state == VoiceSessionState.ENDED

    async def test_close_empty_provider(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        await provider.close()

    # ── _build_config() ─────────────────────────────────────────

    def test_build_config_basic(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        config = provider._build_config()
        assert config is not None

    def test_build_config_with_system_prompt(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        config = provider._build_config(system_prompt="Be helpful")
        assert config.system_instruction == "Be helpful"

    def test_build_config_with_voice(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        config = provider._build_config(voice="Aoede")
        assert config.speech_config is not None

    def test_build_config_with_temperature(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        config = provider._build_config(temperature=0.5)
        assert config.temperature == 0.5

    def test_build_config_with_tools(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        tools = [{"name": "get_weather", "description": "Get weather", "parameters": None}]
        # Need to mock the schema cleaner
        with patch("roomkit.providers.gemini.schema.clean_gemini_schema", return_value=None):
            config = provider._build_config(tools=tools)
        assert config.tools is not None

    def test_build_config_with_provider_config_options(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")

        pc = {
            "response_modalities": ["TEXT"],
            "language": "en-US",
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 1000,
            "seed": 42,
            "enable_affective_dialog": True,
            "thinking_budget": 1024,
            "proactive_audio": True,
        }
        config = provider._build_config(
            voice="Puck",
            provider_config=pc,
        )
        assert config.response_modalities == ["TEXT"]
        assert config.temperature is None  # not set
        assert config.top_p == 0.9
        assert config.top_k == 40.0
        assert config.max_output_tokens == 1000
        assert config.seed == 42
        assert config.enable_affective_dialog is True
        assert config.thinking_config is not None
        assert config.proactivity is not None

    def test_build_config_with_vad_options(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")

        pc = {
            "start_of_speech_sensitivity": "LOW",
            "end_of_speech_sensitivity": "HIGH",
            "silence_duration_ms": 1000,
            "prefix_padding_ms": 200,
        }
        config = provider._build_config(provider_config=pc)
        assert config.realtime_input_config is not None

    def test_build_config_with_no_interruption(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")

        pc = {"no_interruption": True}
        config = provider._build_config(provider_config=pc)
        assert config.realtime_input_config is not None
        assert config.realtime_input_config.activity_handling == "NO_INTERRUPTION"

    def test_build_config_start_sensitivity_full_name(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")

        pc = {"start_of_speech_sensitivity": "START_SENSITIVITY_LOW"}
        config = provider._build_config(provider_config=pc)
        assert config.realtime_input_config is not None

    # ── connect() ───────────────────────────────────────────────

    async def test_connect_success(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        mock_ctxmgr = AsyncMock()
        mock_ctxmgr.__aenter__ = AsyncMock(return_value=mock_live_session)
        mock_ctxmgr.__aexit__ = AsyncMock(return_value=False)

        provider._client = MagicMock()
        provider._client.aio.live.connect = MagicMock(return_value=mock_ctxmgr)

        await provider.connect(
            session,
            system_prompt="Be helpful",
            voice="Aoede",
            input_sample_rate=16000,
        )

        assert session.state == VoiceSessionState.ACTIVE
        assert session.id in provider._sessions
        state = provider._sessions[session.id]
        assert state.live_session is mock_live_session
        assert state.input_sample_rate == 16000

        # Clean up
        state.receive_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await state.receive_task

    # ── send_audio() ────────────────────────────────────────────

    async def test_send_audio(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        session.state = VoiceSessionState.ACTIVE

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            input_sample_rate=16000,
        )
        provider._sessions[session.id] = state

        await provider.send_audio(session, b"\x00\x01\x02")

        mock_live_session.send_realtime_input.assert_awaited_once()
        assert state.error_suppressed is False

    async def test_send_audio_no_session(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        # No session registered — should return
        await provider.send_audio(session, b"\x00")

    async def test_send_audio_connecting_buffers(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        session.state = VoiceSessionState.CONNECTING

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state

        await provider.send_audio(session, b"\x00\x01")
        assert len(state.audio_buffer) == 1
        mock_live_session.send_realtime_input.assert_not_awaited()

    async def test_send_audio_ended_skips(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        session.state = VoiceSessionState.ENDED

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state

        await provider.send_audio(session, b"\x00")
        mock_live_session.send_realtime_input.assert_not_awaited()

    async def test_send_audio_error_transitions_to_connecting(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        session.state = VoiceSessionState.ACTIVE

        mock_live_session = _make_mock_live_session()
        mock_live_session.send_realtime_input.side_effect = ConnectionError("lost")

        errors = []
        provider.on_error(lambda s, code, msg: errors.append((code, msg)))

        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state

        await provider.send_audio(session, b"\x00")
        assert session.state == VoiceSessionState.CONNECTING
        assert state.error_suppressed is True
        assert len(errors) == 1

    async def test_send_audio_error_suppressed_on_second_call(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        session.state = VoiceSessionState.ACTIVE

        mock_live_session = _make_mock_live_session()
        mock_live_session.send_realtime_input.side_effect = ConnectionError("lost")

        errors = []
        provider.on_error(lambda s, code, msg: errors.append((code, msg)))

        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            error_suppressed=True,
        )
        provider._sessions[session.id] = state

        # Already in CONNECTING from previous failure
        session.state = VoiceSessionState.CONNECTING

        # send_audio should buffer, not try to send (state is CONNECTING)
        await provider.send_audio(session, b"\x00")
        # Error not fired again because already suppressed
        assert len(errors) == 0

    # ── inject_text() ──────────────────────────────────────────

    async def test_inject_text(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state

        await provider.inject_text(session, "Hello!")

        mock_live_session.send_client_content.assert_awaited_once()
        call_kwargs = mock_live_session.send_client_content.call_args[1]
        assert call_kwargs["turn_complete"] is True

    async def test_inject_text_silent(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state

        await provider.inject_text(session, "Context only", silent=True)

        call_kwargs = mock_live_session.send_client_content.call_args[1]
        assert call_kwargs["turn_complete"] is False

    async def test_inject_text_model_role(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state

        await provider.inject_text(session, "Model response", role="model")

        call_kwargs = mock_live_session.send_client_content.call_args[1]
        assert call_kwargs["turns"].role == "model"

    async def test_inject_text_invalid_role_defaults_to_user(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state

        await provider.inject_text(session, "Test", role="assistant")

        call_kwargs = mock_live_session.send_client_content.call_args[1]
        assert call_kwargs["turns"].role == "user"

    async def test_inject_text_no_session(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        # No session — should return silently
        await provider.inject_text(session, "No session")

    # ── submit_tool_result() ────────────────────────────────────

    async def test_submit_tool_result_json(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state

        await provider.submit_tool_result(session, "call-1", '{"temperature": 72}')

        mock_live_session.send_tool_response.assert_awaited_once()
        assert state.tool_result_bytes == len('{"temperature": 72}')

    async def test_submit_tool_result_plain_text(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state

        await provider.submit_tool_result(session, "call-2", "plain text result")
        mock_live_session.send_tool_response.assert_awaited_once()

    async def test_submit_tool_result_json_non_dict(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state

        # JSON that parses to a list, not a dict
        await provider.submit_tool_result(session, "call-3", "[1, 2, 3]")
        mock_live_session.send_tool_response.assert_awaited_once()

    async def test_submit_tool_result_large_warns(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state

        # Large result > 16384 chars
        large_result = "x" * 20000
        await provider.submit_tool_result(session, "call-4", large_result)
        assert state.tool_result_bytes == 20000

    async def test_submit_tool_result_no_session(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        await provider.submit_tool_result(session, "call-1", "result")

    # ── interrupt() ─────────────────────────────────────────────

    async def test_interrupt_is_noop(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state

        await provider.interrupt(session)
        # Gemini doesn't support direct cancel, just logs

    async def test_interrupt_no_session(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        await provider.interrupt(session)

    # ── disconnect() ────────────────────────────────────────────

    async def test_disconnect_with_ctxmgr(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        session.state = VoiceSessionState.ACTIVE

        mock_live_session = _make_mock_live_session()
        mock_ctxmgr = AsyncMock()
        mock_ctxmgr.__aexit__ = AsyncMock(return_value=False)

        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            ctxmgr=mock_ctxmgr,
            started_at=1000.0,
            turn_count=5,
            audio_chunk_count=100,
            tool_result_bytes=500,
        )
        provider._sessions[session.id] = state

        await provider.disconnect(session)

        assert session.state == VoiceSessionState.ENDED
        assert session.id not in provider._sessions
        mock_ctxmgr.__aexit__.assert_awaited_once()

    async def test_disconnect_without_ctxmgr_closes_session(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        session.state = VoiceSessionState.ACTIVE

        mock_live_session = _make_mock_live_session()

        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            ctxmgr=None,
        )
        provider._sessions[session.id] = state

        await provider.disconnect(session)

        assert session.state == VoiceSessionState.ENDED
        mock_live_session.close.assert_awaited_once()

    async def test_disconnect_clears_transcription_buffers(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state
        provider._transcription_buffers[(session.id, "user")] = ["chunk"]

        await provider.disconnect(session)

        assert (session.id, "user") not in provider._transcription_buffers

    async def test_disconnect_cancels_receive_task(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        async def dummy():
            await asyncio.sleep(100)

        task = asyncio.create_task(dummy())

        state = mod._GeminiSessionState(
            session=session,
            live_session=_make_mock_live_session(),
            receive_task=task,
        )
        provider._sessions[session.id] = state

        await provider.disconnect(session)
        assert task.cancelled() or task.done()

    # ── close() ─────────────────────────────────────────────────

    async def test_close_disconnects_all(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")

        session1 = _make_session("s1")
        session2 = _make_session("s2")

        for s in [session1, session2]:
            state = mod._GeminiSessionState(
                session=s,
                live_session=_make_mock_live_session(),
            )
            provider._sessions[s.id] = state

        await provider.close()
        assert len(provider._sessions) == 0

    # ── _make_audio_blob() ──────────────────────────────────────

    def test_make_audio_blob(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")

        blob = provider._make_audio_blob(b"\x00\x01", 16000)
        assert blob.mime_type == "audio/pcm;rate=16000"
        assert blob.data == b"\x00\x01"

    def test_make_audio_blob_caches_mime(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")

        provider._make_audio_blob(b"\x00", 24000)
        provider._make_audio_blob(b"\x01", 24000)
        # Second call should use cached mime
        assert 24000 in provider._mime_cache

    # ── _handle_server_response() ───────────────────────────────

    async def test_handle_audio_data(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        session.state = VoiceSessionState.ACTIVE

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        received_audio = []
        provider.on_audio(lambda s, audio: received_audio.append(audio))

        response = SimpleNamespace(data=b"\x00\x01\x02")
        await provider._handle_server_response(session, response)

        assert len(received_audio) == 1
        assert received_audio[0] == b"\x00\x01\x02"
        assert state.audio_chunk_count == 1

    async def test_handle_tool_call(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        tool_calls = []
        provider.on_tool_call(lambda s, cid, name, args: tool_calls.append((cid, name, args)))

        fc = SimpleNamespace(id="fc-1", name="get_weather", args={"city": "NYC"})
        response = SimpleNamespace(
            tool_call=SimpleNamespace(function_calls=[fc]),
        )
        await provider._handle_server_response(session, response)

        assert tool_calls == [("fc-1", "get_weather", {"city": "NYC"})]

    async def test_handle_tool_call_no_args(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        tool_calls = []
        provider.on_tool_call(lambda s, cid, name, args: tool_calls.append((cid, name, args)))

        fc = SimpleNamespace(id="fc-2", name="ping", args=None)
        response = SimpleNamespace(
            tool_call=SimpleNamespace(function_calls=[fc]),
        )
        await provider._handle_server_response(session, response)

        assert tool_calls == [("fc-2", "ping", {})]

    async def test_handle_voice_activity_start(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        starts = []
        provider.on_speech_start(lambda s: starts.append(s.id))

        response = SimpleNamespace(
            voice_activity=SimpleNamespace(voice_activity_type="ACTIVITY_START"),
        )
        await provider._handle_server_response(session, response)

        assert starts == [session.id]
        assert state.user_speech_active is True

    async def test_handle_voice_activity_end(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        ends = []
        provider.on_speech_end(lambda s: ends.append(s.id))

        state.user_speech_active = True  # Simulate prior ACTIVITY_START

        response = SimpleNamespace(
            voice_activity=SimpleNamespace(voice_activity_type="ACTIVITY_END"),
        )
        await provider._handle_server_response(session, response)

        assert ends == [session.id]
        assert state.user_speech_active is False

    async def test_handle_model_turn_starts_response(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        starts = []
        provider.on_response_start(lambda s: starts.append(s.id))

        response = SimpleNamespace(
            server_content=SimpleNamespace(
                model_turn=SimpleNamespace(parts=[]),
                turn_complete=False,
                interrupted=False,
                input_transcription=None,
                output_transcription=None,
            ),
        )
        await provider._handle_server_response(session, response)

        assert state.response_started is True
        assert starts == [session.id]

    async def test_handle_model_turn_no_duplicate_start(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session, response_started=True)
        provider._sessions[session.id] = state

        starts = []
        provider.on_response_start(lambda s: starts.append(s.id))

        response = SimpleNamespace(
            server_content=SimpleNamespace(
                model_turn=SimpleNamespace(parts=[]),
                turn_complete=False,
                interrupted=False,
                input_transcription=None,
                output_transcription=None,
            ),
        )
        await provider._handle_server_response(session, response)

        # Should NOT fire response_start again
        assert starts == []

    async def test_handle_turn_complete(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session, response_started=True)
        provider._sessions[session.id] = state

        ends = []
        provider.on_response_end(lambda s: ends.append(s.id))

        response = SimpleNamespace(
            server_content=SimpleNamespace(
                model_turn=None,
                turn_complete=True,
                interrupted=False,
                input_transcription=None,
                output_transcription=None,
            ),
        )
        await provider._handle_server_response(session, response)

        assert state.response_started is False
        assert state.turn_count == 1
        assert ends == [session.id]

    async def test_handle_interrupted(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session, response_started=True)
        provider._sessions[session.id] = state

        speech_starts = []
        response_ends = []
        provider.on_speech_start(lambda s: speech_starts.append(s.id))
        provider.on_response_end(lambda s: response_ends.append(s.id))

        response = SimpleNamespace(
            server_content=SimpleNamespace(
                model_turn=None,
                turn_complete=False,
                interrupted=True,
                input_transcription=None,
                output_transcription=None,
            ),
        )
        await provider._handle_server_response(session, response)

        assert state.response_started is False
        # speech_start fires from interrupted when ACTIVITY_START wasn't received
        # (user_speech_active was False) — this triggers transport.interrupt()
        assert speech_starts == [session.id]
        assert state.user_speech_active is True
        assert response_ends == [session.id]

    async def test_handle_interrupted_no_double_fire_after_activity_start(self):
        """speech_start should NOT fire from interrupted if ACTIVITY_START already did."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(
            session=session, response_started=True, user_speech_active=True
        )
        provider._sessions[session.id] = state

        speech_starts = []
        response_ends = []
        provider.on_speech_start(lambda s: speech_starts.append(s.id))
        provider.on_response_end(lambda s: response_ends.append(s.id))

        response = SimpleNamespace(
            server_content=SimpleNamespace(
                model_turn=None,
                turn_complete=False,
                interrupted=True,
                input_transcription=None,
                output_transcription=None,
            ),
        )
        await provider._handle_server_response(session, response)

        # speech_start should NOT fire — ACTIVITY_START already set user_speech_active
        assert speech_starts == []
        assert response_ends == [session.id]

    async def test_handle_interrupted_without_response_started(self):
        """Interrupted when response_started=False should not fire response_end."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session, response_started=False)
        provider._sessions[session.id] = state

        response_ends = []
        provider.on_response_end(lambda s: response_ends.append(s.id))

        response = SimpleNamespace(
            server_content=SimpleNamespace(
                model_turn=None,
                turn_complete=False,
                interrupted=True,
                input_transcription=None,
                output_transcription=None,
            ),
        )
        await provider._handle_server_response(session, response)

        # response_end should not fire if response_start was never fired
        assert response_ends == []

    async def test_handle_input_transcription(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        transcriptions = []
        provider.on_transcription(
            lambda s, text, role, final: transcriptions.append((text, role, final))
        )

        response = SimpleNamespace(
            server_content=SimpleNamespace(
                model_turn=None,
                turn_complete=False,
                interrupted=False,
                input_transcription=SimpleNamespace(text="Hello world", finished=True),
                output_transcription=None,
            ),
        )
        await provider._handle_server_response(session, response)

        assert len(transcriptions) == 1
        assert transcriptions[0] == ("Hello world", "user", True)

    async def test_handle_output_transcription(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        transcriptions = []
        provider.on_transcription(
            lambda s, text, role, final: transcriptions.append((text, role, final))
        )

        response = SimpleNamespace(
            server_content=SimpleNamespace(
                model_turn=None,
                turn_complete=False,
                interrupted=False,
                input_transcription=None,
                output_transcription=SimpleNamespace(text="Response", finished=False),
            ),
        )
        await provider._handle_server_response(session, response)

        # Non-final: partial transcription
        assert len(transcriptions) == 1
        assert transcriptions[0] == ("Response", "assistant", False)

    async def test_handle_session_resumption_update(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        response = SimpleNamespace(
            session_resumption_update=SimpleNamespace(
                resumable=True,
                new_handle="handle-123",
            ),
        )
        await provider._handle_server_response(session, response)

        assert state.resumption_handle == "handle-123"

    async def test_handle_usage_metadata(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                total_token_count=150,
            ),
        )
        await provider._handle_server_response(session, response)

    async def test_handle_go_away(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        response = SimpleNamespace(
            go_away=SimpleNamespace(time_left="30s"),
        )

        with __import__("pytest").raises(mod._GoAwayError):
            await provider._handle_server_response(session, response)

    async def test_handle_unknown_response(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        # Bare response with no known attributes
        response = SimpleNamespace()
        await provider._handle_server_response(session, response)

    async def test_handle_response_no_session_state(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        # No state registered — should return early
        response = SimpleNamespace(data=b"\x00")
        await provider._handle_server_response(session, response)

    # ── Transcription buffering ─────────────────────────────────

    async def test_transcription_chunk_accumulation(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        transcriptions = []
        provider.on_transcription(
            lambda s, text, role, final: transcriptions.append((text, role, final))
        )

        # Send multiple non-final chunks followed by a final
        await provider._handle_transcription_chunk(session, "Hello ", "user", False)
        await provider._handle_transcription_chunk(session, "world", "user", True)

        # Should get two callbacks: non-final partial + final full
        assert len(transcriptions) == 2
        assert transcriptions[0] == ("Hello ", "user", False)
        assert transcriptions[1] == ("Hello world", "user", True)

    async def test_flush_transcription_buffer(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        transcriptions = []
        provider.on_transcription(
            lambda s, text, role, final: transcriptions.append((text, role, final))
        )

        # Buffer some text
        provider._transcription_buffers[(session.id, "user")] = ["Hello ", "world"]

        await provider._flush_transcription_buffer(session, "user")

        assert len(transcriptions) == 1
        assert transcriptions[0] == ("Hello world", "user", True)
        assert (session.id, "user") not in provider._transcription_buffers

    async def test_flush_empty_buffer_is_noop(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        transcriptions = []
        provider.on_transcription(
            lambda s, text, role, final: transcriptions.append((text, role, final))
        )

        await provider._flush_transcription_buffer(session, "user")
        assert transcriptions == []

    async def test_flush_whitespace_only_buffer_is_noop(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        transcriptions = []
        provider.on_transcription(
            lambda s, text, role, final: transcriptions.append((text, role, final))
        )

        provider._transcription_buffers[(session.id, "user")] = ["  ", "\n"]
        await provider._flush_transcription_buffer(session, "user")
        assert transcriptions == []

    def test_clear_transcription_buffers(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")

        provider._transcription_buffers[("s1", "user")] = ["a"]
        provider._transcription_buffers[("s1", "assistant")] = ["b"]
        provider._transcription_buffers[("s2", "user")] = ["c"]

        provider._clear_transcription_buffers("s1")

        assert ("s1", "user") not in provider._transcription_buffers
        assert ("s1", "assistant") not in provider._transcription_buffers
        assert ("s2", "user") in provider._transcription_buffers

    # ── _receive_loop() ─────────────────────────────────────────

    async def test_receive_loop_no_state_returns(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        # No state registered — should return immediately
        await provider._receive_loop(session)

    async def test_receive_loop_ended_session_returns(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        session.state = VoiceSessionState.ENDED

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        await provider._receive_loop(session)

    async def test_receive_loop_processes_responses(self):
        """Test that _handle_response dispatches audio data to callbacks."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        session.state = VoiceSessionState.ACTIVE

        received_audio = []
        provider.on_audio(lambda s, audio: received_audio.append(audio))

        state = mod._GeminiSessionState(
            session=session,
            live_session=AsyncMock(),
            started_at=1000.0,
        )
        provider._sessions[session.id] = state

        # Test _handle_response directly instead of the full loop
        # (the loop has reconnection logic that spins)
        audio_response = SimpleNamespace(data=b"\xaa\xbb")
        await provider._handle_server_response(session, audio_response)

        assert len(received_audio) == 1

    # ── Callback error handling ─────────────────────────────────

    async def test_callback_exception_is_caught(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        def bad_cb(s):
            raise ValueError("Callback error")

        provider.on_speech_start(bad_cb)

        response = SimpleNamespace(
            voice_activity=SimpleNamespace(voice_activity_type="ACTIVITY_START"),
        )
        # Should not raise
        await provider._handle_server_response(session, response)

    async def test_audio_callback_exception_is_caught(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        def bad_cb(s, audio):
            raise ValueError("Audio error")

        provider.on_audio(bad_cb)

        response = SimpleNamespace(data=b"\x00")
        await provider._handle_server_response(session, response)

    async def test_error_callback_exception_is_caught(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        def bad_cb(s, code, msg):
            raise ValueError("Error callback error")

        provider.on_error(bad_cb)

        await provider._fire_error_callbacks(session, "test", "test")

    async def test_tool_call_callback_exception_is_caught(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        def bad_cb(s, cid, name, args):
            raise ValueError("Tool callback error")

        provider.on_tool_call(bad_cb)

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        fc = SimpleNamespace(id="fc-1", name="test", args=None)
        response = SimpleNamespace(
            tool_call=SimpleNamespace(function_calls=[fc]),
        )
        await provider._handle_server_response(session, response)

    # ── Async callback support ──────────────────────────────────

    async def test_async_callbacks_are_awaited(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        called = []

        async def async_cb(s):
            called.append("async")

        provider.on_speech_start(async_cb)

        response = SimpleNamespace(
            voice_activity=SimpleNamespace(voice_activity_type="ACTIVITY_START"),
        )
        await provider._handle_server_response(session, response)
        assert called == ["async"]
