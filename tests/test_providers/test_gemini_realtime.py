"""Tests for GeminiLiveProvider."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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

    def test_model_name_reports_the_live_model(self):
        """A log or a trace naming "the model" must not read a provider name."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(
            api_key="test-key",
            model="gemini-2.0-flash-live",
        )
        assert provider.model_name == "gemini-2.0-flash-live"

    def test_3x_models_disable_mid_session_reconfigure(self):
        """gemini-3.* models reject send_client_content with WS 1007
        after the first model turn and offer no documented dynamic
        system_instruction update. The provider must advertise that
        so callers route changes through session-start delivery.
        """
        mod = _load_provider()
        for model in (
            "gemini-3.8-live",
            "gemini-3.8-live-extended-thinking",
            "gemini-3.1-flash-live-preview",
            "gemini-3.0-flash-live",
            "gemini-3-experimental",
        ):
            provider = mod.GeminiLiveProvider(api_key="test-key", model=model)
            assert provider.supports_mid_session_reconfigure is False, (
                f"{model} must disable mid-session reconfigure"
            )

    def test_2x_models_allow_mid_session_reconfigure(self):
        """Pre-3.x Gemini Live models keep the old reconfigure behavior."""
        mod = _load_provider()
        for model in ("gemini-2.5-flash-live", "gemini-2.0-flash-live"):
            provider = mod.GeminiLiveProvider(api_key="test-key", model=model)
            assert provider.supports_mid_session_reconfigure is True, (
                f"{model} must keep mid-session reconfigure enabled"
            )

    def test_callback_registration(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")

        audio_cb = lambda session, audio: None  # noqa: E731
        transcription_cb = lambda session, text, role, final: None  # noqa: E731
        speech_start_cb = lambda session: None  # noqa: E731
        speech_end_cb = lambda session: None  # noqa: E731
        tool_call_cb = lambda session, cid, name, args: None  # noqa: E731
        tool_call_cancelled_cb = lambda session, ids: None  # noqa: E731
        response_start_cb = lambda session: None  # noqa: E731
        response_end_cb = lambda session: None  # noqa: E731
        error_cb = lambda session, code, msg: None  # noqa: E731

        provider.on_audio(audio_cb)
        provider.on_transcription(transcription_cb)
        provider.on_speech_start(speech_start_cb)
        provider.on_speech_end(speech_end_cb)
        provider.on_tool_call(tool_call_cb)
        provider.on_tool_call_cancelled(tool_call_cancelled_cb)
        provider.on_response_start(response_start_cb)
        provider.on_response_end(response_end_cb)
        provider.on_error(error_cb)

        assert audio_cb in provider._audio_callbacks
        assert transcription_cb in provider._transcription_callbacks
        assert speech_start_cb in provider._speech_start_callbacks
        assert speech_end_cb in provider._speech_end_callbacks
        assert tool_call_cb in provider._tool_call_callbacks
        assert tool_call_cancelled_cb in provider._tool_call_cancelled_callbacks
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
        # The cleaner is bound at import time in realtime_config: patch it there.
        with patch(
            "roomkit.providers.gemini.realtime_config.clean_gemini_schema", return_value=None
        ):
            config = provider._build_config(tools=tools)
        assert config.tools is not None

    def test_build_config_with_provider_config_options(self):
        # Pinned to a pre-3.8 model on purpose: affective dialog, proactivity
        # and a thinking budget are exactly the three fields the 3.8 family
        # stopped accepting, and this test is the one that guards them for
        # the generations that still do.
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-2.0-flash-live-001")

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

    # ── model profile: what each generation's setup accepts ─────

    def test_build_config_drops_the_fields_3_8_refuses(self, caplog):
        """A 3.1 config opened against 3.8 must connect, not 400.

        Google removed affective dialog, made proactive audio permanent and
        dropped thinking_config from gemini-3.8-live. Forwarding any of the
        three is a server error, so they are dropped here and reported.
        """
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")

        pc = {
            "enable_affective_dialog": True,
            "proactive_audio": False,
            "thinking_budget": 1024,
        }
        with caplog.at_level("WARNING"):
            config = provider._build_config(provider_config=pc)

        assert config.enable_affective_dialog is None
        assert config.proactivity is None
        assert config.thinking_config is None
        for field in ("enable_affective_dialog", "proactive_audio", "thinking_budget"):
            assert field in caplog.text

    def test_build_config_warns_once_per_field(self, caplog):
        """Reconfigures rebuild the config; within a session the warning must not follow."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")

        pc = {"enable_affective_dialog": True}
        warned: set[str] = set()
        with caplog.at_level("WARNING"):
            provider._build_config(provider_config=pc, warned=warned)
            provider._build_config(provider_config=pc, warned=warned)

        assert caplog.text.count("enable_affective_dialog") == 1

    def test_every_session_hears_the_dropped_field_once(self, caplog):
        """The record is the session's. Kept on the provider, only the first
        call of the process was told and every later one lost the setting
        in silence."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")

        pc = {"enable_affective_dialog": True}
        with caplog.at_level("WARNING"):
            provider._build_config(provider_config=pc, warned=set())
            provider._build_config(provider_config=pc, warned=set())

        assert caplog.text.count("enable_affective_dialog") == 2

    def test_build_config_keeps_the_pre_3_8_fields_on_older_models(self):
        """2.0 Flash Live still takes all three: no retroactive narrowing."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-2.0-flash-live-001")

        pc = {
            "enable_affective_dialog": True,
            "proactive_audio": True,
            "thinking_budget": 1024,
        }
        config = provider._build_config(provider_config=pc)

        assert config.enable_affective_dialog is True
        assert config.proactivity is not None
        assert config.thinking_config is not None

    def test_build_config_unknown_model_keeps_todays_behaviour(self):
        """An id the catalog never saw must not lose its configuration."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-9.9-live-future")

        config = provider._build_config(provider_config={"enable_affective_dialog": True})
        assert config.enable_affective_dialog is True

    def test_build_config_thinking_level_on_extended_thinking(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(
            api_key="test-key", model="gemini-3.8-live-extended-thinking"
        )

        config = provider._build_config(provider_config={"thinking_level": "HIGH"})
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_level == "HIGH"

    def test_extended_thinking_gets_a_level_even_when_none_is_named(self):
        """The model closes the socket on a missing level, it does not pick one.

        Live run 2026-09-17: `1007 Thinking level must be specified for this
        model`. The docs call the level supported; the server calls it
        required.
        """
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(
            api_key="test-key", model="gemini-3.8-live-extended-thinking"
        )

        config = provider._build_config()

        assert config.thinking_config is not None
        assert config.thinking_config.thinking_level == "LOW"

    def test_a_named_level_wins_over_the_default(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(
            api_key="test-key", model="gemini-3.8-live-extended-thinking"
        )

        config = provider._build_config(provider_config={"thinking_level": "high"})
        assert config.thinking_config.thinking_level == "HIGH"

    def test_no_level_is_invented_for_a_model_that_takes_none(self):
        """Plain 3.8 refuses thinking_config outright; the default must not leak."""
        mod = _load_provider()
        for model in ("gemini-3.8-live", "gemini-2.0-flash-live-001"):
            provider = mod.GeminiLiveProvider(api_key="test-key", model=model)
            assert provider._build_config().thinking_config is None, model

    def test_build_config_rejects_a_thinking_level_the_model_refuses(self):
        """`minimal` is refused upstream: refuse it before spending a round trip."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(
            api_key="test-key", model="gemini-3.8-live-extended-thinking"
        )

        with pytest.raises(ValueError, match="thinking_level"):
            provider._build_config(provider_config={"thinking_level": "minimal"})

    def test_build_config_thinking_level_ignored_on_plain_3_8(self, caplog):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")

        with caplog.at_level("WARNING"):
            config = provider._build_config(provider_config={"thinking_level": "high"})

        assert config.thinking_config is None
        assert "thinking_level" in caplog.text

    # ── the SDK only warns, so the boundary refuses ─────────────

    def test_an_unknown_turn_coverage_is_refused(self):
        """google-genai warns and forwards; the server then kills the setup."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")

        with pytest.raises(ValueError, match="turn_coverage"):
            provider._build_config(provider_config={"turn_coverage": "TURN_INCLUDES_EVERYTHING"})

    def test_an_unknown_transcription_mode_is_refused(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")

        with pytest.raises(ValueError, match="transcription.mode"):
            provider._build_config(provider_config={"transcription": {"mode": "creative"}})

    def test_an_unknown_tool_behavior_is_refused(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")

        tools = [{"name": "t", "description": "d", "parameters": {}, "behavior": "EVENTUALLY"}]
        with pytest.raises(ValueError, match="tool behavior"):
            provider._build_config(tools=tools)

    async def test_an_unknown_tool_response_scheduling_is_refused(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=_make_mock_live_session(),
            provider_config={"tool_response_scheduling": "EVENTUALLY"},
        )
        provider._sessions[session.id] = state
        await _issue_calls(provider, session, ("lookup", "call-1"))

        with pytest.raises(ValueError, match="tool_response_scheduling"):
            await provider.submit_tool_result(session, "call-1", '{"ok": true}')
        # Refused before the call left the book: it is still owed, and a
        # later abandonment reports it (RMK-502).
        assert provider._holds_tool_call(session, "call-1")

    # ── turn coverage and transcription options ─────────────────

    def test_build_config_turn_coverage_is_settable(self):
        """3.8 defaults to folding all video into the turn, which bills more."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")

        pc = {"turn_coverage": "turn_includes_only_activity"}
        config = provider._build_config(provider_config=pc)
        assert config.realtime_input_config.turn_coverage == "TURN_INCLUDES_ONLY_ACTIVITY"

    def test_build_config_transcription_options(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")

        pc = {
            "transcription": {
                "language_auto": True,
                "language_hints": ["fr-FR", "en-US"],
                "custom_vocabulary": ["RoomKit", "Tarjan"],
                "diarization": True,
            }
        }
        config = provider._build_config(provider_config=pc)
        inbound = config.input_audio_transcription
        assert inbound.language_auto is not None
        assert inbound.language_hints.language_codes == ["fr-FR", "en-US"]
        assert inbound.custom_vocabulary == ["RoomKit", "Tarjan"]
        assert inbound.diarization is True

    def test_build_config_transcription_defaults_stay_bare(self):
        """No options means the previous behaviour, not an empty-list config."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")

        config = provider._build_config()
        assert config.input_audio_transcription is not None
        assert config.input_audio_transcription.language_auto is None
        assert config.output_audio_transcription is not None

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

        result = await provider.inject_text(session, "Hello!")
        assert result.status == "sent"

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

    async def test_inject_text_unknown_role_defaults_to_user(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state

        await provider.inject_text(session, "Test", role="tool")

        call_kwargs = mock_live_session.send_client_content.call_args[1]
        assert call_kwargs["turns"].role == "user"

    async def test_an_assistant_line_is_asked_for_not_heard_from_the_user(self):
        """No turn makes the model speak a text: the line is an instruction to say it."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state

        await provider.inject_text(session, "Bienvenue !", role="assistant")

        turns = mock_live_session.send_client_content.call_args[1]["turns"]
        text = turns.parts[0].text
        assert text.startswith("Say this to the user now")
        assert '"Bienvenue !"' in text

    async def test_inject_text_no_session(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        result = await provider.inject_text(session, "No session")
        assert result.status == "not_sent" and result.retryable

    async def test_inject_text_after_audio_uses_realtime_input(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            realtime_input_sent=True,
        )
        provider._sessions[session.id] = state

        await provider.inject_text(session, "Hello mid-audio!")

        mock_live_session.send_client_content.assert_not_awaited()
        mock_live_session.send_realtime_input.assert_awaited_once()
        call_kwargs = mock_live_session.send_realtime_input.call_args[1]
        assert call_kwargs["text"] == "Hello mid-audio!"

    async def test_inject_text_silent_after_audio(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            realtime_input_sent=True,
        )
        provider._sessions[session.id] = state

        await provider.inject_text(session, "Context only", silent=True)

        mock_live_session.send_client_content.assert_not_awaited()
        call_kwargs = mock_live_session.send_realtime_input.call_args[1]
        assert "[Context update, do not respond to this]" in call_kwargs["text"]
        assert "Context only" in call_kwargs["text"]

    async def test_inject_text_model_role_after_audio(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            realtime_input_sent=True,
        )
        provider._sessions[session.id] = state

        await provider.inject_text(session, "Model response", role="model")

        mock_live_session.send_client_content.assert_not_awaited()
        call_kwargs = mock_live_session.send_realtime_input.call_args[1]
        assert "[Assistant previously said]" in call_kwargs["text"]
        assert "Model response" in call_kwargs["text"]

    async def test_inject_text_queued_during_tool_calls(self):
        # Pinned to a pre-3.8 model: the queue exists because the API refuses
        # input while a blocking call is outstanding. From 3.8 tools run in
        # the background and there is nothing to wait for, which
        # test_injection_is_not_held_back_by_a_background_call covers.
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-2.0-flash-live-001")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            blocking_call_ids={"call-1"},
        )
        provider._sessions[session.id] = state
        provider._book_tool_call(session, "call-1", "lookup")

        result = await provider.inject_text(session, "Queued text", role="user", silent=True)
        assert result.status == "unknown" and not result.retryable
        assert result.reason == "voice_provider_queued"

        # Should not send anything yet
        mock_live_session.send_client_content.assert_not_awaited()
        mock_live_session.send_realtime_input.assert_not_awaited()
        # Should be queued
        assert len(state.queued_text_injections) == 1
        assert state.queued_text_injections[0] == ("Queued text", "user", True)

    async def test_inject_text_flushed_after_tool_result(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            queued_text_injections=[("Queued text", "user", False)],
        )
        provider._sessions[session.id] = state
        provider._book_tool_call(session, "call-1", "lookup")

        await provider.submit_tool_result(session, "call-1", '{"ok": true}')

        # Tool response sent
        mock_live_session.send_tool_response.assert_awaited_once()
        # Queued text flushed via send_client_content (no audio sent)
        mock_live_session.send_client_content.assert_awaited_once()
        assert len(state.queued_text_injections) == 0

    async def test_an_injection_made_during_the_send_queues_behind_the_held_one(self):
        """RMK-441: the blocking call is released before its result is sent,
        so an injection made while the send yields finds nothing blocking; it
        queues behind the one the call held, and goes out after it."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-2.0-flash-live-001")
        session = _make_session()
        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            blocking_call_ids={"call-1"},
        )
        provider._sessions[session.id] = state
        provider._book_tool_call(session, "call-1", "lookup")
        await provider.inject_text(session, "held", role="user")

        async def send(**kwargs):
            await provider.inject_text(session, "during", role="user")

        mock_live_session.send_tool_response = AsyncMock(side_effect=send)

        await provider.submit_tool_result(session, "call-1", '{"ok": true}')

        sent = [
            str(call.kwargs)
            for call in mock_live_session.send_client_content.await_args_list
            + mock_live_session.send_realtime_input.await_args_list
        ]
        assert len(sent) == 2
        assert "held" in sent[0] and "during" in sent[1]
        assert state.queued_text_injections == []

    async def test_send_audio_sets_realtime_input_sent(self):
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

        assert state.realtime_input_sent is False
        await provider.send_audio(session, b"\x00\x00")
        assert state.realtime_input_sent is True

    async def test_send_image_after_audio_uses_realtime_input(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            realtime_input_sent=True,
        )
        provider._sessions[session.id] = state

        await provider.inject_image(
            session, b"\x89PNG", "image/png", prompt="Describe this", silent=False
        )

        mock_live_session.send_client_content.assert_not_awaited()
        # Two calls: one for text prompt, one for media
        assert mock_live_session.send_realtime_input.await_count == 2
        first_call = mock_live_session.send_realtime_input.call_args_list[0][1]
        assert first_call["text"] == "Describe this"
        second_call = mock_live_session.send_realtime_input.call_args_list[1][1]
        assert second_call["media"].mime_type == "image/png"

    async def test_send_image_before_audio_uses_client_content(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            realtime_input_sent=False,
        )
        provider._sessions[session.id] = state

        await provider.inject_image(
            session, b"\x89PNG", "image/png", prompt="Describe", silent=False
        )

        mock_live_session.send_client_content.assert_awaited_once()
        mock_live_session.send_realtime_input.assert_not_awaited()

    async def test_send_image_after_audio_no_prompt(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            realtime_input_sent=True,
        )
        provider._sessions[session.id] = state

        await provider.inject_image(session, b"\x89PNG", "image/png")

        mock_live_session.send_client_content.assert_not_awaited()
        # Only one call: media only (no prompt, not silent)
        mock_live_session.send_realtime_input.assert_awaited_once()
        call_kwargs = mock_live_session.send_realtime_input.call_args[1]
        assert "media" in call_kwargs

    async def test_send_image_after_audio_silent_no_prompt(self):
        """Silent image without prompt must send a 'do not respond' instruction."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            realtime_input_sent=True,
        )
        provider._sessions[session.id] = state

        await provider.inject_image(session, b"\x89PNG", "image/png", silent=True)

        mock_live_session.send_client_content.assert_not_awaited()
        # Two calls: silent instruction text, then media
        assert mock_live_session.send_realtime_input.await_count == 2
        first = mock_live_session.send_realtime_input.call_args_list[0][1]
        assert "do not respond" in first["text"].lower()
        second = mock_live_session.send_realtime_input.call_args_list[1][1]
        assert "media" in second

    async def test_send_image_after_audio_silent_with_prompt(self):
        """Silent image with prompt: prompt gets the 'do not respond' prefix."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            realtime_input_sent=True,
        )
        provider._sessions[session.id] = state

        await provider.inject_image(
            session, b"\x89PNG", "image/png", prompt="Logo update", silent=True
        )

        mock_live_session.send_client_content.assert_not_awaited()
        assert mock_live_session.send_realtime_input.await_count == 2
        first = mock_live_session.send_realtime_input.call_args_list[0][1]
        assert "[Context update, do not respond to this]" in first["text"]
        assert "Logo update" in first["text"]

    async def test_inject_text_model_role_and_silent_after_audio(self):
        """Both role='model' and silent=True: both prefixes applied."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            realtime_input_sent=True,
        )
        provider._sessions[session.id] = state

        await provider.inject_text(session, "Noted", role="model", silent=True)

        text = mock_live_session.send_realtime_input.call_args[1]["text"]
        assert "[Assistant previously said]" in text
        assert "[Context update, do not respond to this]" in text
        assert "Noted" in text

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
        await _issue_calls(provider, session, ("lookup", "call-1"))

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
        await _issue_calls(provider, session, ("lookup", "call-2"))

        await provider.submit_tool_result(session, "call-2", "plain text result")
        mock_live_session.send_tool_response.assert_awaited_once()

    async def test_a_result_carries_the_name_of_the_call_it_answers(self):
        """The id alone was enough through 3.1 and the name went out empty.

        gemini-3.8-live-extended-thinking reads an unnamed FunctionResponse as
        a failed call and tells the user a system error occurred; under the
        call's name the same payload is read as the result. The name is kept
        per call in flight and released with it.
        """
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()
        state = mod._GeminiSessionState(session=session, live_session=_make_mock_live_session())
        provider._sessions[session.id] = state

        await provider._on_tool_call(
            session,
            state,
            SimpleNamespace(
                function_calls=[SimpleNamespace(name="lookup_contact", id="c1", args={"q": "x"})]
            ),
        )
        assert provider._open_tool_calls[session.id] == {"c1": "lookup_contact"}

        await provider.submit_tool_result(session, "c1", '{"matches": []}')
        sent = state.live_session.send_tool_response.await_args.kwargs["function_responses"][0]
        assert sent.id == "c1"
        assert sent.name == "lookup_contact"
        assert provider._open_tool_calls[session.id] == {}, "released with the call"

    async def test_a_call_is_released_as_its_result_goes(self):
        """Off the books before the send yields, as the channel frees the id
        at the same step: a call Gemini issues under it meanwhile is a new
        call (RMK-441), which keeps its booking. The response still carries
        the call's name."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()
        state = mod._GeminiSessionState(session=session, live_session=_make_mock_live_session())
        provider._sessions[session.id] = state
        issued = SimpleNamespace(function_calls=[SimpleNamespace(name="lookup", id="c1", args={})])
        await provider._on_tool_call(session, state, issued)
        during_send: list[bool] = []

        async def send(**kwargs):
            during_send.append(provider._holds_tool_call(session, "c1"))
            await provider._on_tool_call(session, state, issued)  # issued again meanwhile

        state.live_session.send_tool_response = AsyncMock(side_effect=send)

        await provider.submit_tool_result(session, "c1", "ok")

        assert during_send == [False]
        sent = state.live_session.send_tool_response.await_args_list[0].kwargs[
            "function_responses"
        ][0]
        assert sent.name == "lookup"
        assert provider._open_tool_calls[session.id].get("c1") == "lookup"

    async def test_a_result_for_a_call_never_issued_is_dropped(self):
        """Nothing to name it after, and the server knows no such id: nothing
        goes out, as on every realtime provider (RFC §12.4)."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()
        state = mod._GeminiSessionState(session=session, live_session=_make_mock_live_session())
        provider._sessions[session.id] = state

        await provider.submit_tool_result(session, "unknown", '{"ok": true}')
        state.live_session.send_tool_response.assert_not_awaited()

    async def test_a_reconnect_forgets_the_names_with_the_calls(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()
        state = mod._GeminiSessionState(session=session, live_session=_make_mock_live_session())
        provider._sessions[session.id] = state
        await provider._on_tool_call(
            session,
            state,
            SimpleNamespace(function_calls=[SimpleNamespace(name="lookup", id="c1", args={})]),
        )

        await provider._release_calls_lost_with_the_connection(state)
        assert provider._open_tool_calls.get(session.id, {}) == {}

    # ── background tool calls (3.8 Live) ────────────────────────

    def test_declarations_run_in_background_on_3_8(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")

        tools = [{"name": "get_weather", "description": "w", "parameters": {}}]
        config = provider._build_config(tools=tools)
        assert config.tools[0].function_declarations[0].behavior == "NON_BLOCKING"

    def test_declarations_still_block_on_older_models(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-2.0-flash-live-001")

        tools = [{"name": "get_weather", "description": "w", "parameters": {}}]
        config = provider._build_config(tools=tools)
        assert config.tools[0].function_declarations[0].behavior == "BLOCKING"

    def test_a_tool_may_ask_to_block_where_the_model_allows_it(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")

        tools = [{"name": "t", "description": "d", "parameters": {}, "behavior": "BLOCKING"}]
        config = provider._build_config(tools=tools)
        assert config.tools[0].function_declarations[0].behavior == "BLOCKING"

    def test_blocking_is_downgraded_where_the_model_refuses_it(self, caplog):
        """extended-thinking answers a hard error to BLOCKING: do not send it."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(
            api_key="test-key", model="gemini-3.8-live-extended-thinking"
        )

        tools = [{"name": "t", "description": "d", "parameters": {}, "behavior": "BLOCKING"}]
        with caplog.at_level("WARNING"):
            config = provider._build_config(tools=tools)

        assert config.tools[0].function_declarations[0].behavior == "NON_BLOCKING"
        assert "BLOCKING" in caplog.text

    async def test_no_scheduling_is_sent_unless_it_is_asked_for(self):
        """A default here cost a live session.

        gemini-3.8-live-extended-thinking answers `1007 Function response
        scheduling is not supported for this model` and closes the socket, and
        the models that do take the field deliver a background result sensibly
        without it. Opt-in only.
        """
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(session=session, live_session=mock_live_session)
        provider._sessions[session.id] = state
        await _issue_calls(provider, session, ("lookup", "call-1"))

        await provider.submit_tool_result(session, "call-1", '{"ok": true}')

        sent = mock_live_session.send_tool_response.await_args.kwargs["function_responses"][0]
        assert sent.scheduling is None

    async def test_a_model_that_refuses_scheduling_never_receives_it(self, caplog):
        """Verified live: extended-thinking closes the session over this field."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(
            api_key="test-key", model="gemini-3.8-live-extended-thinking"
        )
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            provider_config={"tool_response_scheduling": "WHEN_IDLE"},
        )
        provider._sessions[session.id] = state
        await _issue_calls(provider, session, ("lookup", "call-1"))

        with caplog.at_level("WARNING"):
            await provider.submit_tool_result(session, "call-1", '{"ok": true}')

        sent = mock_live_session.send_tool_response.await_args.kwargs["function_responses"][0]
        assert sent.scheduling is None
        assert "tool_response_scheduling" in caplog.text

    async def test_the_scheduling_is_settable(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            provider_config={"tool_response_scheduling": "interrupt"},
        )
        provider._sessions[session.id] = state
        await _issue_calls(provider, session, ("lookup", "call-1"))

        await provider.submit_tool_result(session, "call-1", '{"ok": true}')

        sent = mock_live_session.send_tool_response.await_args.kwargs["function_responses"][0]
        assert sent.scheduling == "INTERRUPT"

    async def test_a_blocking_result_keeps_the_pre_3_8_wire(self):
        """The model is already waiting: there is nothing to schedule."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-2.0-flash-live-001")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            blocking_call_ids={"call-1"},
        )
        provider._sessions[session.id] = state
        await _issue_calls(provider, session, ("lookup", "call-1"))

        await provider.submit_tool_result(session, "call-1", '{"ok": true}')

        sent = mock_live_session.send_tool_response.await_args.kwargs["function_responses"][0]
        assert sent.scheduling is None
        assert state.blocking_call_ids == set()

    async def test_injection_is_not_held_back_by_a_background_call(self):
        """Queueing here would delay input the model is perfectly able to take."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
        )
        provider._sessions[session.id] = state
        provider._book_tool_call(session, "call-1", "lookup")

        result = await provider.inject_text(session, "bonjour")

        assert result.status == "sent"
        assert state.queued_text_injections == []

    async def test_a_blocking_tool_on_3_8_still_holds_the_injection(self):
        """The pair the two features form, which neither test covered alone.

        A tool may opt back into BLOCKING on 3.8. Deriving the queue from the
        model's default instead of the outstanding call let that injection go
        straight to a socket the API was refusing input on.
        """
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()

        tools = [
            {"name": "charge", "description": "d", "parameters": {}, "behavior": "BLOCKING"},
            {"name": "lookup", "description": "d", "parameters": {}},
        ]
        state = mod._GeminiSessionState(
            session=session,
            live_session=_make_mock_live_session(),
            blocking_tool_names=provider._blocking_tool_names(tools),
        )
        provider._sessions[session.id] = state
        assert state.blocking_tool_names == {"charge"}

        await provider._on_tool_call(
            session,
            state,
            SimpleNamespace(function_calls=[SimpleNamespace(name="charge", id="c1", args={})]),
        )
        assert state.blocking_call_ids == {"c1"}
        assert set(provider._open_tool_calls[session.id]) == {"c1"}

        result = await provider.inject_text(session, "bonjour")
        assert result.reason == "voice_provider_queued"

        await provider.submit_tool_result(session, "c1", '{"ok": true}')
        sent = state.live_session.send_tool_response.await_args.kwargs["function_responses"][0]
        assert sent.scheduling is None, "a call the API waited on needs no scheduling"
        assert state.blocking_call_ids == set()
        assert provider._open_tool_calls[session.id] == {}

    async def test_a_background_call_on_3_8_registers_no_blocking_id(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()

        state = mod._GeminiSessionState(
            session=session,
            live_session=_make_mock_live_session(),
            blocking_tool_names=provider._blocking_tool_names(
                [{"name": "lookup", "description": "d", "parameters": {}}]
            ),
        )
        provider._sessions[session.id] = state

        await provider._on_tool_call(
            session,
            state,
            SimpleNamespace(function_calls=[SimpleNamespace(name="lookup", id="c1", args={})]),
        )

        assert state.blocking_call_ids == set()
        assert set(provider._open_tool_calls[session.id]) == {"c1"}, (
            "a background call is still in the books"
        )
        assert (await provider.inject_text(session, "bonjour")).status == "sent"

    async def test_injection_still_waits_on_a_blocking_call(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-2.0-flash-live-001")
        session = _make_session()

        mock_live_session = _make_mock_live_session()
        state = mod._GeminiSessionState(
            session=session,
            live_session=mock_live_session,
            blocking_call_ids={"call-1"},
        )
        provider._sessions[session.id] = state
        provider._book_tool_call(session, "call-1", "lookup")

        result = await provider.inject_text(session, "bonjour")

        assert result.reason == "voice_provider_queued"
        assert len(state.queued_text_injections) == 1

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
        await _issue_calls(provider, session, ("lookup", "call-3"))

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
        await _issue_calls(provider, session, ("lookup", "call-4"))

        # Large result > 16384 chars
        large_result = "x" * 20000
        await provider.submit_tool_result(session, "call-4", large_result)
        assert state.tool_result_bytes == 20000

    async def test_submit_tool_result_no_session_is_dropped(self, caplog):
        """A result for a session the provider does not hold goes out
        nowhere, with the log every provider writes (RFC §12.4)."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        with caplog.at_level(logging.INFO):
            await provider.submit_tool_result(session, "call-1", "result")
        assert "dropped: abandoned or never issued" in caplog.text

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
        provider._transcription_buffer._buffers[(session.id, "user")] = ["chunk"]

        await provider.disconnect(session)

        assert (session.id, "user") not in provider._transcription_buffer._buffers

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

    async def test_a_call_without_an_id_is_fired_but_not_tracked(self):
        """Nothing could answer or cancel it, so the books have nothing to keep."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state

        tool_calls = []
        provider.on_tool_call(lambda s, cid, name, args: tool_calls.append((cid, name)))

        fc = SimpleNamespace(id=None, name="ping", args=None)
        await provider._handle_server_response(
            session, SimpleNamespace(tool_call=SimpleNamespace(function_calls=[fc]))
        )

        # The channel refuses an id-less call and reports it (RFC §12.4).
        assert tool_calls == [("", "ping")]
        assert provider._open_tool_calls.get(session.id, {}) == {}

    async def test_a_second_call_under_an_id_in_flight_keeps_the_first(self):
        """The channel refuses the second and sends nothing: the first call's
        response must still name its own tool (RFC §12.4)."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()

        state = mod._GeminiSessionState(session=session)
        provider._sessions[session.id] = state
        provider.on_tool_call(lambda s, cid, name, args: None)

        calls = [
            SimpleNamespace(id="c1", name="lookup", args=None),
            SimpleNamespace(id="c1", name="other_tool", args=None),
        ]
        await provider._handle_server_response(
            session, SimpleNamespace(tool_call=SimpleNamespace(function_calls=calls))
        )

        assert provider._open_tool_calls[session.id] == {"c1": "lookup"}

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

    # ── end of interaction vs end of turn (3.8 Live) ────────────

    @staticmethod
    def _content(**fields):
        """A server_content namespace with the fields the handler reads."""
        base = {
            "model_turn": None,
            "turn_complete": False,
            "interrupted": False,
            "input_transcription": None,
            "output_transcription": None,
            "interaction_status": None,
        }
        base.update(fields)
        return SimpleNamespace(server_content=SimpleNamespace(**base))

    async def test_several_turns_inside_one_interaction_end_the_response_once(self):
        """3.8 speaks while it reasons: only IDLE means the request is over."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()
        state = mod._GeminiSessionState(session=session, response_started=True)
        provider._sessions[session.id] = state

        ends = []
        provider.on_response_end(lambda s: ends.append(s.id))

        await provider._handle_server_response(
            session, self._content(turn_complete=True, interaction_status="IN_PROGRESS")
        )
        await provider._handle_server_response(
            session, self._content(turn_complete=True, interaction_status="IN_PROGRESS")
        )
        assert ends == []
        assert state.turn_count == 2

        await provider._handle_server_response(
            session, self._content(turn_complete=True, interaction_status="IDLE")
        )
        assert ends == [session.id]
        assert state.response_started is False
        assert state.awaiting_new_user_utterance is True

    async def test_idle_without_turn_complete_still_ends_the_response(self):
        """The two signals are independent once the server reports its state."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()
        state = mod._GeminiSessionState(session=session, response_started=True)
        provider._sessions[session.id] = state

        ends = []
        provider.on_response_end(lambda s: ends.append(s.id))

        await provider._handle_server_response(
            session, self._content(interaction_status="IN_PROGRESS")
        )
        await provider._handle_server_response(session, self._content(interaction_status="IDLE"))

        assert ends == [session.id]
        assert state.turn_count == 0

    async def test_a_server_that_reports_no_status_keeps_the_old_meaning(self):
        """2.0 Flash Live sends no interaction_status and must still hand back."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-2.0-flash-live-001")
        session = _make_session()
        state = mod._GeminiSessionState(session=session, response_started=True)
        provider._sessions[session.id] = state

        ends = []
        provider.on_response_end(lambda s: ends.append(s.id))

        await provider._handle_server_response(session, self._content(turn_complete=True))
        await provider._handle_server_response(session, self._content(turn_complete=True))

        assert ends == [session.id, session.id]
        assert state.reports_interaction_status is False

    async def test_the_sdk_enum_is_read_as_well_as_the_bare_string(self):
        """Production receives the enum; every other test here passes a string."""
        from google.genai import types

        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()
        state = mod._GeminiSessionState(session=session, response_started=True)
        provider._sessions[session.id] = state

        ends = []
        provider.on_response_end(lambda s: ends.append(s.id))

        await provider._handle_server_response(
            session,
            self._content(
                turn_complete=True, interaction_status=types.InteractionStatus.IN_PROGRESS
            ),
        )
        assert ends == []

        await provider._handle_server_response(
            session, self._content(interaction_status=types.InteractionStatus.IDLE)
        )
        assert ends == [session.id]

    async def test_an_unrecognised_status_does_not_mute_turn_complete(self):
        """Latching on UNSPECIFIED would retire the only signal such a server sends."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()
        state = mod._GeminiSessionState(session=session, response_started=True)
        provider._sessions[session.id] = state

        ends = []
        provider.on_response_end(lambda s: ends.append(s.id))

        await provider._handle_server_response(
            session,
            self._content(turn_complete=True, interaction_status="INTERACTION_STATUS_UNSPECIFIED"),
        )

        assert state.reports_interaction_status is False
        assert ends == [session.id]

    async def test_barge_in_still_ends_the_response_mid_interaction(self):
        """Interruption does not wait for IDLE: the user took the floor."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()
        state = mod._GeminiSessionState(session=session, response_started=True)
        provider._sessions[session.id] = state

        ends = []
        provider.on_response_end(lambda s: ends.append(s.id))

        await provider._handle_server_response(
            session, self._content(interaction_status="IN_PROGRESS")
        )
        await provider._handle_server_response(session, self._content(interrupted=True))

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
                response_token_count=50,
                total_token_count=150,
            ),
        )
        await provider._handle_server_response(session, response)

        assert session._last_usage["input_tokens"] == 100
        assert session._last_usage["output_tokens"] == 50
        assert session._last_usage["total_token_count"] == 150

    async def test_usage_metadata_reports_its_breakdown(self):
        """The two totals cannot say what a spoken session spent context on.

        Audio, text and the cached share are priced apart, so the modality
        counts ride the usage the host records, as they do on OpenAI.
        """
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        provider._sessions[session.id] = mod._GeminiSessionState(session=session)

        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=23520,
                response_token_count=44,
                total_token_count=23564,
                cached_content_token_count=12000,
                thoughts_token_count=300,
                prompt_tokens_details=[
                    SimpleNamespace(modality=SimpleNamespace(name="AUDIO"), token_count=3520),
                    SimpleNamespace(modality=SimpleNamespace(name="TEXT"), token_count=20000),
                ],
                response_tokens_details=[
                    SimpleNamespace(modality=SimpleNamespace(name="AUDIO"), token_count=44),
                ],
            ),
        )
        await provider._handle_server_response(session, response)

        usage = session._last_usage
        assert usage["input_tokens"] == 23520
        # Thinking is billed as output, outside the response count (RFC §6).
        assert usage["output_tokens"] == 344
        assert usage["prompt_tokens_details"] == {"AUDIO": 3520, "TEXT": 20000}
        assert usage["response_tokens_details"] == {"AUDIO": 44}
        assert usage["cached_content_token_count"] == 12000
        assert usage["thoughts_token_count"] == 300

    async def test_usage_metadata_without_a_breakdown_adds_nothing(self):
        """A server that sends only the totals leaves the payload as it was."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        provider._sessions[session.id] = mod._GeminiSessionState(session=session)

        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=10, response_token_count=2)
        )
        await provider._handle_server_response(session, response)

        assert session._last_usage == {"input_tokens": 10, "output_tokens": 2}

    async def test_the_breakdown_reaches_on_usage(self):
        """A host bills the call from the callback, not from a private attribute."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        provider._sessions[session.id] = mod._GeminiSessionState(session=session)
        seen: list[dict] = []
        provider.on_usage(lambda _s, usage: seen.append(usage))

        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=23520,
                response_token_count=44,
                cached_content_token_count=12000,
                prompt_tokens_details=[
                    SimpleNamespace(modality=SimpleNamespace(name="AUDIO"), token_count=3520),
                    SimpleNamespace(modality=SimpleNamespace(name="TEXT"), token_count=20000),
                ],
            ),
        )
        await provider._handle_server_response(session, response)
        await asyncio.sleep(0.01)

        assert len(seen) == 1
        assert seen[0]["input_tokens"] == 23520
        assert seen[0]["cached_content_token_count"] == 12000
        assert seen[0]["prompt_tokens_details"] == {"AUDIO": 3520, "TEXT": 20000}
        assert session.last_usage == seen[0]

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
        provider._transcription_buffer._buffers[(session.id, "user")] = ["Hello ", "world"]

        await provider._flush_transcription_buffer(session, "user")

        assert len(transcriptions) == 1
        assert transcriptions[0] == ("Hello world", "user", True)
        assert (session.id, "user") not in provider._transcription_buffer._buffers

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

        provider._transcription_buffer._buffers[(session.id, "user")] = ["  ", "\n"]
        await provider._flush_transcription_buffer(session, "user")
        assert transcriptions == []

    def test_clear_transcription_buffers(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")

        provider._transcription_buffer._buffers[("s1", "user")] = ["a"]
        provider._transcription_buffer._buffers[("s1", "assistant")] = ["b"]
        provider._transcription_buffer._buffers[("s2", "user")] = ["c"]

        provider._clear_transcription_buffers("s1")

        assert ("s1", "user") not in provider._transcription_buffer._buffers
        assert ("s1", "assistant") not in provider._transcription_buffer._buffers
        assert ("s2", "user") in provider._transcription_buffer._buffers

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

        await provider._fire(provider._error_callbacks, session, "test", "test", label="error")

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

    # ── reconfigure() clears queued injections ─────────────────

    async def test_reconfigure_clears_queued_injections(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        session = _make_session()
        session.state = VoiceSessionState.ACTIVE

        state = mod._GeminiSessionState(
            session=session,
            live_session=_make_mock_live_session(),
        )
        provider._sessions[session.id] = state
        provider._book_tool_call(session, "call-1", "lookup")

        # Queue some text and image injections (happens when tool calls are pending)
        state.queued_text_injections.append(("stale text", "user", False))
        state.queued_injections.append((b"\x89PNG", "image/png", "", False))

        # Mock _reconnect to avoid real connection logic, and the new
        # connection's receive loop, which starts before reconfigure returns
        # and would spin on the mock session
        provider._reconnect = AsyncMock()
        provider._receive_loop = AsyncMock()

        await provider.reconfigure(session, system_prompt="new prompt")

        assert state.queued_text_injections == []
        assert state.queued_injections == []

        # Clean up the receive_task started by reconfigure()
        if state.receive_task is not None:
            state.receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await state.receive_task

    # ── _mime_cache instance isolation ─────────────────────────

    def test_mime_cache_instance_isolation(self):
        mod = _load_provider()
        p1 = mod.GeminiLiveProvider(api_key="test-key")
        p2 = mod.GeminiLiveProvider(api_key="test-key")

        p1._make_audio_blob(b"\x00", 16000)
        assert 16000 in p1._mime_cache
        assert 16000 not in p2._mime_cache


def _blocking_call_state(model: str = "gemini-3.8-live", tool: str = "lookup"):
    """A session with one blocking call to *tool* outstanding and a text injection
    queued behind it."""
    mod = _load_provider()
    provider = mod.GeminiLiveProvider(api_key="test-key", model=model)
    session = _make_session()
    live = _make_mock_live_session()
    state = mod._GeminiSessionState(
        session=session,
        live_session=live,
        blocking_call_ids={"call-1"},
        queued_text_injections=[("Queued text", "user", False)],
    )
    provider._sessions[session.id] = state
    provider._book_tool_call(session, "call-1", tool)
    return provider, session, state, live


def _background_call_state():
    """A 3.8 session whose only tool runs in the background; no call issued yet."""
    mod = _load_provider()
    provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
    session = _make_session()
    live = _make_mock_live_session()
    state = mod._GeminiSessionState(
        session=session,
        live_session=live,
        blocking_tool_names=provider._blocking_tool_names(
            [{"name": "lookup", "description": "d", "parameters": {}}]
        ),
    )
    provider._sessions[session.id] = state
    return provider, session, state, live


async def _issue_calls(provider, session, *calls: tuple[str, str]) -> None:
    """Deliver one ``tool_call`` server message naming (tool, call id) pairs."""
    function_calls = [SimpleNamespace(name=name, id=call_id, args={}) for name, call_id in calls]
    await provider._handle_server_response(
        session, SimpleNamespace(tool_call=SimpleNamespace(function_calls=function_calls))
    )


class TestAnInterruptedResponseEndsOnce:
    """The barge-in ends the response; the message that closes the request must not."""

    @staticmethod
    def _sc(**fields):
        return TestGeminiLiveProvider._content(**fields)

    @staticmethod
    def _open_response(model: str):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model=model)
        session = _make_session()
        provider._sessions[session.id] = mod._GeminiSessionState(
            session=session, response_started=True
        )
        ends: list[str] = []
        provider.on_response_end(lambda s: ends.append(s.id))
        return provider, session, ends

    async def test_the_idle_that_closes_an_interrupted_request_does_not_end_it_again(self):
        provider, session, ends = self._open_response("gemini-3.8-live")

        await provider._handle_server_response(session, self._sc(interaction_status="IN_PROGRESS"))
        await provider._handle_server_response(session, self._sc(interrupted=True))
        await provider._handle_server_response(
            session, self._sc(turn_complete=True, interaction_status="IDLE")
        )

        assert ends == [session.id]

    async def test_the_turn_complete_after_a_pre_3_8_interruption_does_not_either(self):
        provider, session, ends = self._open_response("gemini-2.0-flash-live-001")

        await provider._handle_server_response(session, self._sc(interrupted=True))
        await provider._handle_server_response(session, self._sc(turn_complete=True))

        assert ends == [session.id]

    async def test_a_response_that_starts_after_the_interruption_ends_normally(self):
        provider, session, ends = self._open_response("gemini-3.8-live")

        await provider._handle_server_response(session, self._sc(interrupted=True))
        await provider._handle_server_response(
            session, self._sc(turn_complete=True, interaction_status="IDLE")
        )
        await provider._handle_server_response(
            session,
            self._sc(model_turn=SimpleNamespace(parts=[]), interaction_status="IN_PROGRESS"),
        )
        await provider._handle_server_response(
            session, self._sc(turn_complete=True, interaction_status="IDLE")
        )

        assert ends == [session.id, session.id]

    async def test_an_interruption_with_no_response_open_leaves_the_closing_end_in_place(self):
        """Nothing ended at the interruption, so the closing message still hands back."""
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
        session = _make_session()
        provider._sessions[session.id] = mod._GeminiSessionState(session=session)
        ends: list[str] = []
        provider.on_response_end(lambda s: ends.append(s.id))

        await provider._handle_server_response(session, self._sc(interrupted=True))
        await provider._handle_server_response(
            session, self._sc(turn_complete=True, interaction_status="IDLE")
        )

        assert ends == [session.id]


class TestServerCancelledToolCalls:
    """``tool_call_cancellation``: the server discarded the call, so must the books."""

    @staticmethod
    def _cancellation(*ids: str):
        return SimpleNamespace(tool_call_cancellation=SimpleNamespace(ids=list(ids)))

    async def test_a_cancellation_releases_the_call_and_sends_what_it_held(self):
        provider, session, state, live = _blocking_call_state()

        await provider._handle_server_response(session, self._cancellation("call-1"))

        assert state.blocking_call_ids == set()
        assert not provider._holds_tool_call(session, "call-1")
        assert state.queued_text_injections == []
        live.send_client_content.assert_awaited()

    async def test_a_result_for_a_cancelled_call_is_dropped_not_sent(self):
        provider, session, _state, live = _blocking_call_state()
        await provider._handle_server_response(session, self._cancellation("call-1"))

        await provider.submit_tool_result(session, "call-1", '{"ok": true}')

        live.send_tool_response.assert_not_awaited()
        assert not provider._holds_tool_call(session, "call-1")

    async def test_an_empty_cancellation_changes_nothing(self):
        provider, session, state, _live = _blocking_call_state()
        told: list[tuple[str, list[str]]] = []
        provider.on_tool_call_cancelled(lambda s, ids: told.append((s.id, ids)))

        await provider._handle_server_response(session, self._cancellation())

        assert state.blocking_call_ids == {"call-1"}
        assert state.queued_text_injections == [("Queued text", "user", False)]
        assert told == []

    async def test_a_cancellation_tells_the_application_which_calls(self):
        """The handler is still working for those ids; the application must hear it."""
        provider, session, _state, _live = _blocking_call_state()
        await _issue_calls(provider, session, ("lookup", "call-2"))
        told: list[tuple[str, list[str]]] = []
        provider.on_tool_call_cancelled(lambda s, ids: told.append((s.id, ids)))

        await provider._handle_server_response(session, self._cancellation("call-1", "call-2"))

        assert told == [(session.id, ["call-1", "call-2"])]

    async def test_a_cancelled_call_is_told_then_its_late_result_is_dropped(self):
        """The flow under test: cancellation first, the result arrives after."""
        provider, session, _state, live = _blocking_call_state()
        told: list[list[str]] = []
        provider.on_tool_call_cancelled(lambda _s, ids: told.append(ids))

        await provider._handle_server_response(session, self._cancellation("call-1"))
        await provider.submit_tool_result(session, "call-1", '{"in_stock": 42}')

        assert told == [["call-1"]]
        live.send_tool_response.assert_not_awaited()
        assert not provider._holds_tool_call(session, "call-1")

    async def test_a_callback_that_raises_does_not_stop_the_release(self):
        provider, session, state, live = _blocking_call_state()

        def boom(_s, _ids):  # noqa: ANN001, ANN202
            raise RuntimeError("application bug")

        provider.on_tool_call_cancelled(boom)

        await provider._handle_server_response(session, self._cancellation("call-1"))

        assert state.blocking_call_ids == set()
        assert state.queued_text_injections == []
        live.send_client_content.assert_awaited()

    async def test_a_cancelled_background_call_leaves_the_books(self):
        provider, session, _state, live = _background_call_state()
        await _issue_calls(provider, session, ("lookup", "call-1"), ("lookup", "call-2"))

        await provider._handle_server_response(session, self._cancellation("call-1"))

        assert set(provider._open_tool_calls[session.id]) == {"call-2"}
        await provider.submit_tool_result(session, "call-1", "{}")
        live.send_tool_response.assert_not_awaited()


class TestReconnectForgetsTheOldSocketsCalls:
    """Call ids are connection-scoped: nothing on the new socket will answer them."""

    async def test_blocking_calls_are_released_and_their_injections_sent(self):
        provider, session, state, live = _blocking_call_state()

        await provider._release_calls_lost_with_the_connection(state)

        assert state.blocking_call_ids == set()
        assert provider._open_tool_calls.get(session.id, {}) == {}
        assert state.queued_text_injections == []
        live.send_client_content.assert_awaited()
        await provider.submit_tool_result(session, "call-1", "{}")
        live.send_tool_response.assert_not_awaited()

    async def test_orphaned_blocking_calls_are_reported_as_cancelled(self):
        provider, session, state, _live = _blocking_call_state()
        told: list[tuple[str, list[str]]] = []
        provider.on_tool_call_cancelled(lambda s, ids: told.append((s.id, ids)))

        await provider._release_calls_lost_with_the_connection(state)

        assert told == [(session.id, ["call-1"])]

    async def test_a_reconnect_with_nothing_outstanding_tells_nobody(self):
        provider, session, state, _live = _blocking_call_state()
        provider._forget_tool_calls(session.id)
        state.blocking_call_ids.clear()
        told: list[list[str]] = []
        provider.on_tool_call_cancelled(lambda _s, ids: told.append(ids))

        await provider._release_calls_lost_with_the_connection(state)

        assert told == []

    async def test_a_late_result_for_an_orphaned_call_is_dropped(self):
        provider, session, state, live = _blocking_call_state()
        await provider._release_calls_lost_with_the_connection(state)

        await provider.submit_tool_result(session, "call-1", "{}")

        live.send_tool_response.assert_not_awaited()

    async def test_a_result_arriving_during_the_back_off_is_dropped_not_an_error(self):
        """The socket is gone and its calls were released: a late result is stale."""
        provider, session, state, live = _blocking_call_state()
        await provider._release_calls_lost_with_the_connection(state)
        state.live_session = None

        await provider.submit_tool_result(session, "call-1", "{}")

        live.send_tool_response.assert_not_awaited()
        assert not provider._holds_tool_call(session, "call-1")

    async def test_a_result_with_no_connection_for_a_live_call_is_still_refused(self):
        provider, session, state, _live = _blocking_call_state()
        state.live_session = None

        with pytest.raises(RuntimeError):
            await provider.submit_tool_result(session, "call-1", "{}")

    async def test_the_old_sockets_calls_are_released_before_the_back_off(self):
        """The receive loop says so on the drop, not after the handshake."""
        provider, session, state, live = _blocking_call_state()
        told: list[list[str]] = []
        provider.on_tool_call_cancelled(lambda _s, ids: told.append(ids))

        async def dropped():
            raise RuntimeError("connection dropped")
            yield  # an async generator, so the loop iterates it

        live.receive = dropped
        loop_task = asyncio.create_task(provider._receive_loop(session))
        try:
            await asyncio.sleep(0.1)  # inside the first 0.5 s back-off
            assert told == [["call-1"]]
            assert not provider._holds_tool_call(session, "call-1")
            assert state.live_session is None  # no reconnect yet
            await provider.submit_tool_result(session, "call-1", "{}")
            live.send_tool_response.assert_not_awaited()
        finally:
            loop_task.cancel()
            await asyncio.gather(loop_task, return_exceptions=True)

    async def test_a_flush_that_fails_does_not_undo_the_reconnect(self):
        provider, _session, state, live = _blocking_call_state()
        live.send_client_content.side_effect = RuntimeError("socket gone")

        await provider._release_calls_lost_with_the_connection(state)

        assert state.blocking_call_ids == set()

    async def test_a_background_call_is_reported_and_its_late_result_dropped(self):
        """NON_BLOCKING is the 3.8 default: the reconnect owes it the same release."""
        provider, session, state, live = _background_call_state()
        told: list[tuple[str, list[str]]] = []
        provider.on_tool_call_cancelled(lambda s, ids: told.append((s.id, ids)))
        await _issue_calls(provider, session, ("lookup", "call-1"))
        assert provider._holds_tool_call(session, "call-1")
        assert state.blocking_call_ids == set()

        await provider._release_calls_lost_with_the_connection(state)
        await provider.submit_tool_result(session, "call-1", '{"in_stock": 42}')

        assert told == [(session.id, ["call-1"])]
        live.send_tool_response.assert_not_awaited()
        assert provider._open_tool_calls.get(session.id, {}) == {}

    async def test_blocking_and_background_calls_are_released_together(self):
        provider, session, state, live = _background_call_state()
        state.blocking_tool_names.add("charge")
        told: list[list[str]] = []
        provider.on_tool_call_cancelled(lambda _s, ids: told.append(ids))
        await _issue_calls(provider, session, ("charge", "call-1"), ("lookup", "call-2"))
        assert state.blocking_call_ids == {"call-1"}
        state.queued_text_injections.append(("Queued text", "user", False))

        await provider._release_calls_lost_with_the_connection(state)

        assert told == [["call-1", "call-2"]]
        assert provider._open_tool_calls.get(session.id, {}) == {}
        assert state.blocking_call_ids == set()
        assert state.queued_text_injections == []
        live.send_client_content.assert_awaited()
        for call_id in ("call-1", "call-2"):
            await provider.submit_tool_result(session, call_id, "{}")
        live.send_tool_response.assert_not_awaited()

    async def test_a_call_answered_before_the_reconnect_is_not_reported(self):
        provider, session, state, _live = _background_call_state()
        told: list[list[str]] = []
        provider.on_tool_call_cancelled(lambda _s, ids: told.append(ids))
        await _issue_calls(provider, session, ("lookup", "call-1"))
        await provider.submit_tool_result(session, "call-1", "{}")
        assert not provider._holds_tool_call(session, "call-1")

        await provider._release_calls_lost_with_the_connection(state)

        assert told == []
        assert provider._open_tool_calls.get(session.id, {}) == {}

    async def test_an_id_the_new_socket_reissues_is_answered_again(self):
        """Ids are connection-scoped: the new socket may hand out one the old socket lost."""
        provider, session, state, live = _background_call_state()
        await _issue_calls(provider, session, ("lookup", "call-1"))
        await provider._release_calls_lost_with_the_connection(state)
        assert not provider._holds_tool_call(session, "call-1")

        await _issue_calls(provider, session, ("lookup", "call-1"))
        await provider.submit_tool_result(session, "call-1", "{}")

        live.send_tool_response.assert_awaited_once()
        assert not provider._holds_tool_call(session, "call-1")
