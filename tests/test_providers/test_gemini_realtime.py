"""Tests for GeminiLiveProvider."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

from roomkit.voice.base import VoiceSession


def _make_session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(
        id=sid,
        room_id="room1",
        participant_id="p1",
        channel_id="ch1",
    )


def _build_fake_genai():
    """Build a fake google.genai module tree."""
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

    async def test_close_empty_provider(self):
        mod = _load_provider()
        provider = mod.GeminiLiveProvider(api_key="test-key")
        await provider.close()
