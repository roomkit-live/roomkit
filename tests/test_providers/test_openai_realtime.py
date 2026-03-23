"""Tests for OpenAIRealtimeProvider."""

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


def _load_provider():
    """Import the provider module with websockets mocked."""
    fake_ws = SimpleNamespace(
        connect=lambda *a, **kw: None,
    )
    mods = {"websockets": fake_ws}
    with patch.dict(sys.modules, mods):
        import roomkit.providers.openai.realtime as mod

        importlib.reload(mod)
        return mod


class TestOpenAIRealtimeProvider:
    def test_constructor_and_name(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        assert provider.name == "OpenAIRealtimeProvider"

    def test_constructor_with_model_and_base_url(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(
            api_key="sk-test",
            model="gpt-realtime-2.0",
            base_url="wss://custom.api.com/v1/realtime",
        )
        assert provider._model == "gpt-realtime-2.0"
        assert provider._base_url == "wss://custom.api.com/v1/realtime"

    def test_is_responding_default(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        assert provider.is_responding("unknown-session") is False

    def test_callback_registration(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")

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
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        session = _make_session("unknown")
        # Should not raise
        await provider.disconnect(session)

    async def test_close_empty_provider(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        # Should not raise when no sessions exist
        await provider.close()

    def test_build_turn_detection_semantic_vad(self):
        mod = _load_provider()
        result = mod.OpenAIRealtimeProvider._build_turn_detection(
            "semantic_vad", {"eagerness": "high"}
        )
        assert result is not None
        assert result["type"] == "semantic_vad"
        assert result["eagerness"] == "high"

    def test_build_turn_detection_server_vad(self):
        mod = _load_provider()
        result = mod.OpenAIRealtimeProvider._build_turn_detection(
            "server_vad", {"threshold": 0.5, "silence_duration_ms": 500}
        )
        assert result is not None
        assert result["type"] == "server_vad"
        assert result["threshold"] == 0.5
        assert result["silence_duration_ms"] == 500

    def test_build_turn_detection_none(self):
        mod = _load_provider()
        result = mod.OpenAIRealtimeProvider._build_turn_detection(None, {})
        assert result is None
