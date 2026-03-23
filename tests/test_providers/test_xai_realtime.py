"""Tests for XAIRealtimeProvider."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from roomkit.voice.base import VoiceSession


def _make_session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(
        id=sid,
        room_id="room1",
        participant_id="p1",
        channel_id="ch1",
    )


def _load_provider():
    """Import the provider module with websockets mocked.

    We avoid ``importlib.reload()`` to prevent class-identity issues
    with other test files that import XAIRealtimeProvider at module level.
    Instead we patch websockets into sys.modules and do a fresh import
    from a clean state.
    """
    fake_ws = SimpleNamespace(
        connect=lambda *a, **kw: None,
    )
    mod_key = "roomkit.providers.xai.realtime"
    saved = sys.modules.pop(mod_key, None)
    try:
        with patch.dict(sys.modules, {"websockets": fake_ws}):
            import roomkit.providers.xai.realtime as mod

            return mod
    finally:
        # Restore original module so other tests keep their class identity
        if saved is not None:
            sys.modules[mod_key] = saved


class TestXAIRealtimeProvider:
    def test_constructor_with_config_and_name(self):
        mod = _load_provider()
        from roomkit.providers.xai.config import XAIRealtimeConfig

        config = XAIRealtimeConfig(api_key="xai-test")
        provider = mod.XAIRealtimeProvider(config)
        assert provider.name == "XAIRealtimeProvider"

    def test_constructor_with_api_key(self):
        mod = _load_provider()
        provider = mod.XAIRealtimeProvider(api_key="xai-test")
        assert provider._model == "grok-2-audio"

    def test_constructor_with_custom_model(self):
        mod = _load_provider()
        provider = mod.XAIRealtimeProvider(
            api_key="xai-test",
            model="grok-3-audio",
            base_url="wss://custom.x.ai/v1/realtime",
        )
        assert provider._model == "grok-3-audio"
        assert provider._config.base_url == "wss://custom.x.ai/v1/realtime"

    def test_constructor_requires_api_key_or_config(self):
        mod = _load_provider()
        with pytest.raises(ValueError, match="Either config or api_key"):
            mod.XAIRealtimeProvider()

    def test_is_responding_default(self):
        mod = _load_provider()
        provider = mod.XAIRealtimeProvider(api_key="xai-test")
        assert provider.is_responding("unknown-session") is False

    def test_callback_registration(self):
        mod = _load_provider()
        provider = mod.XAIRealtimeProvider(api_key="xai-test")

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
        provider = mod.XAIRealtimeProvider(api_key="xai-test")
        session = _make_session("unknown")
        await provider.disconnect(session)

    async def test_close_empty_provider(self):
        mod = _load_provider()
        provider = mod.XAIRealtimeProvider(api_key="xai-test")
        await provider.close()
