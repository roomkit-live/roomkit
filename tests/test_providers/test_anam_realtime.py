"""Tests for AnamRealtimeProvider."""

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
    """Import the provider module with anam, av, numpy mocked."""
    fake_anam = SimpleNamespace(
        AnamClient=lambda **kw: SimpleNamespace(connect=lambda: None),
        PersonaConfig=lambda **kw: SimpleNamespace(**kw),
    )
    fake_av = SimpleNamespace()
    fake_np = SimpleNamespace(
        int16=None,
        float32=None,
        frombuffer=lambda *a, **kw: SimpleNamespace(astype=lambda dt: b""),
    )

    mods = {
        "anam": fake_anam,
        "av": fake_av,
        "numpy": fake_np,
    }
    with patch.dict(sys.modules, mods):
        import roomkit.providers.anam.realtime as mod

        importlib.reload(mod)
        # Reset the lazy-loaded globals so _ensure_deps succeeds
        mod._anam_mod = fake_anam
        mod._np = fake_np
        return mod


class TestAnamRealtimeProvider:
    def test_constructor_and_name(self):
        mod = _load_provider()
        from roomkit.providers.anam.config import AnamConfig

        config = AnamConfig(api_key="ak-test", persona_id="persona-1")
        provider = mod.AnamRealtimeProvider(config)
        assert provider.name == "AnamRealtimeProvider"

    def test_callback_registration(self):
        mod = _load_provider()
        from roomkit.providers.anam.config import AnamConfig

        config = AnamConfig(api_key="ak-test")
        provider = mod.AnamRealtimeProvider(config)

        audio_cb = lambda session, audio: None  # noqa: E731
        video_cb = lambda session, frame: None  # noqa: E731
        transcription_cb = lambda session, text, role, final: None  # noqa: E731
        speech_start_cb = lambda session: None  # noqa: E731
        speech_end_cb = lambda session: None  # noqa: E731
        tool_call_cb = lambda session, cid, name, args: None  # noqa: E731
        response_start_cb = lambda session: None  # noqa: E731
        response_end_cb = lambda session: None  # noqa: E731
        error_cb = lambda session, code, msg: None  # noqa: E731

        provider.on_audio(audio_cb)
        provider.on_video(video_cb)
        provider.on_transcription(transcription_cb)
        provider.on_speech_start(speech_start_cb)
        provider.on_speech_end(speech_end_cb)
        provider.on_tool_call(tool_call_cb)
        provider.on_response_start(response_start_cb)
        provider.on_response_end(response_end_cb)
        provider.on_error(error_cb)

        assert audio_cb in provider._audio_cbs
        assert video_cb in provider._video_cbs
        assert transcription_cb in provider._transcription_cbs
        assert speech_start_cb in provider._speech_start_cbs
        assert speech_end_cb in provider._speech_end_cbs
        assert tool_call_cb in provider._tool_call_cbs
        assert response_start_cb in provider._response_start_cbs
        assert response_end_cb in provider._response_end_cbs
        assert error_cb in provider._error_cbs

    async def test_close_empty_provider(self):
        mod = _load_provider()
        from roomkit.providers.anam.config import AnamConfig

        config = AnamConfig(api_key="ak-test")
        provider = mod.AnamRealtimeProvider(config)
        # Should not raise when no sessions exist
        await provider.close()

    async def test_disconnect_unknown_session_is_noop(self):
        mod = _load_provider()
        from roomkit.providers.anam.config import AnamConfig

        config = AnamConfig(api_key="ak-test")
        provider = mod.AnamRealtimeProvider(config)
        session = _make_session("unknown")
        # disconnect on unknown session returns without error
        await provider.disconnect(session)
