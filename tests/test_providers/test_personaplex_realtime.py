"""Tests for PersonaPlexRealtimeProvider."""

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
    """Import the provider module with websockets, sphn, numpy mocked."""
    fake_ws = SimpleNamespace(
        connect=lambda *a, **kw: None,
    )
    fake_sphn = SimpleNamespace(
        OpusStreamWriter=lambda rate: SimpleNamespace(
            append_pcm=lambda pcm: b"opus",
            read_bytes=lambda: b"opus",
        ),
        OpusStreamReader=lambda rate: SimpleNamespace(
            append_bytes=lambda data: None,
            read_pcm=lambda: None,
        ),
    )
    fake_np = SimpleNamespace(
        int16="int16",
        float32="float32",
        frombuffer=lambda *a, **kw: SimpleNamespace(
            astype=lambda dt: SimpleNamespace(__truediv__=lambda s, v: b"")
        ),
    )

    mods = {
        "websockets": fake_ws,
        "sphn": fake_sphn,
        "numpy": fake_np,
    }
    with patch.dict(sys.modules, mods):
        import roomkit.providers.personaplex.realtime as mod

        importlib.reload(mod)
        return mod


class TestPersonaPlexRealtimeProvider:
    def test_constructor_and_name(self):
        mod = _load_provider()
        provider = mod.PersonaPlexRealtimeProvider(
            server_url="wss://localhost:8998/api/chat",
        )
        assert provider.name == "PersonaPlexRealtimeProvider"

    def test_constructor_with_custom_params(self):
        mod = _load_provider()
        provider = mod.PersonaPlexRealtimeProvider(
            server_url="wss://gpu-host:8998/api/chat",
            default_voice_prompt="NATM1.pt",
            response_end_timeout=2.0,
            seed=42,
        )
        assert provider._server_url == "wss://gpu-host:8998/api/chat"
        assert provider._default_voice_prompt == "NATM1.pt"
        assert provider._default_response_end_timeout == 2.0
        assert provider._default_seed == 42

    def test_callback_registration(self):
        mod = _load_provider()
        provider = mod.PersonaPlexRealtimeProvider()

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

        assert audio_cb in provider._audio_cbs
        assert transcription_cb in provider._transcription_cbs
        assert speech_start_cb in provider._speech_start_cbs
        assert speech_end_cb in provider._speech_end_cbs
        assert tool_call_cb in provider._tool_call_cbs
        assert response_start_cb in provider._response_start_cbs
        assert response_end_cb in provider._response_end_cbs
        assert error_cb in provider._error_cbs

    async def test_close_empty_provider(self):
        mod = _load_provider()
        provider = mod.PersonaPlexRealtimeProvider()
        await provider.close()

    async def test_disconnect_unknown_session_is_noop(self):
        mod = _load_provider()
        provider = mod.PersonaPlexRealtimeProvider()
        session = _make_session("unknown")
        # disconnect on unknown session pops None and returns
        await provider.disconnect(session)
