"""Opt-in Rime service checks. Set RIME_API_KEY to enable billable requests."""

from __future__ import annotations

import asyncio
import os

import pytest

from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime import RimeRealtimeConfig, RimeRealtimeProvider
from roomkit.voice.stt import RimeSTTConfig, RimeSTTProvider
from roomkit.voice.tts import RimeTTSConfig, RimeTTSProvider

pytestmark = pytest.mark.skipif(not os.environ.get("RIME_API_KEY"), reason="RIME_API_KEY not set")


async def test_live_tts_then_stt() -> None:
    key = os.environ["RIME_API_KEY"]
    tts = RimeTTSProvider(RimeTTSConfig(api_key=key))
    stt = RimeSTTProvider(RimeSTTConfig(api_key=key))
    try:
        audio = await tts.synthesize("The library closes at four in the afternoon.")
        assert audio.duration_seconds and audio.duration_seconds > 0
        result = await stt.transcribe(audio)
        assert result.is_final and result.language == "en"
        assert "library" in result.text.lower()
    finally:
        await asyncio.gather(tts.close(), stt.close())


async def test_live_prism_acknowledges_text_and_streams_a_response() -> None:
    rt = RimeRealtimeProvider(RimeRealtimeConfig(api_key=os.environ["RIME_API_KEY"]))
    session = VoiceSession(id="live", room_id="live", participant_id="test", channel_id="voice")
    done = asyncio.Event()
    audio, transcripts, errors = [], [], []
    rt.on_audio(lambda _, data: audio.append(data))
    rt.on_transcription(lambda _, text, role, final: transcripts.append((text, role, final)))
    rt.on_error(lambda _, code, message: errors.append(code))
    rt.on_response_end(lambda _: done.set())
    try:
        await rt.connect(session, system_prompt="Keep your response under five words.")
        result = await rt.inject_text(session, "Say hello.")
        assert result.status == "sent"
        await asyncio.wait_for(done.wait(), 30)
        assert audio and not errors
        assert any(text and role == "assistant" and final for text, role, final in transcripts)
        await rt.reconfigure(session, voice="aurora")
    finally:
        await rt.close()
