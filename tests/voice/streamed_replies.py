"""Replies an agent streams into a voice room, for tests of what the voice says.

One voice channel, a room per reply, each with an agent that streams its reply
token by token (with a pause after each when replies must overlap).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from roomkit import RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.stt.mock import MockSTTProvider
from tests.test_voice_streaming_ai_tts import _speech_events
from tests.voice.test_voice_tts_filter import _StreamingMockTTS


class StreamedReply(AIProvider):
    """Streams *tokens*, *gap* seconds apart, and counts the ones read."""

    def __init__(self, tokens: list[str], gap: float = 0.0) -> None:
        self._tokens, self._gap = tokens, gap
        self.read = 0

    @property
    def model_name(self) -> str:
        return "streamed-reply"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        return AIResponse(content="".join(self._tokens))

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        for token in self._tokens:
            self.read += 1
            yield token
            await asyncio.sleep(self._gap)


async def voice_rooms(
    *replies: list[str], gap: float = 0.0, tts: Any = None, **voice_options: Any
) -> tuple[RoomKit, VoiceChannel, MockVoiceBackend, Any, list[VoiceSession]]:
    """A voice channel (*voice_options*, a TTS taking streamed text unless *tts*
    is given), and a room per reply whose agent ``ai-<n>`` streams it."""
    tts, backend = tts or _StreamingMockTTS(), MockVoiceBackend()
    stt = MockSTTProvider(transcripts=["A question."] * len(replies))
    vad = MockVADProvider(events=_speech_events() * len(replies))
    voice = VoiceChannel(
        "voice-1",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=AudioPipelineConfig(vad=vad),
        **voice_options,
    )
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(voice)
    sessions = []
    for n, tokens in enumerate(replies):
        kit.register_channel(AIChannel(f"ai-{n}", provider=StreamedReply(tokens, gap)))
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, f"ai-{n}")
        sessions.append(await kit.join(room.id, "voice-1", participant_id=f"user-{n}"))
    return kit, voice, backend, tts, sessions


async def ask(backend: MockVoiceBackend, session: VoiceSession) -> None:
    """Speak into *session*: the agent of its room answers."""
    for data in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        await backend.simulate_audio_received(session, AudioFrame(data=data))


async def said(tts: _StreamingMockTTS, replies: int) -> list[str]:
    """What the TTS was given of each streamed reply, once *replies* came."""
    for _ in range(100):
        if len(tts.stream_input_texts) >= replies:
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.2)
    return [" ".join(chunks) for chunks in tts.stream_input_texts]
