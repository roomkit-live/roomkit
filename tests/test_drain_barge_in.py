"""Tests for drain-period barge-in prevention and early streaming storage."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from roomkit import (
    HookExecution,
    HookTrigger,
    MockSTTProvider,
    MockVoiceBackend,
    RoomKit,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk, VoiceCapability
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.tts.base import TTSProvider


def _speech_events(audio: bytes = b"fake-audio-data\x00") -> list[VADEvent | None]:
    return [
        VADEvent(type=VADEventType.SPEECH_START),
        VADEvent(type=VADEventType.SPEECH_END, audio_bytes=audio),
    ]


# ---------------------------------------------------------------------------
# Fix 1: Drain-period barge-in prevention
# ---------------------------------------------------------------------------


class TestDrainBargeInPrevention:
    """After _playback_done_events is set, barge-in should NOT fire."""

    async def test_vad_speech_start_during_drain_skips_barge_in(self) -> None:
        """VAD SPEECH_START during drain period does NOT trigger barge-in."""
        caps = VoiceCapability.INTERRUPTION
        backend = MockVoiceBackend(capabilities=caps)

        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])
        pipeline = AudioPipelineConfig(vad=vad)

        channel = VoiceChannel(
            "voice-1",
            backend=backend,
            pipeline=pipeline,
            enable_barge_in=True,
            barge_in_threshold_ms=0,
        )

        kit = RoomKit(voice=backend)
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        binding = ChannelBinding(
            room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        channel.bind_session(session, room.id, binding)

        barge_in_events: list[object] = []

        @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
        async def on_barge_in(event, context):
            barge_in_events.append(event)

        # Simulate TTS playing (active playback state)
        channel._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text="Hello there",
        )
        # Mark playback as done (drain period)
        done_ev = asyncio.Event()
        done_ev.set()
        channel._playback_done_events[session.id] = done_ev

        # VAD fires SPEECH_START during drain
        await backend.simulate_audio_received(session, AudioFrame(data=b"speech"))
        await asyncio.sleep(0.15)

        # Barge-in should NOT have fired
        assert len(barge_in_events) == 0

        await kit.close()

    async def test_vad_speech_start_during_active_playback_triggers_barge_in(self) -> None:
        """Sanity check: barge-in still works during real playback (not drain)."""
        caps = VoiceCapability.INTERRUPTION
        backend = MockVoiceBackend(capabilities=caps)

        vad = MockVADProvider(events=[VADEvent(type=VADEventType.SPEECH_START)])
        pipeline = AudioPipelineConfig(vad=vad)

        channel = VoiceChannel(
            "voice-1",
            backend=backend,
            pipeline=pipeline,
            enable_barge_in=True,
            barge_in_threshold_ms=0,
        )

        kit = RoomKit(voice=backend)
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        binding = ChannelBinding(
            room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        channel.bind_session(session, room.id, binding)

        barge_in_events: list[object] = []

        @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
        async def on_barge_in(event, context):
            barge_in_events.append(event)

        # Simulate active TTS playback (done event NOT set)
        channel._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text="Hello there",
        )
        done_ev = asyncio.Event()
        # NOT set — real playback
        channel._playback_done_events[session.id] = done_ev

        # VAD fires SPEECH_START
        await backend.simulate_audio_received(session, AudioFrame(data=b"speech"))
        await asyncio.sleep(0.15)

        # Barge-in SHOULD fire
        assert len(barge_in_events) == 1

        await kit.close()

    async def test_backend_barge_in_during_drain_skips(self) -> None:
        """Backend-detected barge-in during drain period does NOT fire."""
        caps = VoiceCapability.INTERRUPTION | VoiceCapability.BARGE_IN
        backend = MockVoiceBackend(capabilities=caps)
        channel = VoiceChannel("voice-1", backend=backend, enable_barge_in=True)

        kit = RoomKit(voice=backend)
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        binding = ChannelBinding(
            room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        channel.bind_session(session, room.id, binding)

        barge_in_events: list[object] = []

        @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
        async def on_barge_in(event, context):
            barge_in_events.append(event)

        # Playing + drain
        channel._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id, text="Some text"
        )
        done_ev = asyncio.Event()
        done_ev.set()
        channel._playback_done_events[session.id] = done_ev

        # Backend fires barge-in
        channel._on_backend_barge_in(session)
        await asyncio.sleep(0.1)

        assert len(barge_in_events) == 0

        await kit.close()


# ---------------------------------------------------------------------------
# Fix 2: Early streaming response storage
# ---------------------------------------------------------------------------


class _SlowTTS(TTSProvider):
    """TTS provider that delays audio delivery to simulate slow playback."""

    def __init__(self, delay: float = 0.5) -> None:
        self._delay = delay
        self.called = False

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize(self, text: str, *, voice: str | None = None) -> object:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(data=b"audio", sample_rate=16000, is_final=True)

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        # Consume the text stream first (this is fast)
        chunks = []
        async for text in text_stream:
            chunks.append(text)
        self.called = True
        # Simulate slow audio delivery
        await asyncio.sleep(self._delay)
        yield AudioChunk(data=b"slow-audio", sample_rate=16000, is_final=True)


class _StreamingAIProvider(AIProvider):
    """Mock AI that produces a streaming response."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens

    @property
    def model_name(self) -> str:
        return "mock-streaming"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        return AIResponse(content="".join(self._tokens))

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        for t in self._tokens:
            yield t


class TestEarlyStreamingStorage:
    """Response should be in store before deliver_stream returns."""

    async def test_response_stored_before_audio_playback_ends(self) -> None:
        """Response event in store as soon as text stream exhausts, before TTS finishes."""
        ai = _StreamingAIProvider(["Hello ", "from ", "AI."])
        tts = _SlowTTS(delay=1.0)
        stt = MockSTTProvider(transcripts=["User message"])
        backend = MockVoiceBackend()
        vad = MockVADProvider(events=_speech_events())
        pipeline = AudioPipelineConfig(vad=vad)

        voice_ch = VoiceChannel("voice-1", stt=stt, tts=tts, backend=backend, pipeline=pipeline)
        ai_ch = AIChannel("ai-1", provider=ai, system_prompt="Test")

        kit = RoomKit(stt=stt, voice=backend)
        kit.register_channel(voice_ch)
        kit.register_channel(ai_ch)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")
        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        # Trigger speech flow
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x02\x00"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x03\x00"))

        # Wait long enough for text to stream but NOT for TTS audio delay
        # Text stream should exhaust quickly, TTS takes 1s
        await asyncio.sleep(0.5)

        # Check store — response should already be there
        events = await kit._store.list_events(room.id, offset=0, limit=50)
        texts = [e.content.body for e in events if hasattr(e.content, "body")]
        assert "Hello from AI." in texts, (
            f"Expected 'Hello from AI.' in store before TTS finishes, got: {texts}"
        )

        # Wait for full completion
        await asyncio.sleep(1.5)
        await kit.close()
