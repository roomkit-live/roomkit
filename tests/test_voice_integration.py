"""Integration tests for voice pipeline (audio -> STT -> AI -> TTS -> audio)."""

import asyncio

from roomkit import (
    HookExecution,
    HookTrigger,
    MockAIProvider,
    MockSTTProvider,
    MockTTSProvider,
    MockVoiceBackend,
    RoomKit,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.event import TextContent
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSessionState
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType


def _speech_events(audio: bytes = b"fake-audio-data\x00") -> list[VADEvent | None]:
    """Create a standard speech start + speech end VAD event sequence."""
    return [
        VADEvent(type=VADEventType.SPEECH_START),
        VADEvent(type=VADEventType.SPEECH_END, audio_bytes=audio),
    ]


class TestVoicePipelineIntegration:
    """Test the complete voice pipeline: audio -> STT -> AI -> TTS -> audio."""

    async def test_full_voice_pipeline(self) -> None:
        """Test: speech -> transcription -> AI response -> TTS -> audio output."""
        stt = MockSTTProvider(transcripts=["Hello, how are you?"])
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        ai = MockAIProvider(responses=["I'm doing great, thanks for asking!"])

        vad = MockVADProvider(events=_speech_events())
        pipeline = AudioPipelineConfig(vad=vad)

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel(
            "voice-1", stt=stt, tts=tts, backend=backend, pipeline=pipeline
        )
        ai_channel = AIChannel("ai-1", provider=ai, system_prompt="Be helpful")

        kit.register_channel(voice_channel)
        kit.register_channel(ai_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        assert session.state == VoiceSessionState.ACTIVE

        # Simulate two audio frames -> VAD fires SPEECH_START then SPEECH_END
        await backend.simulate_audio_received(session, AudioFrame(data=b"frame-10"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"frame-20"))

        # Give the async pipeline time to process
        await asyncio.sleep(0.15)

        # Verify STT was called
        assert len(stt.calls) == 1

        # Verify AI was called
        assert len(ai.calls) == 1

        # Verify TTS was called with AI response
        assert len(tts.calls) >= 1
        assert "great" in tts.calls[0]["text"].lower()

        # Verify audio was sent to the session
        assert len(backend.sent_audio) >= 1

        await kit.close()

    async def test_voice_hooks_fire_in_order(self) -> None:
        """Test that voice hooks fire in the correct order."""
        stt = MockSTTProvider(transcripts=["Test message"])
        tts = MockTTSProvider()
        backend = MockVoiceBackend()

        # Three frames: SPEECH_START, then nothing, then SPEECH_END with audio
        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_START),
                None,
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio\x00"),
            ]
        )
        pipeline = AudioPipelineConfig(vad=vad)

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel(
            "voice-1", stt=stt, tts=tts, backend=backend, pipeline=pipeline
        )
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        hook_order: list[str] = []

        @kit.hook(HookTrigger.ON_SPEECH_START, HookExecution.ASYNC)
        async def on_speech_start(event, context):
            hook_order.append("ON_SPEECH_START")

        @kit.hook(HookTrigger.ON_SPEECH_END, HookExecution.ASYNC)
        async def on_speech_end(event, context):
            hook_order.append("ON_SPEECH_END")

        @kit.hook(HookTrigger.ON_TRANSCRIPTION, HookExecution.SYNC)
        async def on_transcription(event, context):
            from roomkit.models.hook import HookResult

            hook_order.append("ON_TRANSCRIPTION")
            return HookResult.allow()

        @kit.hook(HookTrigger.BEFORE_BROADCAST, HookExecution.SYNC)
        async def before_broadcast(event, context):
            from roomkit.models.hook import HookResult

            hook_order.append("BEFORE_BROADCAST")
            return HookResult.allow()

        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        # Simulate 3 audio frames (matching the 3 VAD events)
        for i in range(3):
            await backend.simulate_audio_received(session, AudioFrame(data=f"fr-{i:03d}".encode()))

        await asyncio.sleep(0.15)

        assert "ON_SPEECH_START" in hook_order
        assert "ON_SPEECH_END" in hook_order
        assert "ON_TRANSCRIPTION" in hook_order
        assert "BEFORE_BROADCAST" in hook_order

        # Verify order: speech_start before speech_end
        assert hook_order.index("ON_SPEECH_START") < hook_order.index("ON_SPEECH_END")
        # Verify order: speech_end before transcription
        assert hook_order.index("ON_SPEECH_END") < hook_order.index("ON_TRANSCRIPTION")
        # Verify order: transcription before broadcast
        assert hook_order.index("ON_TRANSCRIPTION") < hook_order.index("BEFORE_BROADCAST")

        await kit.close()

    async def test_on_transcription_hook_can_modify_text(self) -> None:
        """Test that ON_TRANSCRIPTION hook can observe the transcribed text."""
        stt = MockSTTProvider(transcripts=["hello world"])
        backend = MockVoiceBackend()

        vad = MockVADProvider(events=_speech_events(b"audio\x00"))
        pipeline = AudioPipelineConfig(vad=vad)

        kit = RoomKit(stt=stt, voice=backend)

        voice_channel = VoiceChannel("voice-1", stt=stt, backend=backend, pipeline=pipeline)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        received_messages: list[str] = []

        @kit.hook(HookTrigger.ON_TRANSCRIPTION, HookExecution.SYNC)
        async def modify_transcription(event, context):
            from roomkit.models.hook import HookResult

            return HookResult.allow()

        @kit.hook(HookTrigger.BEFORE_BROADCAST, HookExecution.SYNC)
        async def capture_message(event, context):
            from roomkit.models.hook import HookResult as HR

            if isinstance(event.content, TextContent):
                received_messages.append(event.content.body)
            return HR.allow()

        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        # Two frames for SPEECH_START + SPEECH_END
        await backend.simulate_audio_received(session, AudioFrame(data=b"f1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"f2"))
        await asyncio.sleep(0.15)

        assert len(received_messages) == 1
        assert received_messages[0] == "hello world"

        await kit.close()

    async def test_on_transcription_hook_can_block(self) -> None:
        """Test that ON_TRANSCRIPTION hook can block the message."""
        stt = MockSTTProvider(transcripts=["blocked message"])
        backend = MockVoiceBackend()

        vad = MockVADProvider(events=_speech_events(b"audio\x00"))
        pipeline = AudioPipelineConfig(vad=vad)

        kit = RoomKit(stt=stt, voice=backend)

        voice_channel = VoiceChannel("voice-1", stt=stt, backend=backend, pipeline=pipeline)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        broadcast_called: list[bool] = []

        @kit.hook(HookTrigger.ON_TRANSCRIPTION, HookExecution.SYNC)
        async def block_transcription(event, context):
            from roomkit.models.hook import HookResult

            return HookResult.block(reason="profanity_filter")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, HookExecution.SYNC)
        async def track_broadcast(event, context):
            from roomkit.models.hook import HookResult

            broadcast_called.append(True)
            return HookResult.allow()

        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"f1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"f2"))
        await asyncio.sleep(0.15)

        # Broadcast should not be called because transcription was blocked
        assert len(broadcast_called) == 0

        await kit.close()

    async def test_before_tts_hook_fires(self) -> None:
        """Test that BEFORE_TTS hook fires before synthesis."""
        stt = MockSTTProvider(transcripts=["hello"])
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        ai = MockAIProvider(responses=["Original response"])

        vad = MockVADProvider(events=_speech_events(b"audio\x00"))
        pipeline = AudioPipelineConfig(vad=vad)

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel(
            "voice-1", stt=stt, tts=tts, backend=backend, pipeline=pipeline
        )
        ai_channel = AIChannel("ai-1", provider=ai)
        kit.register_channel(voice_channel)
        kit.register_channel(ai_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")

        before_tts_texts: list[str] = []

        @kit.hook(HookTrigger.BEFORE_TTS, HookExecution.SYNC)
        async def observe_tts(event, context):
            from roomkit.models.hook import HookResult

            before_tts_texts.append(event)
            return HookResult.allow()

        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"f1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"f2"))
        await asyncio.sleep(0.2)

        assert len(before_tts_texts) >= 1
        assert "Original response" in before_tts_texts[0]

        await kit.close()

    async def test_before_tts_hook_can_block(self) -> None:
        """Test that BEFORE_TTS hook can prevent audio synthesis."""
        stt = MockSTTProvider(transcripts=["hello"])
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        ai = MockAIProvider(responses=["This should not be spoken"])

        vad = MockVADProvider(events=_speech_events(b"audio\x00"))
        pipeline = AudioPipelineConfig(vad=vad)

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel(
            "voice-1", stt=stt, tts=tts, backend=backend, pipeline=pipeline
        )
        ai_channel = AIChannel("ai-1", provider=ai)
        kit.register_channel(voice_channel)
        kit.register_channel(ai_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")

        @kit.hook(HookTrigger.BEFORE_TTS, HookExecution.SYNC)
        async def block_tts(event, context):
            from roomkit.models.hook import HookResult

            return HookResult.block(reason="silent_mode")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"f1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"f2"))
        await asyncio.sleep(0.2)

        # No audio should be sent
        assert len(backend.sent_audio) == 0

        await kit.close()

    async def test_after_tts_hook_fires(self) -> None:
        """Test that AFTER_TTS hook fires after synthesis."""
        stt = MockSTTProvider(transcripts=["hello"])
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        ai = MockAIProvider(responses=["Response"])

        vad = MockVADProvider(events=_speech_events(b"audio\x00"))
        pipeline = AudioPipelineConfig(vad=vad)

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel(
            "voice-1", stt=stt, tts=tts, backend=backend, pipeline=pipeline
        )
        ai_channel = AIChannel("ai-1", provider=ai)
        kit.register_channel(voice_channel)
        kit.register_channel(ai_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")

        after_tts_events: list[object] = []

        @kit.hook(HookTrigger.AFTER_TTS, HookExecution.ASYNC)
        async def after_tts(event, context):
            after_tts_events.append(event)

        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        await backend.simulate_audio_received(session, AudioFrame(data=b"f1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"f2"))
        await asyncio.sleep(0.2)

        assert len(after_tts_events) >= 1

        await kit.close()

    async def test_multiple_sessions_same_room(self) -> None:
        """Test multiple voice sessions in the same room."""
        stt = MockSTTProvider(transcripts=["From user 1", "From user 2"])
        backend = MockVoiceBackend()

        # 4 events: 2 per session (SPEECH_START + SPEECH_END)
        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_START),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1"),
                VADEvent(type=VADEventType.SPEECH_START),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio2"),
            ]
        )
        pipeline = AudioPipelineConfig(vad=vad)

        kit = RoomKit(stt=stt, voice=backend)

        voice_channel = VoiceChannel("voice-1", stt=stt, backend=backend, pipeline=pipeline)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        messages: list[tuple[str, str]] = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST, HookExecution.SYNC)
        async def capture(event, context):
            from roomkit.models.hook import HookResult

            if isinstance(event.content, TextContent):
                messages.append((event.source.participant_id, event.content.body))
            return HookResult.allow()

        session1 = await kit.connect_voice(room.id, "user-1", "voice-1")
        session2 = await kit.connect_voice(room.id, "user-2", "voice-1")

        # Session 1 speaks (2 frames -> SPEECH_START + SPEECH_END)
        await backend.simulate_audio_received(session1, AudioFrame(data=b"f1"))
        await backend.simulate_audio_received(session1, AudioFrame(data=b"f2"))
        await asyncio.sleep(0.15)

        # Session 2 speaks (2 frames -> SPEECH_START + SPEECH_END)
        await backend.simulate_audio_received(session2, AudioFrame(data=b"f3"))
        await backend.simulate_audio_received(session2, AudioFrame(data=b"f4"))
        await asyncio.sleep(0.15)

        # Both messages should be captured with correct participant IDs
        assert len(messages) == 2
        participants = [m[0] for m in messages]
        assert "user-1" in participants
        assert "user-2" in participants

        await kit.close()

    async def test_session_metadata_preserved(self) -> None:
        """Test that session metadata is passed through connect."""
        backend = MockVoiceBackend()
        kit = RoomKit(voice=backend)

        voice_channel = VoiceChannel("voice-1", backend=backend)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(
            room.id, "user-1", "voice-1", metadata={"language": "en", "device": "mobile"}
        )

        assert session.metadata["language"] == "en"
        assert session.metadata["device"] == "mobile"

        await kit.close()

    async def test_voice_channel_close_cleans_up(self) -> None:
        """Test that closing voice channel cleans up resources."""
        stt = MockSTTProvider()
        tts = MockTTSProvider()
        backend = MockVoiceBackend()

        vad = MockVADProvider()
        pipeline = AudioPipelineConfig(vad=vad)

        voice_channel = VoiceChannel(
            "voice-1", stt=stt, tts=tts, backend=backend, pipeline=pipeline
        )

        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.base import VoiceSession

        session = VoiceSession(
            id="test-session",
            room_id="room-1",
            participant_id="user-1",
            channel_id="voice-1",
        )
        binding = ChannelBinding(
            room_id="room-1",
            channel_id="voice-1",
            channel_type=ChannelType.VOICE,
        )
        voice_channel.bind_session(session, "room-1", binding)

        await voice_channel.close()

        assert len(voice_channel._session_bindings) == 0
        assert vad.closed


class TestVoiceHookTriggers:
    """Test that all voice hook triggers are available."""

    def test_voice_hooks_in_enum(self) -> None:
        """Verify all voice hooks are defined in HookTrigger enum."""
        assert HookTrigger.ON_SPEECH_START == "on_speech_start"
        assert HookTrigger.ON_SPEECH_END == "on_speech_end"
        assert HookTrigger.ON_TRANSCRIPTION == "on_transcription"
        assert HookTrigger.BEFORE_TTS == "before_tts"
        assert HookTrigger.AFTER_TTS == "after_tts"

    def test_enhanced_voice_hooks_in_enum(self) -> None:
        """Verify enhanced voice hooks are defined."""
        assert HookTrigger.ON_BARGE_IN == "on_barge_in"
        assert HookTrigger.ON_TTS_CANCELLED == "on_tts_cancelled"
        assert HookTrigger.ON_PARTIAL_TRANSCRIPTION == "on_partial_transcription"
        assert HookTrigger.ON_VAD_SILENCE == "on_vad_silence"
        assert HookTrigger.ON_VAD_AUDIO_LEVEL == "on_vad_audio_level"
        assert HookTrigger.ON_SPEAKER_CHANGE == "on_speaker_change"


class TestPipelineVADHooksIntegration:
    """Integration tests for VAD events flowing through the pipeline to hooks."""

    async def test_vad_silence_hook_fires(self) -> None:
        """ON_VAD_SILENCE hook fires when pipeline detects silence."""
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.events import VADSilenceEvent

        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SILENCE, duration_ms=500),
                VADEvent(type=VADEventType.SILENCE, duration_ms=1000),
            ]
        )
        pipeline = AudioPipelineConfig(vad=vad)
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend, pipeline=pipeline)

        kit = RoomKit(voice=backend)
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        binding = ChannelBinding(
            room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        channel.bind_session(session, room.id, binding)

        silence_events: list[object] = []

        @kit.hook(HookTrigger.ON_VAD_SILENCE, HookExecution.ASYNC)
        async def on_silence(event, context):
            silence_events.append(event)

        # Two frames -> two SILENCE events
        await backend.simulate_audio_received(session, AudioFrame(data=b"f1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"f2"))
        await asyncio.sleep(0.1)

        assert len(silence_events) == 2
        assert all(isinstance(e, VADSilenceEvent) for e in silence_events)
        assert silence_events[0].silence_duration_ms == 500
        assert silence_events[1].silence_duration_ms == 1000

        await kit.close()

    async def test_vad_audio_level_hook_fires(self) -> None:
        """ON_VAD_AUDIO_LEVEL hook fires with audio level updates."""
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.events import VADAudioLevelEvent

        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.AUDIO_LEVEL, level_db=-30.0, confidence=0.2),
                VADEvent(type=VADEventType.AUDIO_LEVEL, level_db=-15.0, confidence=0.8),
            ]
        )
        pipeline = AudioPipelineConfig(vad=vad)
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend, pipeline=pipeline)

        kit = RoomKit(voice=backend)
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        binding = ChannelBinding(
            room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        channel.bind_session(session, room.id, binding)

        levels: list[object] = []

        @kit.hook(HookTrigger.ON_VAD_AUDIO_LEVEL, HookExecution.ASYNC)
        async def on_level(event, context):
            levels.append(event)

        await backend.simulate_audio_received(session, AudioFrame(data=b"f1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"f2"))
        await asyncio.sleep(0.1)

        assert len(levels) == 2
        assert all(isinstance(e, VADAudioLevelEvent) for e in levels)
        assert levels[0].level_db == -30.0
        assert levels[0].is_speech is False
        assert levels[1].level_db == -15.0
        assert levels[1].is_speech is True

        await kit.close()

    async def test_speaker_change_hook_fires(self) -> None:
        """ON_SPEAKER_CHANGE hook fires when diarization detects a new speaker."""
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.events import SpeakerChangeEvent
        from roomkit.voice.pipeline.diarization.base import DiarizationResult
        from roomkit.voice.pipeline.diarization.mock import MockDiarizationProvider

        diarizer = MockDiarizationProvider(
            results=[
                DiarizationResult(speaker_id="speaker_0", confidence=0.9, is_new_speaker=True),
                DiarizationResult(speaker_id="speaker_1", confidence=0.85, is_new_speaker=True),
            ]
        )
        vad = MockVADProvider()
        pipeline = AudioPipelineConfig(vad=vad, diarization=diarizer)
        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend, pipeline=pipeline)

        kit = RoomKit(voice=backend)
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        binding = ChannelBinding(
            room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        channel.bind_session(session, room.id, binding)

        speaker_events: list[object] = []

        @kit.hook(HookTrigger.ON_SPEAKER_CHANGE, HookExecution.ASYNC)
        async def on_speaker(event, context):
            speaker_events.append(event)

        await backend.simulate_audio_received(session, AudioFrame(data=b"f1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"f2"))
        await asyncio.sleep(0.1)

        assert len(speaker_events) == 2
        assert all(isinstance(e, SpeakerChangeEvent) for e in speaker_events)
        assert speaker_events[0].speaker_id == "speaker_0"
        assert speaker_events[1].speaker_id == "speaker_1"

        await kit.close()


class TestBargeInIntegration:
    """Integration tests for barge-in detection and handling."""

    async def test_barge_in_detected_during_tts(self) -> None:
        """Barge-in is detected when pipeline fires SPEECH_START during TTS playback."""
        from roomkit.channels.voice import TTSPlaybackState
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.base import VoiceCapability
        from roomkit.voice.events import BargeInEvent, TTSCancelledEvent

        caps = VoiceCapability.INTERRUPTION | VoiceCapability.BARGE_IN
        stt = MockSTTProvider(transcripts=["Interrupt!"])
        tts = MockTTSProvider()
        backend = MockVoiceBackend(capabilities=caps)

        # VAD will fire SPEECH_START (which triggers barge-in check)
        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_START),
            ]
        )
        pipeline = AudioPipelineConfig(vad=vad)

        channel = VoiceChannel(
            "voice-1",
            stt=stt,
            tts=tts,
            backend=backend,
            pipeline=pipeline,
            enable_barge_in=True,
            barge_in_threshold_ms=50,
        )

        kit = RoomKit(stt=stt, tts=tts, voice=backend)
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        binding = ChannelBinding(
            room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        channel.bind_session(session, room.id, binding)

        barge_in_events: list[object] = []
        cancelled_events: list[object] = []

        @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
        async def on_barge_in(event, context):
            barge_in_events.append(event)

        @kit.hook(HookTrigger.ON_TTS_CANCELLED, HookExecution.ASYNC)
        async def on_cancelled(event, context):
            cancelled_events.append(event)

        # Simulate TTS playing
        channel._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text="Hello, how can I help you today?",
        )

        # Wait for threshold
        await asyncio.sleep(0.1)

        # Pipeline fires SPEECH_START -> triggers barge-in
        await backend.simulate_audio_received(session, AudioFrame(data=b"speech"))
        await asyncio.sleep(0.1)

        assert len(barge_in_events) == 1
        assert isinstance(barge_in_events[0], BargeInEvent)
        assert barge_in_events[0].interrupted_text == "Hello, how can I help you today?"
        assert barge_in_events[0].audio_position_ms > 0

        assert len(cancelled_events) == 1
        assert isinstance(cancelled_events[0], TTSCancelledEvent)
        assert cancelled_events[0].reason == "barge_in"

        await kit.close()

    async def test_barge_in_not_triggered_below_threshold(self) -> None:
        """Barge-in is NOT triggered if TTS just started (below threshold)."""
        from datetime import UTC, datetime

        from roomkit.channels.voice import TTSPlaybackState
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.base import VoiceCapability

        caps = VoiceCapability.INTERRUPTION
        backend = MockVoiceBackend(capabilities=caps)

        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_START),
            ]
        )
        pipeline = AudioPipelineConfig(vad=vad)

        channel = VoiceChannel(
            "voice-1",
            backend=backend,
            pipeline=pipeline,
            enable_barge_in=True,
            barge_in_threshold_ms=500,  # High threshold
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

        # Set TTS as just started (below threshold)
        channel._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text="Quick message",
            started_at=datetime.now(UTC),
        )

        # User speaks immediately (below 500ms threshold)
        await backend.simulate_audio_received(session, AudioFrame(data=b"speech"))
        await asyncio.sleep(0.1)

        assert len(barge_in_events) == 0

        await kit.close()

    async def test_barge_in_disabled(self) -> None:
        """Barge-in can be disabled via configuration."""
        from roomkit.channels.voice import TTSPlaybackState
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.base import VoiceCapability

        caps = VoiceCapability.INTERRUPTION
        backend = MockVoiceBackend(capabilities=caps)

        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_START),
            ]
        )
        pipeline = AudioPipelineConfig(vad=vad)

        channel = VoiceChannel(
            "voice-1",
            backend=backend,
            pipeline=pipeline,
            enable_barge_in=False,  # Disabled
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

        channel._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text="Playing audio",
        )

        await asyncio.sleep(0.1)

        # Pipeline fires SPEECH_START but barge-in is disabled
        await backend.simulate_audio_received(session, AudioFrame(data=b"speech"))
        await asyncio.sleep(0.1)

        assert len(barge_in_events) == 0

        await kit.close()

    async def test_interrupt_method_fires_cancelled_hook(self) -> None:
        """VoiceChannel.interrupt() fires ON_TTS_CANCELLED hook."""
        from roomkit.channels.voice import TTSPlaybackState
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.base import VoiceCapability
        from roomkit.voice.events import TTSCancelledEvent

        caps = VoiceCapability.INTERRUPTION
        backend = MockVoiceBackend(capabilities=caps)
        channel = VoiceChannel("voice-1", backend=backend)

        kit = RoomKit(voice=backend)
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        binding = ChannelBinding(
            room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        channel.bind_session(session, room.id, binding)

        cancelled_events: list[object] = []

        @kit.hook(HookTrigger.ON_TTS_CANCELLED, HookExecution.ASYNC)
        async def on_cancelled(event, context):
            cancelled_events.append(event)

        channel._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text="Message being played",
        )

        result = await channel.interrupt(session, reason="explicit")

        assert result is True
        await asyncio.sleep(0.1)

        assert len(cancelled_events) == 1
        assert isinstance(cancelled_events[0], TTSCancelledEvent)
        assert cancelled_events[0].reason == "explicit"

        await kit.close()

    async def test_interrupt_returns_false_when_nothing_playing(self) -> None:
        """VoiceChannel.interrupt() returns False if nothing is playing."""
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType

        backend = MockVoiceBackend()
        channel = VoiceChannel("voice-1", backend=backend)

        kit = RoomKit(voice=backend)
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        binding = ChannelBinding(
            room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        channel.bind_session(session, room.id, binding)

        result = await channel.interrupt(session)
        assert result is False

        await kit.close()

    async def test_backend_barge_in_callback(self) -> None:
        """Backend-detected barge-in triggers ON_BARGE_IN hook."""
        from roomkit.channels.voice import TTSPlaybackState
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.base import VoiceCapability
        from roomkit.voice.events import BargeInEvent

        caps = VoiceCapability.INTERRUPTION | VoiceCapability.BARGE_IN
        backend = MockVoiceBackend(capabilities=caps)
        channel = VoiceChannel("voice-1", backend=backend)

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

        channel._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text="Long response text",
        )

        await backend.simulate_barge_in(session)
        await asyncio.sleep(0.1)

        assert len(barge_in_events) == 1
        assert isinstance(barge_in_events[0], BargeInEvent)

        await kit.close()


class TestVoiceCapabilitiesIntegration:
    """Test VoiceCapability-based feature detection."""

    async def test_channel_checks_interruption_capability(self) -> None:
        """VoiceChannel only calls cancel_audio() when backend has INTERRUPTION capability."""
        from roomkit.channels.voice import TTSPlaybackState
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.base import VoiceCapability

        # Backend WITHOUT INTERRUPTION capability
        backend_no_int = MockVoiceBackend(capabilities=VoiceCapability.NONE)
        channel_no_int = VoiceChannel("voice-1", backend=backend_no_int)

        kit = RoomKit(voice=backend_no_int)
        kit.register_channel(channel_no_int)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        binding = ChannelBinding(
            room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        channel_no_int.bind_session(session, room.id, binding)

        channel_no_int._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text="Test",
        )

        cancel_called: list[bool] = []
        original_cancel = backend_no_int.cancel_audio

        async def track_cancel(s):
            cancel_called.append(True)
            return await original_cancel(s)

        backend_no_int.cancel_audio = track_cancel

        result = await channel_no_int.interrupt(session)
        assert result is True  # Playback was cleared
        assert len(cancel_called) == 0  # But backend.cancel_audio NOT called

        await kit.close()

        # Backend WITH INTERRUPTION capability
        backend_with_int = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
        channel_with_int = VoiceChannel("voice-2", backend=backend_with_int)

        kit2 = RoomKit(voice=backend_with_int)
        kit2.register_channel(channel_with_int)

        room2 = await kit2.create_room()
        await kit2.attach_channel(room2.id, "voice-2")

        session2 = await kit2.connect_voice(room2.id, "user-1", "voice-2")
        binding2 = ChannelBinding(
            room_id=room2.id, channel_id="voice-2", channel_type=ChannelType.VOICE
        )
        channel_with_int.bind_session(session2, room2.id, binding2)

        channel_with_int._playing_sessions[session2.id] = TTSPlaybackState(
            session_id=session2.id,
            text="Test",
        )

        cancel_called2: list[bool] = []
        original_cancel2 = backend_with_int.cancel_audio

        async def track_cancel2(s):
            cancel_called2.append(True)
            return await original_cancel2(s)

        backend_with_int.cancel_audio = track_cancel2

        result2 = await channel_with_int.interrupt(session2)
        assert result2 is True
        assert len(cancel_called2) == 1  # backend.cancel_audio WAS called

        await kit2.close()
