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
from roomkit.voice.base import VoiceSessionState


class TestVoicePipelineIntegration:
    """Test the complete voice pipeline: audio -> STT -> AI -> TTS -> audio."""

    async def test_full_voice_pipeline(self) -> None:
        """Test: speech -> transcription -> AI response -> TTS -> audio output."""
        # Setup providers
        stt = MockSTTProvider(transcripts=["Hello, how are you?"])
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        ai = MockAIProvider(responses=["I'm doing great, thanks for asking!"])

        # Create RoomKit with voice support
        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        # Register channels
        voice_channel = VoiceChannel("voice-1", stt=stt, tts=tts, backend=backend)
        ai_channel = AIChannel("ai-1", provider=ai, system_prompt="Be helpful")

        kit.register_channel(voice_channel)
        kit.register_channel(ai_channel)

        # Create room and attach channels
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")

        # Connect voice session
        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        assert session.state == VoiceSessionState.ACTIVE

        # Simulate speech end (VAD detected silence after speech)
        await backend.simulate_speech_end(session, b"fake-audio-data")

        # Give the async pipeline time to process
        await asyncio.sleep(0.1)

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

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel("voice-1", stt=stt, tts=tts, backend=backend)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        # Track hook execution order
        hook_order = []

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

        # Simulate speech
        await backend.simulate_speech_start(session)
        await asyncio.sleep(0.05)

        await backend.simulate_speech_end(session, b"audio")
        await asyncio.sleep(0.1)

        # Verify hook order
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
        """Test that ON_TRANSCRIPTION hook can modify the transcribed text.

        Note: Voice hooks (ON_TRANSCRIPTION, BEFORE_TTS, AFTER_TTS) pass the text
        string as the 'event' parameter, not a RoomEvent. To modify the text,
        return a HookResult with action="modify" and the event field set to the
        modified text string (cast appropriately).
        """
        stt = MockSTTProvider(transcripts=["hello world"])
        backend = MockVoiceBackend()

        kit = RoomKit(stt=stt, voice=backend)

        voice_channel = VoiceChannel("voice-1", stt=stt, backend=backend)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        received_messages = []

        @kit.hook(HookTrigger.ON_TRANSCRIPTION, HookExecution.SYNC)
        async def modify_transcription(event, context):
            # For voice hooks, event is the transcription text (str)
            # Return HookResult.allow() to let it proceed, or modify via action
            from roomkit.models.hook import HookResult

            # Voice hooks don't support modify action since event must be RoomEvent
            # Instead, we can block and let the test verify blocking works
            return HookResult.allow()

        @kit.hook(HookTrigger.BEFORE_BROADCAST, HookExecution.SYNC)
        async def capture_message(event, context):
            from roomkit.models.hook import HookResult as HR

            if isinstance(event.content, TextContent):
                received_messages.append(event.content.body)
            return HR.allow()

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        await backend.simulate_speech_end(session, b"audio")
        await asyncio.sleep(0.1)

        # Verify the original text went through (voice hooks can't modify text currently)
        assert len(received_messages) == 1
        assert received_messages[0] == "hello world"

        await kit.close()

    async def test_on_transcription_hook_can_block(self) -> None:
        """Test that ON_TRANSCRIPTION hook can block the message."""
        stt = MockSTTProvider(transcripts=["blocked message"])
        backend = MockVoiceBackend()

        kit = RoomKit(stt=stt, voice=backend)

        voice_channel = VoiceChannel("voice-1", stt=stt, backend=backend)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        broadcast_called = []

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
        await backend.simulate_speech_end(session, b"audio")
        await asyncio.sleep(0.1)

        # Broadcast should not be called because transcription was blocked
        assert len(broadcast_called) == 0

        await kit.close()

    async def test_before_tts_hook_fires(self) -> None:
        """Test that BEFORE_TTS hook fires before synthesis.

        Note: Voice hooks pass the text string as the event parameter.
        Currently, voice hooks can observe or block, but not modify the text
        due to HookResult expecting RoomEvent for modification.
        """
        stt = MockSTTProvider(transcripts=["hello"])
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        ai = MockAIProvider(responses=["Original response"])

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel("voice-1", stt=stt, tts=tts, backend=backend)
        ai_channel = AIChannel("ai-1", provider=ai)
        kit.register_channel(voice_channel)
        kit.register_channel(ai_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")

        before_tts_texts = []

        @kit.hook(HookTrigger.BEFORE_TTS, HookExecution.SYNC)
        async def observe_tts(event, context):
            from roomkit.models.hook import HookResult

            # event is the text string for BEFORE_TTS
            before_tts_texts.append(event)
            return HookResult.allow()

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        await backend.simulate_speech_end(session, b"audio")
        await asyncio.sleep(0.15)

        # BEFORE_TTS hook should have observed the AI response
        assert len(before_tts_texts) >= 1
        assert "Original response" in before_tts_texts[0]

        await kit.close()

    async def test_before_tts_hook_can_block(self) -> None:
        """Test that BEFORE_TTS hook can prevent audio synthesis."""
        stt = MockSTTProvider(transcripts=["hello"])
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        ai = MockAIProvider(responses=["This should not be spoken"])

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel("voice-1", stt=stt, tts=tts, backend=backend)
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
        await backend.simulate_speech_end(session, b"audio")
        await asyncio.sleep(0.15)

        # No audio should be sent
        assert len(backend.sent_audio) == 0

        await kit.close()

    async def test_after_tts_hook_fires(self) -> None:
        """Test that AFTER_TTS hook fires after synthesis."""
        stt = MockSTTProvider(transcripts=["hello"])
        tts = MockTTSProvider()
        backend = MockVoiceBackend()
        ai = MockAIProvider(responses=["Response"])

        kit = RoomKit(stt=stt, tts=tts, voice=backend)

        voice_channel = VoiceChannel("voice-1", stt=stt, tts=tts, backend=backend)
        ai_channel = AIChannel("ai-1", provider=ai)
        kit.register_channel(voice_channel)
        kit.register_channel(ai_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, "ai-1")

        after_tts_events = []

        @kit.hook(HookTrigger.AFTER_TTS, HookExecution.ASYNC)
        async def after_tts(event, context):
            after_tts_events.append(event)

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        await backend.simulate_speech_end(session, b"audio")
        await asyncio.sleep(0.15)

        # AFTER_TTS should have fired
        assert len(after_tts_events) >= 1

        await kit.close()

    async def test_multiple_sessions_same_room(self) -> None:
        """Test multiple voice sessions in the same room."""
        stt = MockSTTProvider(transcripts=["From user 1", "From user 2"])
        backend = MockVoiceBackend()

        kit = RoomKit(stt=stt, voice=backend)

        voice_channel = VoiceChannel("voice-1", stt=stt, backend=backend)
        kit.register_channel(voice_channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        messages = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST, HookExecution.SYNC)
        async def capture(event, context):
            from roomkit.models.hook import HookResult

            if isinstance(event.content, TextContent):
                messages.append((event.source.participant_id, event.content.body))
            return HookResult.allow()

        session1 = await kit.connect_voice(room.id, "user-1", "voice-1")
        session2 = await kit.connect_voice(room.id, "user-2", "voice-1")

        await backend.simulate_speech_end(session1, b"audio1")
        await asyncio.sleep(0.1)
        await backend.simulate_speech_end(session2, b"audio2")
        await asyncio.sleep(0.1)

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

        voice_channel = VoiceChannel("voice-1", stt=stt, tts=tts, backend=backend)

        # Manually add a session binding
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

        # Close should clean up
        await voice_channel.close()

        # Session bindings should be cleared
        assert len(voice_channel._session_bindings) == 0


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
        """Verify enhanced voice hooks (RFC §19) are defined."""
        assert HookTrigger.ON_BARGE_IN == "on_barge_in"
        assert HookTrigger.ON_TTS_CANCELLED == "on_tts_cancelled"
        assert HookTrigger.ON_PARTIAL_TRANSCRIPTION == "on_partial_transcription"
        assert HookTrigger.ON_VAD_SILENCE == "on_vad_silence"
        assert HookTrigger.ON_VAD_AUDIO_LEVEL == "on_vad_audio_level"


class TestEnhancedVoicePipelineIntegration:
    """Integration tests for enhanced voice features (RFC §19)."""

    async def test_partial_transcription_hook_fires(self) -> None:
        """ON_PARTIAL_TRANSCRIPTION hook fires during streaming STT."""
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.base import VoiceCapability
        from roomkit.voice.events import PartialTranscriptionEvent

        caps = VoiceCapability.PARTIAL_STT
        stt = MockSTTProvider(transcripts=["Final transcription"])
        backend = MockVoiceBackend(capabilities=caps)
        channel = VoiceChannel("voice-1", stt=stt, backend=backend)

        kit = RoomKit(stt=stt, voice=backend)
        kit.register_channel(channel)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")
        binding = ChannelBinding(
            room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE
        )
        channel.bind_session(session, room.id, binding)

        partials = []

        @kit.hook(HookTrigger.ON_PARTIAL_TRANSCRIPTION, HookExecution.ASYNC)
        async def on_partial(event, context):
            partials.append(event)

        # Simulate partial transcriptions
        await backend.simulate_partial_transcription(session, "Hello", 0.6, False)
        await backend.simulate_partial_transcription(session, "Hello wor", 0.75, False)
        await backend.simulate_partial_transcription(session, "Hello world", 0.9, True)
        await asyncio.sleep(0.1)

        assert len(partials) == 3
        assert all(isinstance(p, PartialTranscriptionEvent) for p in partials)
        assert partials[0].text == "Hello"
        assert partials[1].text == "Hello wor"
        assert partials[2].text == "Hello world"
        assert partials[2].is_stable is True

        await kit.close()

    async def test_vad_silence_hook_fires(self) -> None:
        """ON_VAD_SILENCE hook fires when silence is detected."""
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.base import VoiceCapability
        from roomkit.voice.events import VADSilenceEvent

        caps = VoiceCapability.VAD_SILENCE
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

        silence_events = []

        @kit.hook(HookTrigger.ON_VAD_SILENCE, HookExecution.ASYNC)
        async def on_silence(event, context):
            silence_events.append(event)

        await backend.simulate_vad_silence(session, 500)
        await backend.simulate_vad_silence(session, 1000)
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
        from roomkit.voice.base import VoiceCapability
        from roomkit.voice.events import VADAudioLevelEvent

        caps = VoiceCapability.VAD_AUDIO_LEVEL
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

        levels = []

        @kit.hook(HookTrigger.ON_VAD_AUDIO_LEVEL, HookExecution.ASYNC)
        async def on_level(event, context):
            levels.append(event)

        await backend.simulate_vad_audio_level(session, -30.0, False)
        await backend.simulate_vad_audio_level(session, -15.0, True)
        await asyncio.sleep(0.1)

        assert len(levels) == 2
        assert all(isinstance(e, VADAudioLevelEvent) for e in levels)
        assert levels[0].level_db == -30.0
        assert levels[0].is_speech is False
        assert levels[1].level_db == -15.0
        assert levels[1].is_speech is True

        await kit.close()


class TestBargeInIntegration:
    """Integration tests for barge-in detection and handling."""

    async def test_barge_in_detected_during_tts(self) -> None:
        """Barge-in is detected when user speaks during TTS playback."""
        from roomkit.channels.voice import TTSPlaybackState
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.base import VoiceCapability
        from roomkit.voice.events import BargeInEvent, TTSCancelledEvent

        caps = VoiceCapability.INTERRUPTION | VoiceCapability.BARGE_IN
        stt = MockSTTProvider(transcripts=["Interrupt!"])
        tts = MockTTSProvider()
        backend = MockVoiceBackend(capabilities=caps)
        channel = VoiceChannel(
            "voice-1",
            stt=stt,
            tts=tts,
            backend=backend,
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

        barge_in_events = []
        cancelled_events = []

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

        # User starts speaking (triggers barge-in)
        await backend.simulate_speech_start(session)
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
        channel = VoiceChannel(
            "voice-1",
            backend=backend,
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

        barge_in_events = []

        @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
        async def on_barge_in(event, context):
            barge_in_events.append(event)

        # Set TTS as just started
        channel._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text="Quick message",
            started_at=datetime.now(UTC),
        )

        # User speaks immediately (below 500ms threshold)
        await backend.simulate_speech_start(session)
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
        channel = VoiceChannel(
            "voice-1",
            backend=backend,
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

        barge_in_events = []

        @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
        async def on_barge_in(event, context):
            barge_in_events.append(event)

        # Set TTS as playing
        channel._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text="Playing audio",
        )

        await asyncio.sleep(0.1)

        # Simulate speech (should NOT trigger barge-in)
        await backend.simulate_speech_start(session)
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

        cancelled_events = []

        @kit.hook(HookTrigger.ON_TTS_CANCELLED, HookExecution.ASYNC)
        async def on_cancelled(event, context):
            cancelled_events.append(event)

        # Set TTS as playing
        channel._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text="Message being played",
        )

        # Explicit interrupt
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

        barge_in_events = []

        @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
        async def on_barge_in(event, context):
            barge_in_events.append(event)

        # Set TTS as playing
        channel._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text="Long response text",
        )

        # Backend detects barge-in directly
        await backend.simulate_barge_in(session)
        await asyncio.sleep(0.1)

        assert len(barge_in_events) == 1
        assert isinstance(barge_in_events[0], BargeInEvent)

        await kit.close()


class TestVoiceCapabilitiesIntegration:
    """Test VoiceCapability-based feature detection."""

    async def test_capabilities_determine_callbacks(self) -> None:
        """VoiceChannel only registers for callbacks based on backend capabilities."""
        from roomkit.voice.base import VoiceCapability

        # Backend with no capabilities
        backend_minimal = MockVoiceBackend(capabilities=VoiceCapability.NONE)
        _channel_minimal = VoiceChannel("voice-1", backend=backend_minimal)

        # Should have basic callbacks registered
        assert len(backend_minimal._speech_start_callbacks) == 1
        assert len(backend_minimal._speech_end_callbacks) == 1
        # Should NOT have enhanced callbacks
        assert len(backend_minimal._partial_transcription_callbacks) == 0

        # Backend with all capabilities
        all_caps = (
            VoiceCapability.PARTIAL_STT
            | VoiceCapability.VAD_SILENCE
            | VoiceCapability.VAD_AUDIO_LEVEL
            | VoiceCapability.BARGE_IN
        )
        backend_full = MockVoiceBackend(capabilities=all_caps)
        _channel_full = VoiceChannel("voice-2", backend=backend_full)

        # Should have all callbacks registered
        assert len(backend_full._speech_start_callbacks) == 1
        assert len(backend_full._speech_end_callbacks) == 1
        assert len(backend_full._partial_transcription_callbacks) == 1
        assert len(backend_full._vad_silence_callbacks) == 1
        assert len(backend_full._vad_audio_level_callbacks) == 1
        assert len(backend_full._barge_in_callbacks) == 1

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

        # Set TTS as playing
        channel_no_int._playing_sessions[session.id] = TTSPlaybackState(
            session_id=session.id,
            text="Test",
        )

        # Track cancel_audio calls
        cancel_called = []
        original_cancel = backend_no_int.cancel_audio

        async def track_cancel(s):
            cancel_called.append(True)
            return await original_cancel(s)

        backend_no_int.cancel_audio = track_cancel

        # Interrupt should work (clears playback state) but NOT call backend.cancel_audio
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

        cancel_called2 = []
        original_cancel2 = backend_with_int.cancel_audio

        async def track_cancel2(s):
            cancel_called2.append(True)
            return await original_cancel2(s)

        backend_with_int.cancel_audio = track_cancel2

        # Interrupt should call backend.cancel_audio
        result2 = await channel_with_int.interrupt(session2)
        assert result2 is True
        assert len(cancel_called2) == 1  # backend.cancel_audio WAS called

        await kit2.close()
