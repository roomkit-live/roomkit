"""Tests for AudioBridge — session-to-session audio forwarding."""

from __future__ import annotations

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk
from roomkit.voice.bridge import AudioBridge, AudioBridgeConfig


class TestAudioBridge:
    """Core AudioBridge tests."""

    async def test_two_party_forward(self) -> None:
        """Audio from session A is forwarded to session B and vice versa."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        frame_a = AudioFrame(data=b"\x01\x02" * 160, sample_rate=16000)
        bridge.forward(s1, frame_a)

        # s2 should have received the audio
        assert len(backend.sent_audio) == 1
        assert backend.sent_audio[0][0] == s2.id
        assert backend.sent_audio[0][1] == frame_a.data

    async def test_bidirectional_forward(self) -> None:
        """Audio flows both ways."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        frame_a = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        frame_b = AudioFrame(data=b"\x02" * 320, sample_rate=16000)

        bridge.forward(s1, frame_a)
        bridge.forward(s2, frame_b)

        assert len(backend.sent_audio) == 2
        # First forward: s1 -> s2
        assert backend.sent_audio[0][0] == s2.id
        assert backend.sent_audio[0][1] == frame_a.data
        # Second forward: s2 -> s1
        assert backend.sent_audio[1][0] == s1.id
        assert backend.sent_audio[1][1] == frame_b.data

    async def test_no_forward_to_self(self) -> None:
        """Audio from a session is never forwarded back to itself."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        frame = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        bridge.forward(s1, frame)

        for sid, _ in backend.sent_audio:
            assert sid != s1.id

    async def test_single_session_no_forward(self) -> None:
        """No forwarding when only one session in the room."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        bridge.add_session(s1, "room-1", backend)

        frame = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        bridge.forward(s1, frame)

        assert len(backend.sent_audio) == 0

    async def test_remove_session(self) -> None:
        """Removing a session stops forwarding to it."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)
        bridge.remove_session(s2.id)

        frame = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        bridge.forward(s1, frame)

        assert len(backend.sent_audio) == 0

    async def test_separate_rooms(self) -> None:
        """Sessions in different rooms don't hear each other."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-2", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-2", backend)

        frame = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        bridge.forward(s1, frame)

        assert len(backend.sent_audio) == 0

    async def test_max_participants(self) -> None:
        """Exceeding max_participants raises RuntimeError."""
        backend = MockVoiceBackend()
        bridge = AudioBridge(AudioBridgeConfig(max_participants=2))

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")
        s3 = await backend.connect("room-1", "user-3", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        with pytest.raises(RuntimeError, match="max bridge participants"):
            bridge.add_session(s3, "room-1", backend)

    async def test_get_participant_count(self) -> None:
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        assert bridge.get_participant_count("room-1") == 0

        s1 = await backend.connect("room-1", "user-1", "voice")
        bridge.add_session(s1, "room-1", backend)
        assert bridge.get_participant_count("room-1") == 1

        s2 = await backend.connect("room-1", "user-2", "voice")
        bridge.add_session(s2, "room-1", backend)
        assert bridge.get_participant_count("room-1") == 2

        bridge.remove_session(s1.id)
        assert bridge.get_participant_count("room-1") == 1

    async def test_close(self) -> None:
        """close() removes all sessions."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")
        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        bridge.close()

        assert bridge.get_participant_count("room-1") == 0
        # Forwarding after close is a no-op
        frame = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        bridge.forward(s1, frame)
        assert len(backend.sent_audio) == 0

    async def test_add_same_session_twice(self) -> None:
        """Adding the same session twice is idempotent."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s1, "room-1", backend)

        assert bridge.get_participant_count("room-1") == 1

    async def test_three_party_forward(self) -> None:
        """With 3 sessions, audio is forwarded to the other two."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")
        s3 = await backend.connect("room-1", "user-3", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)
        bridge.add_session(s3, "room-1", backend)

        frame = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        bridge.forward(s1, frame)

        assert len(backend.sent_audio) == 2
        target_ids = {sid for sid, _ in backend.sent_audio}
        assert target_ids == {s2.id, s3.id}

    async def test_remove_unknown_session(self) -> None:
        """Removing an unknown session is a no-op."""
        bridge = AudioBridge()
        bridge.remove_session("unknown")  # Should not raise

    async def test_forward_unknown_session(self) -> None:
        """Forwarding from an unknown session is a no-op."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        frame = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        bridge.forward(s1, frame)  # Not registered — should not raise

        assert len(backend.sent_audio) == 0

    async def test_frame_filter_blocks(self) -> None:
        """Frame filter returning None drops the frame."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        # Block all frames
        bridge.set_frame_filter(lambda session, frame: None)

        frame = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        bridge.forward(s1, frame)

        assert len(backend.sent_audio) == 0

    async def test_frame_filter_modifies(self) -> None:
        """Frame filter can modify the frame before forwarding."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        # Replace data with silence
        silence = b"\x00" * 320

        def replace_with_silence(session, frame):
            return AudioFrame(data=silence, sample_rate=frame.sample_rate)

        bridge.set_frame_filter(replace_with_silence)

        frame = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        bridge.forward(s1, frame)

        assert len(backend.sent_audio) == 1
        assert backend.sent_audio[0][1] == silence

    async def test_frame_filter_passthrough(self) -> None:
        """Frame filter returning the frame passes it through."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        bridge.set_frame_filter(lambda session, frame: frame)

        frame = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        bridge.forward(s1, frame)

        assert len(backend.sent_audio) == 1
        assert backend.sent_audio[0][1] == frame.data

    async def test_frame_filter_remove(self) -> None:
        """Setting filter to None removes it."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        bridge.set_frame_filter(lambda session, frame: None)
        bridge.set_frame_filter(None)  # Remove filter

        frame = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        bridge.forward(s1, frame)

        assert len(backend.sent_audio) == 1

    async def test_frame_processor(self) -> None:
        """Frame processor transforms audio before sending to each target."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        processed_data = b"\xaa" * 320
        processed_targets: list[str] = []

        def processor(target_session, frame):
            processed_targets.append(target_session.id)
            return AudioChunk(data=processed_data, sample_rate=16000, channels=1)

        bridge.set_frame_processor(processor)

        frame = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        bridge.forward(s1, frame)

        assert len(backend.sent_audio) == 1
        assert backend.sent_audio[0][0] == s2.id
        assert backend.sent_audio[0][1] == processed_data
        assert processed_targets == [s2.id]

    async def test_frame_processor_per_target(self) -> None:
        """Frame processor is called once per target session."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")
        s3 = await backend.connect("room-1", "user-3", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)
        bridge.add_session(s3, "room-1", backend)

        call_count = 0

        def processor(target_session, frame):
            nonlocal call_count
            call_count += 1
            return AudioChunk(data=frame.data, sample_rate=16000, channels=1)

        bridge.set_frame_processor(processor)

        frame = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        bridge.forward(s1, frame)

        assert call_count == 2  # s2 and s3

    async def test_filter_and_processor_combined(self) -> None:
        """Filter runs before processor — blocked frames skip processor."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        processor_called = False

        def processor(target_session, frame):
            nonlocal processor_called
            processor_called = True
            return AudioChunk(data=frame.data, sample_rate=16000, channels=1)

        bridge.set_frame_filter(lambda session, frame: None)
        bridge.set_frame_processor(processor)

        frame = AudioFrame(data=b"\x01" * 320, sample_rate=16000)
        bridge.forward(s1, frame)

        assert len(backend.sent_audio) == 0
        assert not processor_called


class TestVoiceChannelBridge:
    """Test VoiceChannel bridge integration."""

    async def test_bridge_true_creates_bridge(self) -> None:
        from roomkit import VoiceChannel

        channel = VoiceChannel("voice", bridge=True)
        assert channel._bridge is not None
        assert isinstance(channel._bridge, AudioBridge)

    async def test_bridge_config_creates_bridge(self) -> None:
        from roomkit import VoiceChannel

        config = AudioBridgeConfig(max_participants=5)
        channel = VoiceChannel("voice", bridge=config)
        assert channel._bridge is not None
        assert channel._bridge.config.max_participants == 5

    async def test_bridge_none_by_default(self) -> None:
        from roomkit import VoiceChannel

        channel = VoiceChannel("voice")
        assert channel._bridge is None

    async def test_bind_registers_with_bridge(self) -> None:
        from roomkit import VoiceChannel
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType

        backend = MockVoiceBackend()
        channel = VoiceChannel("voice", backend=backend, bridge=True)

        s1 = await backend.connect("room-1", "user-1", "voice")
        binding = ChannelBinding(
            room_id="room-1",
            channel_id="voice",
            channel_type=ChannelType.VOICE,
        )
        channel.bind_session(s1, "room-1", binding)

        assert channel._bridge is not None
        assert channel._bridge.get_participant_count("room-1") == 1

    async def test_unbind_unregisters_from_bridge(self) -> None:
        from roomkit import VoiceChannel
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType

        backend = MockVoiceBackend()
        channel = VoiceChannel("voice", backend=backend, bridge=True)

        s1 = await backend.connect("room-1", "user-1", "voice")
        binding = ChannelBinding(
            room_id="room-1",
            channel_id="voice",
            channel_type=ChannelType.VOICE,
        )
        channel.bind_session(s1, "room-1", binding)
        channel.unbind_session(s1)

        assert channel._bridge is not None
        assert channel._bridge.get_participant_count("room-1") == 0

    async def test_bridge_forwards_audio_between_sessions(self) -> None:
        """End-to-end: audio from session A reaches session B via bridge."""
        from roomkit import VoiceChannel
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType

        backend = MockVoiceBackend()
        channel = VoiceChannel("voice", backend=backend, bridge=True)

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")
        binding = ChannelBinding(
            room_id="room-1",
            channel_id="voice",
            channel_type=ChannelType.VOICE,
        )

        channel.bind_session(s1, "room-1", binding)
        channel.bind_session(s2, "room-1", binding)

        # Simulate inbound audio from s1
        frame = AudioFrame(data=b"\x01\x02" * 160, sample_rate=16000)
        await backend.simulate_audio_received(s1, frame)

        # The bridge should have forwarded to s2 via send_audio_sync
        forwarded = [(sid, data) for sid, data in backend.sent_audio if sid == s2.id]
        assert len(forwarded) >= 1
        assert forwarded[0][1] == frame.data

    async def test_close_cleans_up_bridge(self) -> None:
        from roomkit import VoiceChannel

        backend = MockVoiceBackend()
        channel = VoiceChannel("voice", backend=backend, bridge=True)

        s1 = await backend.connect("room-1", "user-1", "voice")
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType

        binding = ChannelBinding(
            room_id="room-1",
            channel_id="voice",
            channel_type=ChannelType.VOICE,
        )
        channel.bind_session(s1, "room-1", binding)

        await channel.close()

        assert channel._bridge is not None
        assert channel._bridge.get_participant_count("room-1") == 0

    async def test_set_bridge_filter_mutes_session(self) -> None:
        """set_bridge_filter can mute a specific session."""
        from roomkit import VoiceChannel
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType

        backend = MockVoiceBackend()
        channel = VoiceChannel("voice", backend=backend, bridge=True)

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")
        binding = ChannelBinding(
            room_id="room-1",
            channel_id="voice",
            channel_type=ChannelType.VOICE,
        )
        channel.bind_session(s1, "room-1", binding)
        channel.bind_session(s2, "room-1", binding)

        # Mute s1 via bridge filter
        channel.set_bridge_filter(lambda session, frame: None if session.id == s1.id else frame)

        # Audio from s1 should be blocked
        frame = AudioFrame(data=b"\x01\x02" * 160, sample_rate=16000)
        await backend.simulate_audio_received(s1, frame)
        blocked = [(sid, data) for sid, data in backend.sent_audio if sid == s2.id]
        assert len(blocked) == 0

        # Audio from s2 should pass through
        backend.sent_audio.clear()
        await backend.simulate_audio_received(s2, frame)
        forwarded = [(sid, data) for sid, data in backend.sent_audio if sid == s1.id]
        assert len(forwarded) >= 1

    async def test_bridge_outbound_pipeline_processes_frames(self) -> None:
        """Bridged audio passes through the outbound pipeline (recorder tap)."""
        from roomkit import VoiceChannel
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType
        from roomkit.voice.pipeline.config import AudioPipelineConfig
        from roomkit.voice.pipeline.recorder.base import RecordingConfig
        from roomkit.voice.pipeline.recorder.mock import MockAudioRecorder

        recorder = MockAudioRecorder()
        pipeline = AudioPipelineConfig(
            recorder=recorder,
            recording_config=RecordingConfig(),
        )
        backend = MockVoiceBackend()
        channel = VoiceChannel(
            "voice",
            backend=backend,
            bridge=True,
            pipeline=pipeline,
        )

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")
        binding = ChannelBinding(
            room_id="room-1",
            channel_id="voice",
            channel_type=ChannelType.VOICE,
        )
        channel.bind_session(s1, "room-1", binding)
        channel.bind_session(s2, "room-1", binding)

        # Both sessions should have recording started via on_session_active
        assert len(recorder.started) == 2

        # Simulate inbound audio from s1
        frame = AudioFrame(data=b"\x01\x02" * 160, sample_rate=16000)
        await backend.simulate_audio_received(s1, frame)

        # Audio should reach s2
        forwarded = [(sid, data) for sid, data in backend.sent_audio if sid == s2.id]
        assert len(forwarded) >= 1

        # Inbound pipeline tapped s1's recording (recorder sees inbound frame)
        assert len(recorder.inbound_frames) >= 1

        # Outbound pipeline tapped s2's recording (bridge forwarded through
        # process_outbound which calls recorder.tap_outbound for target session)
        assert len(recorder.outbound_frames) >= 1
        # The outbound tap should be for s2's recording handle, not s1's
        s2_rec_handle_id = "rec_2"  # second session started = rec_2
        outbound_for_s2 = [
            (hid, f) for hid, f in recorder.outbound_frames if hid == s2_rec_handle_id
        ]
        assert len(outbound_for_s2) >= 1

    async def test_set_bridge_filter_on_no_bridge(self) -> None:
        """set_bridge_filter is a no-op when bridge is not configured."""
        from roomkit import VoiceChannel

        channel = VoiceChannel("voice")
        channel.set_bridge_filter(lambda s, f: f)  # Should not raise
