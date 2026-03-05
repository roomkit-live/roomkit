"""Tests for AudioBridge — session-to-session audio forwarding."""

from __future__ import annotations

import struct

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk
from roomkit.voice.bridge import AudioBridge, AudioBridgeConfig
from roomkit.voice.pipeline.mixer.python import PythonMixerProvider


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


# ======================================================================
# Helper: create int16 PCM frames with known sample values
# ======================================================================


def _make_frame(
    samples: list[int],
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> AudioFrame:
    """Create an AudioFrame from a list of int16 sample values."""
    data = struct.pack(f"<{len(samples)}h", *samples)
    return AudioFrame(
        data=data,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )


def _decode_frame(frame: AudioFrame) -> list[int]:
    """Decode an AudioFrame to a list of int16 sample values."""
    count = len(frame.data) // 2
    return list(struct.unpack(f"<{count}h", frame.data))


def _get_mixer_providers():
    """Return all available mixer providers for parametrized tests."""
    providers = [PythonMixerProvider()]
    try:
        from roomkit.voice.pipeline.mixer.numpy import NumpyMixerProvider

        providers.append(NumpyMixerProvider())
    except ImportError:
        pass
    return providers


_MIXERS = _get_mixer_providers()
_MIXER_IDS = [m.name for m in _MIXERS]


class TestMixerProviders:
    """Unit tests for MixerProvider implementations (Python + NumPy)."""

    @pytest.mark.parametrize("mixer", _MIXERS, ids=_MIXER_IDS)
    def test_single_frame_passthrough(self, mixer) -> None:
        """Single frame is returned unchanged."""
        frame = _make_frame([100, -200, 300])
        result = mixer.mix([frame])
        assert result is frame

    @pytest.mark.parametrize("mixer", _MIXERS, ids=_MIXER_IDS)
    def test_two_frame_sum(self, mixer) -> None:
        """Two frames are summed sample-by-sample (no scaling)."""
        f1 = _make_frame([1000, -2000, 3000])
        f2 = _make_frame([500, 1000, -1000])
        result = mixer.mix([f1, f2])
        samples = _decode_frame(result)
        assert samples == [1500, -1000, 2000]

    @pytest.mark.parametrize("mixer", _MIXERS, ids=_MIXER_IDS)
    def test_two_frame_clipping(self, mixer) -> None:
        """Two-frame sum clips to int16 range."""
        f1 = _make_frame([30000, -30000])
        f2 = _make_frame([10000, -10000])
        result = mixer.mix([f1, f2])
        samples = _decode_frame(result)
        assert samples == [32767, -32768]

    @pytest.mark.parametrize("mixer", _MIXERS, ids=_MIXER_IDS)
    def test_three_frame_average(self, mixer) -> None:
        """Three+ frames are averaged (divided by n)."""
        f1 = _make_frame([3000, -6000])
        f2 = _make_frame([6000, -3000])
        f3 = _make_frame([9000, 0])
        result = mixer.mix([f1, f2, f3])
        samples = _decode_frame(result)
        assert samples == [6000, -3000]

    @pytest.mark.parametrize("mixer", _MIXERS, ids=_MIXER_IDS)
    def test_different_lengths_uses_shortest(self, mixer) -> None:
        """When frames differ in length, result uses shortest."""
        f1 = _make_frame([100, 200, 300])
        f2 = _make_frame([100, 200])
        result = mixer.mix([f1, f2])
        samples = _decode_frame(result)
        assert len(samples) == 2
        assert samples == [200, 400]

    @pytest.mark.parametrize("mixer", _MIXERS, ids=_MIXER_IDS)
    def test_preserves_metadata(self, mixer) -> None:
        """Result preserves sample_rate, channels, sample_width from first frame."""
        f1 = _make_frame([100], sample_rate=8000, channels=1, sample_width=2)
        f2 = _make_frame([200], sample_rate=8000, channels=1, sample_width=2)
        result = mixer.mix([f1, f2])
        assert result.sample_rate == 8000
        assert result.channels == 1
        assert result.sample_width == 2

    @pytest.mark.parametrize("mixer", _MIXERS, ids=_MIXER_IDS)
    def test_silence_mix(self, mixer) -> None:
        """Mixing with silence (all zeros) returns the non-silent frame's values."""
        f1 = _make_frame([1000, -500])
        f2 = _make_frame([0, 0])
        result = mixer.mix([f1, f2])
        samples = _decode_frame(result)
        assert samples == [1000, -500]


class TestNPartyMixing:
    """Integration tests for N-party mixing strategy."""

    async def test_three_party_mix_each_hears_others(self) -> None:
        """3-party mix: each participant hears a mix of the other two."""
        backend = MockVoiceBackend()
        config = AudioBridgeConfig(mixing_strategy="mix")
        bridge = AudioBridge(config)

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")
        s3 = await backend.connect("room-1", "user-3", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)
        bridge.add_session(s3, "room-1", backend)

        # s1 sends audio
        frame_s1 = _make_frame([1000] * 10)
        bridge.forward(s1, frame_s1)

        # Only s1 has sent — s2 and s3 each hear s1's frame (single source, no mixing)
        assert len(backend.sent_audio) == 2
        target_ids = {sid for sid, _ in backend.sent_audio}
        assert target_ids == {s2.id, s3.id}

        backend.sent_audio.clear()

        # s2 now sends audio
        frame_s2 = _make_frame([2000] * 10)
        bridge.forward(s2, frame_s2)

        # Now both s1 and s2 have sent.
        # s1 should hear s2's frame (only other contributor)
        # s3 should hear s1 + s2 mixed
        sent = {sid: data for sid, data in backend.sent_audio}
        assert s1.id in sent
        assert s3.id in sent

        # s3 hears mix of s1 and s2 (sum for 2 sources)
        s3_samples = list(struct.unpack(f"<{10}h", sent[s3.id][:20]))
        assert s3_samples == [3000] * 10  # 1000 + 2000

        # s1 hears only s2 (single source, passthrough)
        s1_samples = list(struct.unpack(f"<{10}h", sent[s1.id][:20]))
        assert s1_samples == [2000] * 10

    async def test_four_party_mix(self) -> None:
        """4-party mix: each participant hears averaged mix of the other three."""
        backend = MockVoiceBackend()
        config = AudioBridgeConfig(mixing_strategy="mix")
        bridge = AudioBridge(config)

        sessions = []
        for i in range(4):
            s = await backend.connect("room-1", f"user-{i}", "voice")
            bridge.add_session(s, "room-1", backend)
            sessions.append(s)

        # All 4 send audio with distinct values
        for i, s in enumerate(sessions):
            bridge.forward(s, _make_frame([(i + 1) * 1000] * 10))

        # After all 4 have sent, the last forward() triggers sends.
        # Find what session[0] received — should be mix of sessions 1,2,3
        s0_audio = [data for sid, data in backend.sent_audio if sid == sessions[0].id]
        assert len(s0_audio) >= 1
        # The most recent audio for s0 is from the last forward() call
        last_audio = s0_audio[-1]
        samples = list(struct.unpack(f"<{10}h", last_audio[:20]))
        # Sessions 1,2,3 had values 2000,3000,4000 — averaged: (2000+3000+4000)//3 = 3000
        assert samples == [3000] * 10

    async def test_mix_no_self_audio(self) -> None:
        """A participant never hears their own audio in the mix."""
        backend = MockVoiceBackend()
        config = AudioBridgeConfig(mixing_strategy="mix")
        bridge = AudioBridge(config)

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        frame = _make_frame([5000] * 10)
        bridge.forward(s1, frame)

        # s1 should NOT receive anything — only s2 gets the audio
        for sid, _ in backend.sent_audio:
            assert sid != s1.id

    async def test_mix_single_session_no_send(self) -> None:
        """With only one session, nothing is sent."""
        backend = MockVoiceBackend()
        config = AudioBridgeConfig(mixing_strategy="mix")
        bridge = AudioBridge(config)

        s1 = await backend.connect("room-1", "user-1", "voice")
        bridge.add_session(s1, "room-1", backend)

        bridge.forward(s1, _make_frame([1000] * 10))
        assert len(backend.sent_audio) == 0

    async def test_mix_filter_applies(self) -> None:
        """Frame filter applies before mix storage."""
        backend = MockVoiceBackend()
        config = AudioBridgeConfig(mixing_strategy="mix")
        bridge = AudioBridge(config)

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        # Block all frames from s1
        bridge.set_frame_filter(lambda session, frame: None if session.id == s1.id else frame)

        bridge.forward(s1, _make_frame([1000] * 10))
        assert len(backend.sent_audio) == 0

    async def test_mix_remove_session_cleans_buffer(self) -> None:
        """Removing a session removes its frame from the mix buffer."""
        backend = MockVoiceBackend()
        config = AudioBridgeConfig(mixing_strategy="mix")
        bridge = AudioBridge(config)

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")
        s3 = await backend.connect("room-1", "user-3", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)
        bridge.add_session(s3, "room-1", backend)

        # All three send audio
        bridge.forward(s1, _make_frame([1000] * 10))
        bridge.forward(s2, _make_frame([2000] * 10))

        backend.sent_audio.clear()

        # Remove s1 — its frame should be gone from the buffer
        bridge.remove_session(s1.id)

        # s3 sends — s2 should only hear s3 (not s1)
        bridge.forward(s3, _make_frame([3000] * 10))
        s2_audio = [data for sid, data in backend.sent_audio if sid == s2.id]
        assert len(s2_audio) >= 1
        samples = list(struct.unpack(f"<{10}h", s2_audio[-1][:20]))
        # Only s3 contributed (value 3000) — no mixing needed
        assert samples == [3000] * 10

    async def test_mix_processor_applied(self) -> None:
        """Frame processor runs on mixed output before sending."""
        backend = MockVoiceBackend()
        config = AudioBridgeConfig(mixing_strategy="mix")
        bridge = AudioBridge(config)

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        processed_data = b"\xbb" * 20

        def processor(target_session, frame):
            return AudioChunk(data=processed_data, sample_rate=16000, channels=1)

        bridge.set_frame_processor(processor)

        bridge.forward(s1, _make_frame([1000] * 10))
        assert len(backend.sent_audio) == 1
        assert backend.sent_audio[0][1] == processed_data


class TestCrossRateResampling:
    """Tests for cross-rate resampling in the bridge."""

    async def test_resample_when_rates_differ(self) -> None:
        """Bridge resamples audio when source and target rates differ."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        # s1 at 16kHz, s2 at 8kHz
        s1 = await backend.connect("room-1", "user-1", "voice")
        s1.metadata["input_sample_rate"] = 16000
        s2 = await backend.connect("room-1", "user-2", "voice")
        s2.metadata["input_sample_rate"] = 8000

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        # s1 sends 16kHz audio (160 samples = 10ms)
        frame_16k = _make_frame([1000] * 160, sample_rate=16000)
        bridge.forward(s1, frame_16k)

        # s2 should receive audio resampled to 8kHz (80 samples = 10ms)
        assert len(backend.sent_audio) == 1
        sent_data = backend.sent_audio[0][1]
        # 8kHz, 10ms = 80 samples = 160 bytes
        assert len(sent_data) == 160

    async def test_same_rate_no_resample(self) -> None:
        """No resampling when source and target have the same rate."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s1.metadata["input_sample_rate"] = 16000
        s2 = await backend.connect("room-1", "user-2", "voice")
        s2.metadata["input_sample_rate"] = 16000

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        frame = _make_frame([1000] * 160, sample_rate=16000)
        bridge.forward(s1, frame)

        assert len(backend.sent_audio) == 1
        # Same rate — data passes through unchanged
        assert backend.sent_audio[0][1] == frame.data

    async def test_resample_skipped_with_processor(self) -> None:
        """When a frame processor is set, bridge skips its own resampling."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "user-1", "voice")
        s1.metadata["input_sample_rate"] = 16000
        s2 = await backend.connect("room-1", "user-2", "voice")
        s2.metadata["input_sample_rate"] = 8000

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        custom_data = b"\xcc" * 100

        def processor(target_session, frame):
            return AudioChunk(data=custom_data, sample_rate=8000, channels=1)

        bridge.set_frame_processor(processor)

        frame = _make_frame([1000] * 160, sample_rate=16000)
        bridge.forward(s1, frame)

        assert len(backend.sent_audio) == 1
        assert backend.sent_audio[0][1] == custom_data

    async def test_resample_8k_to_48k(self) -> None:
        """Bridge resamples 8kHz SIP audio to 48kHz WebRTC."""
        backend = MockVoiceBackend()
        bridge = AudioBridge()

        s1 = await backend.connect("room-1", "sip-user", "voice")
        s1.metadata["input_sample_rate"] = 8000
        s2 = await backend.connect("room-1", "webrtc-user", "voice")
        s2.metadata["input_sample_rate"] = 48000

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        # 80 samples at 8kHz = 10ms
        frame_8k = _make_frame([1000] * 80, sample_rate=8000)
        bridge.forward(s1, frame_8k)

        assert len(backend.sent_audio) == 1
        sent_data = backend.sent_audio[0][1]
        # 48kHz, 10ms = 480 samples = 960 bytes
        assert len(sent_data) == 960

    async def test_mix_cross_rate_resamples_before_mixing(self) -> None:
        """Mix strategy resamples all source frames to target rate before mixing."""
        backend = MockVoiceBackend()
        config = AudioBridgeConfig(mixing_strategy="mix")
        bridge = AudioBridge(config)

        # s1 at 8kHz, s2 at 16kHz, s3 at 16kHz (target)
        s1 = await backend.connect("room-1", "sip-user", "voice")
        s1.metadata["input_sample_rate"] = 8000
        s2 = await backend.connect("room-1", "webrtc-user", "voice")
        s2.metadata["input_sample_rate"] = 16000
        s3 = await backend.connect("room-1", "listener", "voice")
        s3.metadata["input_sample_rate"] = 16000

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)
        bridge.add_session(s3, "room-1", backend)

        # s1 sends 10ms at 8kHz (80 samples)
        bridge.forward(s1, _make_frame([1000] * 80, sample_rate=8000))
        # s2 sends 10ms at 16kHz (160 samples)
        bridge.forward(s2, _make_frame([2000] * 160, sample_rate=16000))

        # s3 (16kHz) should receive a mix where s1's 8kHz frame was
        # resampled to 16kHz before mixing — result should be 160 samples
        s3_audio = [data for sid, data in backend.sent_audio if sid == s3.id]
        assert len(s3_audio) >= 1
        last = s3_audio[-1]
        # 160 samples * 2 bytes = 320 bytes (16kHz, 10ms)
        assert len(last) == 320

    async def test_enabled_false_disables_forwarding(self) -> None:
        """Setting enabled=False stops all forwarding."""
        backend = MockVoiceBackend()
        config = AudioBridgeConfig(enabled=False)
        bridge = AudioBridge(config)

        s1 = await backend.connect("room-1", "user-1", "voice")
        s2 = await backend.connect("room-1", "user-2", "voice")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        bridge.forward(s1, _make_frame([1000] * 160))
        assert len(backend.sent_audio) == 0
