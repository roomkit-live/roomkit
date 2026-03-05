"""Tests for AudioBridge — session-to-session audio forwarding."""

from __future__ import annotations

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
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
