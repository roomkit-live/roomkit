"""Tests for audio frame rate limiting in VoiceChannel."""

from __future__ import annotations

from roomkit.channels.voice import VoiceChannel
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.pipeline.config import AudioPipelineConfig
from roomkit.voice.pipeline.vad.mock import MockVADProvider


def _make_frame(sample_rate: int = 16000) -> AudioFrame:
    """Create a minimal audio frame."""
    return AudioFrame(
        data=b"\x00\x00" * 160,  # 10ms at 16kHz
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
    )


def _make_session(session_id: str = "sess-1") -> VoiceSession:
    return VoiceSession(
        id=session_id,
        room_id="room-1",
        participant_id="user-1",
        channel_id="voice",
        state=VoiceSessionState.ACTIVE,
    )


def _make_binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="voice",
        room_id="room-1",
        channel_type=ChannelType.VOICE,
        capabilities=ChannelCapabilities(media_types=[ChannelMediaType.AUDIO]),
    )


def test_excess_frames_dropped() -> None:
    """Frames exceeding max_audio_frames_per_second should be dropped."""
    backend = MockVoiceBackend()
    pipeline_config = AudioPipelineConfig(
        vad=MockVADProvider(events=[]),
    )

    channel = VoiceChannel(
        "voice",
        backend=backend,
        pipeline=pipeline_config,
        max_audio_frames_per_second=5,
    )

    session = _make_session()
    binding = _make_binding()

    # Bind session
    channel._session_bindings[session.id] = ("room-1", binding)

    # Track how many frames reach the pipeline
    process_count = 0
    original_process_frame = channel._pipeline.process_frame  # type: ignore[union-attr]

    def counting_process_frame(s: VoiceSession, f: AudioFrame) -> None:
        nonlocal process_count
        process_count += 1
        original_process_frame(s, f)

    channel._pipeline.process_frame = counting_process_frame  # type: ignore[assignment]

    # Send 10 frames rapidly (within same 1-second window)
    for _ in range(10):
        channel._on_audio_received(session, _make_frame())

    # Only 5 should have been passed through
    assert process_count == 5


def test_frame_counter_resets_after_one_second() -> None:
    """Frame counter should reset after 1 second window."""
    backend = MockVoiceBackend()
    pipeline_config = AudioPipelineConfig(
        vad=MockVADProvider(events=[]),
    )

    channel = VoiceChannel(
        "voice",
        backend=backend,
        pipeline=pipeline_config,
        max_audio_frames_per_second=2,
    )

    session = _make_session()
    binding = _make_binding()
    channel._session_bindings[session.id] = ("room-1", binding)

    process_count = 0

    def counting_process_frame(s: VoiceSession, f: AudioFrame) -> None:
        nonlocal process_count
        process_count += 1

    channel._pipeline.process_frame = counting_process_frame  # type: ignore[assignment]

    # Send 2 frames (fills the limit)
    channel._on_audio_received(session, _make_frame())
    channel._on_audio_received(session, _make_frame())
    assert process_count == 2

    # Third frame should be dropped
    channel._on_audio_received(session, _make_frame())
    assert process_count == 2

    # Simulate time advancing by 1+ second
    window_start, count = channel._frame_counts[session.id]
    channel._frame_counts[session.id] = (window_start - 1.1, count)

    # Now frames should pass again
    channel._on_audio_received(session, _make_frame())
    assert process_count == 3


def test_no_limit_allows_all_frames() -> None:
    """Without max_audio_frames_per_second, all frames should pass."""
    backend = MockVoiceBackend()
    pipeline_config = AudioPipelineConfig(
        vad=MockVADProvider(events=[]),
    )

    channel = VoiceChannel(
        "voice",
        backend=backend,
        pipeline=pipeline_config,
        # No max_audio_frames_per_second
    )

    session = _make_session()
    binding = _make_binding()
    channel._session_bindings[session.id] = ("room-1", binding)

    process_count = 0

    def counting_process_frame(s: VoiceSession, f: AudioFrame) -> None:
        nonlocal process_count
        process_count += 1

    channel._pipeline.process_frame = counting_process_frame  # type: ignore[assignment]

    # All 100 frames should pass
    for _ in range(100):
        channel._on_audio_received(session, _make_frame())

    assert process_count == 100


def test_cleanup_on_unbind() -> None:
    """Frame counters should be cleaned up when session is unbound."""
    backend = MockVoiceBackend()
    pipeline_config = AudioPipelineConfig(
        vad=MockVADProvider(events=[]),
    )

    channel = VoiceChannel(
        "voice",
        backend=backend,
        pipeline=pipeline_config,
        max_audio_frames_per_second=10,
    )

    session = _make_session()
    binding = _make_binding()
    channel._session_bindings[session.id] = ("room-1", binding)

    # Send a frame to populate the counter
    channel._on_audio_received(session, _make_frame())
    assert session.id in channel._frame_counts

    # Unbind
    channel.unbind_session(session)
    assert session.id not in channel._frame_counts
