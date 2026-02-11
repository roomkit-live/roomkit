"""Tests for VoiceChannel batch STT mode."""

from __future__ import annotations

import asyncio

import pytest

from roomkit import (
    HookExecution,
    HookTrigger,
    MockSTTProvider,
    MockVoiceBackend,
    RoomKit,
    VoiceChannel,
)
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider

# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestBatchModeConstruction:
    """Validation rules for batch_mode parameter."""

    def test_batch_mode_sets_flag(self) -> None:
        stt = MockSTTProvider()
        backend = MockVoiceBackend()
        pipeline = AudioPipelineConfig()  # no VAD
        ch = VoiceChannel("v1", stt=stt, backend=backend, pipeline=pipeline, batch_mode=True)
        assert ch._batch_mode is True
        assert ch.info["batch_mode"] is True

    def test_batch_mode_disables_continuous_stt(self) -> None:
        """batch_mode=True must suppress continuous STT even if provider streams."""
        stt = MockSTTProvider(streaming=True)
        backend = MockVoiceBackend()
        pipeline = AudioPipelineConfig()  # no VAD
        ch = VoiceChannel("v1", stt=stt, backend=backend, pipeline=pipeline, batch_mode=True)
        assert ch._continuous_stt is False

    def test_batch_mode_without_stt_raises(self) -> None:
        backend = MockVoiceBackend()
        pipeline = AudioPipelineConfig()
        with pytest.raises(ValueError, match="requires an STT provider"):
            VoiceChannel("v1", backend=backend, pipeline=pipeline, batch_mode=True)

    def test_batch_mode_with_vad_raises(self) -> None:
        stt = MockSTTProvider()
        backend = MockVoiceBackend()
        vad = MockVADProvider()
        pipeline = AudioPipelineConfig(vad=vad)
        with pytest.raises(ValueError, match="incompatible with VAD"):
            VoiceChannel("v1", stt=stt, backend=backend, pipeline=pipeline, batch_mode=True)

    def test_default_batch_mode_is_false(self) -> None:
        backend = MockVoiceBackend()
        ch = VoiceChannel("v1", backend=backend)
        assert ch._batch_mode is False
        assert ch.info["batch_mode"] is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel_and_kit(
    transcripts: list[str] | None = None,
) -> tuple[VoiceChannel, MockVoiceBackend, MockSTTProvider, RoomKit]:
    """Create a batch-mode VoiceChannel wired to a RoomKit instance."""
    stt = MockSTTProvider(transcripts=transcripts or ["hello world"])
    backend = MockVoiceBackend()
    pipeline = AudioPipelineConfig()  # no VAD
    ch = VoiceChannel("voice-1", stt=stt, backend=backend, pipeline=pipeline, batch_mode=True)
    kit = RoomKit(voice=backend)
    kit.register_channel(ch)
    return ch, backend, stt, kit


async def _bind_session(ch: VoiceChannel, kit: RoomKit) -> tuple[VoiceSession, str]:
    """Create a room, attach the channel, connect a voice session."""
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    session = await kit.connect_voice(room.id, "user-1", "voice-1")
    return session, room.id


# ---------------------------------------------------------------------------
# Buffer accumulation
# ---------------------------------------------------------------------------


class TestBatchBufferAccumulation:
    """Frames accumulate in the batch buffer."""

    async def test_frames_accumulate(self) -> None:
        ch, backend, _stt, kit = _make_channel_and_kit()
        session, _ = await _bind_session(ch, kit)

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x01" * 100))
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x02" * 50))

        assert ch.stt_buffer_size(session) == 300  # 200 + 100 bytes
        await kit.close()

    async def test_stt_buffer_size_tracks_correctly(self) -> None:
        ch, backend, _stt, kit = _make_channel_and_kit()
        session, _ = await _bind_session(ch, kit)

        assert ch.stt_buffer_size(session) == 0

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x01" * 80))
        assert ch.stt_buffer_size(session) == 160

        await kit.close()

    async def test_buffer_clears_after_flush(self) -> None:
        ch, backend, stt, kit = _make_channel_and_kit()
        session, _ = await _bind_session(ch, kit)

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x01" * 80))
        assert ch.stt_buffer_size(session) > 0

        await ch.flush_stt(session)
        assert ch.stt_buffer_size(session) == 0
        await kit.close()

    async def test_buffer_clears_after_clear_stt_buffer(self) -> None:
        ch, backend, _stt, kit = _make_channel_and_kit()
        session, _ = await _bind_session(ch, kit)

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x01" * 80))
        discarded = ch.clear_stt_buffer(session)

        assert discarded == 160
        assert ch.stt_buffer_size(session) == 0
        await kit.close()

    async def test_buffer_clears_on_unbind(self) -> None:
        ch, backend, _stt, kit = _make_channel_and_kit()
        session, _ = await _bind_session(ch, kit)

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x01" * 80))
        assert ch.stt_buffer_size(session) > 0

        ch.unbind_session(session)
        assert ch.stt_buffer_size(session) == 0
        await kit.close()

    async def test_buffer_capped_at_max(self) -> None:
        ch, backend, _stt, kit = _make_channel_and_kit()
        session, _ = await _bind_session(ch, kit)

        # Fill buffer to near max
        max_bytes = ch._MAX_BATCH_BUFFER_BYTES
        chunk_size = 32000  # 1 second at 16kHz mono 16-bit
        written = 0
        while written + chunk_size <= max_bytes:
            await backend.simulate_audio_received(session, AudioFrame(data=b"\x00" * chunk_size))
            written += chunk_size

        # This should push us over the limit — frame should be dropped
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00" * chunk_size))

        assert ch.stt_buffer_size(session) <= max_bytes
        await kit.close()


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


class TestBatchTranscription:
    """flush_stt() calls STT provider and returns results."""

    async def test_flush_calls_stt_transcribe(self) -> None:
        ch, backend, stt, kit = _make_channel_and_kit(transcripts=["test result"])
        session, _ = await _bind_session(ch, kit)

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x01" * 80))
        result = await ch.flush_stt(session)

        assert result.text == "test result"
        assert result.is_final is True
        assert len(stt.calls) == 1
        await kit.close()

    async def test_flush_empty_buffer_returns_empty(self) -> None:
        ch, _backend, stt, kit = _make_channel_and_kit()
        session, _ = await _bind_session(ch, kit)

        result = await ch.flush_stt(session)

        assert result.text == ""
        assert result.is_final is True
        assert len(stt.calls) == 0  # STT not called for empty buffer
        await kit.close()

    async def test_flush_without_batch_mode_raises(self) -> None:
        backend = MockVoiceBackend()
        stt = MockSTTProvider()
        ch = VoiceChannel("v1", stt=stt, backend=backend)

        session = VoiceSession(id="s1", room_id="r1", participant_id="u1", channel_id="v1")
        with pytest.raises(RuntimeError, match="requires batch_mode"):
            await ch.flush_stt(session)

    async def test_clear_without_batch_mode_raises(self) -> None:
        backend = MockVoiceBackend()
        stt = MockSTTProvider()
        ch = VoiceChannel("v1", stt=stt, backend=backend)

        session = VoiceSession(id="s1", room_id="r1", participant_id="u1", channel_id="v1")
        with pytest.raises(RuntimeError, match="requires batch_mode"):
            ch.clear_stt_buffer(session)

    async def test_multiple_flush_cycles(self) -> None:
        ch, backend, stt, kit = _make_channel_and_kit(transcripts=["first", "second"])
        session, _ = await _bind_session(ch, kit)

        # First cycle
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x01" * 80))
        result1 = await ch.flush_stt(session)
        assert result1.text == "first"

        # Second cycle
        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x02" * 80))
        result2 = await ch.flush_stt(session)
        assert result2.text == "second"

        assert len(stt.calls) == 2
        await kit.close()


# ---------------------------------------------------------------------------
# Routing (flush_stt with route=True)
# ---------------------------------------------------------------------------


class TestBatchRouting:
    """flush_stt(route=True) fires hooks and routes text."""

    async def test_route_true_fires_hooks(self) -> None:
        ch, backend, _stt, kit = _make_channel_and_kit(transcripts=["routed text"])
        session, _ = await _bind_session(ch, kit)

        hook_order: list[str] = []

        @kit.hook(HookTrigger.ON_SPEECH_END, HookExecution.ASYNC)
        async def on_speech_end(event, context):
            hook_order.append("ON_SPEECH_END")

        @kit.hook(HookTrigger.ON_TRANSCRIPTION, HookExecution.SYNC)
        async def on_transcription(event, context):
            from roomkit.models.hook import HookResult

            hook_order.append("ON_TRANSCRIPTION")
            return HookResult.allow()

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x01" * 80))
        result = await ch.flush_stt(session, route=True)

        assert result.text == "routed text"
        # Give async hooks time to fire
        await asyncio.sleep(0.05)

        assert "ON_SPEECH_END" in hook_order
        assert "ON_TRANSCRIPTION" in hook_order
        await kit.close()

    async def test_route_false_does_not_fire_hooks(self) -> None:
        ch, backend, _stt, kit = _make_channel_and_kit(transcripts=["quiet text"])
        session, _ = await _bind_session(ch, kit)

        hook_fired: list[str] = []

        @kit.hook(HookTrigger.ON_SPEECH_END, HookExecution.ASYNC)
        async def on_speech_end(event, context):
            hook_fired.append("ON_SPEECH_END")

        @kit.hook(HookTrigger.ON_TRANSCRIPTION, HookExecution.SYNC)
        async def on_transcription(event, context):
            from roomkit.models.hook import HookResult

            hook_fired.append("ON_TRANSCRIPTION")
            return HookResult.allow()

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x01" * 80))
        result = await ch.flush_stt(session, route=False)

        assert result.text == "quiet text"
        await asyncio.sleep(0.05)

        assert len(hook_fired) == 0
        await kit.close()

    async def test_route_with_empty_text_skips_hooks(self) -> None:
        ch, backend, _stt, kit = _make_channel_and_kit(transcripts=[""])
        session, _ = await _bind_session(ch, kit)

        hook_fired: list[str] = []

        @kit.hook(HookTrigger.ON_TRANSCRIPTION, HookExecution.SYNC)
        async def on_transcription(event, context):
            from roomkit.models.hook import HookResult

            hook_fired.append("ON_TRANSCRIPTION")
            return HookResult.allow()

        await backend.simulate_audio_received(session, AudioFrame(data=b"\x00\x01" * 80))
        await ch.flush_stt(session, route=True)
        await asyncio.sleep(0.05)

        assert len(hook_fired) == 0
        await kit.close()
