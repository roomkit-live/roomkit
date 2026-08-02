"""Tests for the inbound DSP offload — FIFO per stream, parallel across.

The unit tests exercise :class:`InboundFrameOffload` directly; the
integration test proves a VoiceChannel configured with
``inbound_dsp_threads`` still runs the full frame→VAD→STT path, just off
the event loop.
"""

from __future__ import annotations

import asyncio
import threading

from roomkit import RoomKit, VoiceChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.offload import InboundFrameOffload
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider


class TestInboundFrameOffload:
    def test_one_stream_is_fifo_whatever_the_pool_size(self) -> None:
        # Queue bound above the burst: this test is about ordering, not drops.
        offload = InboundFrameOffload(4, max_queued_frames=1000)
        seen: list[int] = []
        for i in range(200):
            offload.submit("s1", seen.append, i)
        assert offload.wait_idle(timeout=5.0)
        offload.shutdown()
        assert seen == list(range(200))

    def test_streams_run_in_parallel(self) -> None:
        """A blocked stream must not hold another stream's frames hostage."""
        offload = InboundFrameOffload(2)
        gate = threading.Event()
        s2_done = threading.Event()

        offload.submit("s1", gate.wait, 5.0)
        offload.submit("s2", s2_done.set)
        try:
            assert s2_done.wait(2.0), "s2 waited behind s1's blocked frame"
        finally:
            gate.set()
            offload.shutdown()

    def test_a_full_stream_queue_drops_its_oldest_frames(self) -> None:
        offload = InboundFrameOffload(1, max_queued_frames=4)
        gate = threading.Event()
        seen: list[int] = []

        offload.submit("s1", gate.wait, 5.0)  # occupies the drainer
        for i in range(1, 10):  # 9 more; queue keeps the newest 4
            offload.submit("s1", seen.append, i)
        gate.set()
        assert offload.wait_idle(timeout=5.0)
        offload.shutdown()

        assert seen == [6, 7, 8, 9]
        assert offload.dropped("s1") == 5

    def test_release_drops_a_gone_streams_queue(self) -> None:
        offload = InboundFrameOffload(1)
        gate = threading.Event()
        seen: list[int] = []

        offload.submit("s1", gate.wait, 5.0)
        offload.submit("s1", seen.append, 1)
        offload.release("s1")
        gate.set()
        assert offload.wait_idle(timeout=5.0)
        offload.shutdown()
        assert seen == []

    def test_a_failing_frame_does_not_stall_the_stream(self) -> None:
        offload = InboundFrameOffload(1)
        seen: list[int] = []

        def boom() -> None:
            raise RuntimeError("stage blew up")

        offload.submit("s1", boom)
        offload.submit("s1", seen.append, 1)
        assert offload.wait_idle(timeout=5.0)
        offload.shutdown()
        assert seen == [1]

    def test_submit_after_shutdown_is_a_noop(self) -> None:
        offload = InboundFrameOffload(1)
        offload.shutdown()
        offload.submit("s1", lambda: None)  # must not raise


class TestVoiceChannelWithOffload:
    async def test_the_frame_to_stt_path_runs_through_the_pool(self) -> None:
        stt = MockSTTProvider(transcripts=["Hello"])
        backend = MockVoiceBackend()
        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_START),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"\x01\x00" * 80),
            ]
        )
        pipeline = AudioPipelineConfig(vad=vad, inbound_dsp_threads=2)

        kit = RoomKit(voice=backend)
        channel = VoiceChannel(
            "voice-1", stt=stt, tts=MockTTSProvider(), backend=backend, pipeline=pipeline
        )
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.connect_voice(room.id, "user-1", "voice-1")

        assert channel._inbound_offload is not None

        sessions = list(channel._session_bindings.keys())
        session = backend.get_session(sessions[0])
        assert session is not None
        await backend.simulate_audio_received(session, AudioFrame(data=b"frame1"))
        await backend.simulate_audio_received(session, AudioFrame(data=b"frame2"))

        # The DSP ran on the pool; wait for it, then for the scheduled
        # speech-end coroutine on the loop.
        assert await asyncio.to_thread(channel._inbound_offload.wait_idle, timeout=5.0)
        await asyncio.sleep(0.15)

        assert len(stt.calls) >= 1
        await kit.close()

    async def test_without_the_knob_processing_stays_inline(self) -> None:
        backend = MockVoiceBackend()
        pipeline = AudioPipelineConfig(vad=MockVADProvider(events=[]))
        kit = RoomKit(voice=backend)
        channel = VoiceChannel(
            "voice-1",
            stt=MockSTTProvider(transcripts=[]),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=pipeline,
        )
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.connect_voice(room.id, "user-1", "voice-1")

        assert channel._inbound_offload is None
        await kit.close()
