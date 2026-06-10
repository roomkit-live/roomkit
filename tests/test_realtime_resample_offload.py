"""Off-loop resampling: ordering, state safety, and executor lifecycle.

The per-channel single-thread resample executor is the ordering backbone of
the outbound audio path: chunk resampling, the end-of-response flush,
barge-in resets, and session-close resampler cleanup all serialize through
it, keeping the event loop free without reordering frames.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport


async def wait_until(predicate: Any, timeout: float = 2.0) -> None:
    """Poll until ``predicate()`` is true — bounded, executor-aware wait."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met within timeout")


class RecordingResampler:
    """Stateful fake resampler that records ops, threads, and sequence.

    ``resample`` returns the 2-byte little-endian sequence number so the
    receiving side can assert frame order byte-for-byte.
    """

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.seq = 0
        self.ops: list[tuple[str, str]] = []  # (op, thread name)

    def _record(self, op: str) -> None:
        self.ops.append((op, threading.current_thread().name))

    def resample(
        self, frame: AudioFrame, target_rate: int, channels: int, width: int
    ) -> AudioFrame:
        if self.delay:
            time.sleep(self.delay)
        self.seq += 1
        self._record("resample")
        return AudioFrame(
            data=self.seq.to_bytes(2, "little"),
            sample_rate=target_rate,
            channels=channels,
            sample_width=width,
        )

    def flush(self, target_rate: int, channels: int, width: int) -> AudioFrame | None:
        self._record("flush")
        return None

    def reset(self) -> None:
        self._record("reset")

    def close(self) -> None:
        self._record("close")


@pytest.fixture
def provider() -> MockRealtimeProvider:
    return MockRealtimeProvider()


@pytest.fixture
def transport() -> MockRealtimeTransport:
    return MockRealtimeTransport()


@pytest.fixture
def channel(
    provider: MockRealtimeProvider, transport: MockRealtimeTransport
) -> RealtimeVoiceChannel:
    return RealtimeVoiceChannel(
        "rt-offload",
        provider=provider,
        transport=transport,
        input_sample_rate=16000,
        output_sample_rate=24000,
        transport_sample_rate=8000,
    )


@pytest.fixture
async def session(channel: RealtimeVoiceChannel) -> VoiceSession:
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "rt-offload")
    return await channel.start_session(room.id, "user-1", "fake-ws")


def _swap_resamplers(
    channel: RealtimeVoiceChannel,
    session: VoiceSession,
    inbound: RecordingResampler,
    outbound: RecordingResampler,
) -> None:
    channel._session_resamplers[session.id] = (inbound, outbound)


class TestOffLoopExecution:
    async def test_outbound_frames_keep_order_and_run_off_loop(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        session: VoiceSession,
    ) -> None:
        outbound = RecordingResampler(delay=0.005)
        _swap_resamplers(channel, session, RecordingResampler(), outbound)

        for _ in range(6):
            await provider.simulate_audio(session, b"\x01\x00" * 240)
        await wait_until(lambda: len(transport.sent_audio) == 6)

        sent = [audio for _, audio in transport.sent_audio]
        assert sent == [seq.to_bytes(2, "little") for seq in range(1, 7)]
        assert all(thread.startswith("rk-resample") for _, thread in outbound.ops)

    async def test_inbound_frames_keep_order_and_run_off_loop(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        session: VoiceSession,
    ) -> None:
        inbound = RecordingResampler(delay=0.005)
        _swap_resamplers(channel, session, inbound, RecordingResampler())

        for _ in range(4):
            await transport.simulate_client_audio(session, b"\x01\x00" * 80)
        await wait_until(lambda: len(provider.sent_audio) == 4)

        sent = [audio for _, audio in provider.sent_audio]
        assert sent == [seq.to_bytes(2, "little") for seq in range(1, 5)]
        assert all(thread.startswith("rk-resample") for _, thread in inbound.ops)

    async def test_real_resampler_output_matches_sequential_reference(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        session: VoiceSession,
    ) -> None:
        """Off-loop execution must not change resampler output or state flow."""
        live = channel._session_resamplers[session.id][1]
        reference = type(live)()

        chunks = [
            bytes((i * 7 + j) % 251 for j in range(480)) for i in range(5)
        ]  # 240 samples @ 24kHz per chunk, deterministic non-trivial content
        expected = []
        for chunk in chunks:
            frame = AudioFrame(data=chunk, sample_rate=24000, channels=1, sample_width=2)
            out = reference.resample(frame, 8000, 1, 2)
            if out.data:
                expected.append(bytes(out.data))

        for chunk in chunks:
            await provider.simulate_audio(session, chunk)
        await wait_until(lambda: len(transport.sent_audio) == len(expected))

        assert [audio for _, audio in transport.sent_audio] == expected


class TestOrderingContracts:
    async def test_end_of_response_arrives_after_slow_resampled_audio(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        session: VoiceSession,
    ) -> None:
        """The eor marker takes the same executor hop as the chunks."""
        _swap_resamplers(channel, session, RecordingResampler(), RecordingResampler(delay=0.02))

        call_log: list[str] = []
        orig_send_audio = transport.send_audio
        orig_end_of_response = transport.end_of_response

        async def tracking_send_audio(s: VoiceSession, audio: Any) -> None:
            call_log.append("audio")
            await orig_send_audio(s, audio)

        def tracking_end_of_response(s: VoiceSession) -> None:
            call_log.append("end_of_response")
            orig_end_of_response(s)

        transport.send_audio = tracking_send_audio  # type: ignore[method-assign]
        transport.end_of_response = tracking_end_of_response  # type: ignore[method-assign]

        await provider.simulate_response_start(session)
        for _ in range(3):
            await provider.simulate_audio(session, b"\x01\x00" * 240)
        await provider.simulate_response_end(session)

        await wait_until(lambda: "end_of_response" in call_log)
        eor_index = call_log.index("end_of_response")
        assert call_log[:eor_index].count("audio") == 3, call_log

    async def test_barge_in_reset_serializes_behind_inflight_resample(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        session: VoiceSession,
    ) -> None:
        outbound = RecordingResampler(delay=0.05)
        _swap_resamplers(channel, session, RecordingResampler(), outbound)

        await provider.simulate_audio(session, b"\x01\x00" * 240)
        await asyncio.sleep(0)  # let the audio task submit its executor job
        await provider.simulate_speech_start(session)

        await wait_until(lambda: ("reset", "rk-resample_0") in outbound.ops)
        ops = [op for op, _ in outbound.ops]
        assert ops.index("resample") < ops.index("reset")
        # The pre-barge-in chunk is stale (generation bumped) — never sent.
        assert transport.sent_audio == []


class TestExecutorLifecycle:
    async def test_session_end_closes_resamplers_on_executor(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        session: VoiceSession,
    ) -> None:
        outbound = RecordingResampler()
        _swap_resamplers(channel, session, RecordingResampler(), outbound)
        await provider.simulate_audio(session, b"\x01\x00" * 240)
        await wait_until(lambda: len(transport.sent_audio) == 1)

        await channel.end_session(session)
        await wait_until(lambda: ("close", "rk-resample_0") in outbound.ops)

    async def test_channel_close_shuts_down_executor(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
        session: VoiceSession,
    ) -> None:
        await provider.simulate_audio(session, b"\x01\x00" * 240)
        await wait_until(lambda: channel._resample_executor is not None)

        await channel.close()
        assert channel._resample_executor is None
