"""Unit tests for OutboundAudioPacer."""

from __future__ import annotations

import asyncio

from roomkit.voice.realtime.pacer import OutboundAudioPacer


def _make_audio(duration_ms: int, sample_rate: int = 8000) -> bytes:
    """Create silent PCM audio of the given duration."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * num_samples


class TestPushAndReceive:
    async def test_push_and_receive(self) -> None:
        """Pushed audio arrives via send_fn."""
        received: list[bytes] = []

        async def send_fn(audio: bytes) -> None:
            received.append(audio)

        pacer = OutboundAudioPacer(send_fn, sample_rate=8000, prebuffer_ms=0)
        await pacer.start()

        audio = _make_audio(20)
        pacer.push(audio)
        pacer.end_of_response()
        await asyncio.sleep(0.1)

        await pacer.stop()

        assert len(received) >= 1
        # All pushed bytes should have been sent
        total = b"".join(received)
        assert total == audio


class TestPrebuffer:
    async def test_prebuffer_accumulates(self) -> None:
        """First burst should be ~150ms worth of audio."""
        received: list[bytes] = []

        async def send_fn(audio: bytes) -> None:
            received.append(audio)

        pacer = OutboundAudioPacer(send_fn, sample_rate=8000, prebuffer_ms=150)
        await pacer.start()

        # Push 200ms of audio in 20ms chunks (10 chunks)
        chunk = _make_audio(20)
        for _ in range(10):
            pacer.push(chunk)

        pacer.end_of_response()
        await asyncio.sleep(0.3)
        await pacer.stop()

        assert len(received) >= 1
        # First burst should be at least 150ms (2400 bytes at 8kHz mono 16-bit)
        first_burst = received[0]
        assert len(first_burst) >= int(8000 * 2 * 0.15)


class TestInterrupt:
    async def test_interrupt_drains_queue(self) -> None:
        """After interrupt(), no more audio should be sent."""
        received: list[bytes] = []

        async def send_fn(audio: bytes) -> None:
            received.append(audio)

        pacer = OutboundAudioPacer(send_fn, sample_rate=8000, prebuffer_ms=0)
        await pacer.start()

        # Push a lot of audio
        chunk = _make_audio(20)
        for _ in range(50):
            pacer.push(chunk)

        # Immediately interrupt
        pacer.interrupt()
        await asyncio.sleep(0.1)

        count_after_interrupt = len(received)

        # Push more audio (new response) to verify pacer still works
        pacer.push(_make_audio(20))
        pacer.end_of_response()
        await asyncio.sleep(0.1)

        await pacer.stop()

        # Some audio from the new response should arrive
        assert len(received) > count_after_interrupt

    async def test_interrupt_wakes_pacing_sleep(self) -> None:
        """Interrupt should wake the pacer from a pacing sleep."""
        received: list[bytes] = []
        send_event = asyncio.Event()

        async def send_fn(audio: bytes) -> None:
            received.append(audio)
            send_event.set()

        pacer = OutboundAudioPacer(send_fn, sample_rate=8000, prebuffer_ms=50)
        await pacer.start()

        # Push enough audio to enter pacing (500ms = well over prebuffer)
        chunk = _make_audio(20)
        for _ in range(25):
            pacer.push(chunk)

        # Wait for first burst to be sent
        await asyncio.wait_for(send_event.wait(), timeout=1.0)

        # Interrupt mid-pacing
        pacer.interrupt()
        await asyncio.sleep(0.05)

        # Now send a new response to confirm pacer is alive
        send_event.clear()
        pacer.push(_make_audio(20))
        pacer.end_of_response()
        await asyncio.wait_for(send_event.wait(), timeout=1.0)

        await pacer.stop()


class TestEndOfResponse:
    async def test_end_of_response_resets_pacing(self) -> None:
        """Two consecutive responses each get their own pre-buffer."""
        received: list[bytes] = []

        async def send_fn(audio: bytes) -> None:
            received.append(audio)

        pacer = OutboundAudioPacer(send_fn, sample_rate=8000, prebuffer_ms=100)
        await pacer.start()

        # First response
        chunk = _make_audio(20)
        for _ in range(10):
            pacer.push(chunk)
        pacer.end_of_response()
        await asyncio.sleep(0.2)

        first_response_count = len(received)

        # Second response
        for _ in range(10):
            pacer.push(chunk)
        pacer.end_of_response()
        await asyncio.sleep(0.2)

        await pacer.stop()

        # Both responses should have produced output
        assert first_response_count >= 1
        assert len(received) > first_response_count


class TestStop:
    async def test_stop_exits_cleanly(self) -> None:
        """Pacer stops without error."""

        async def send_fn(audio: bytes) -> None:
            pass

        pacer = OutboundAudioPacer(send_fn, sample_rate=8000)
        await pacer.start()
        assert pacer._task is not None

        await pacer.stop()
        assert pacer._task is None

    async def test_stop_idempotent(self) -> None:
        """Calling stop twice doesn't raise."""

        async def send_fn(audio: bytes) -> None:
            pass

        pacer = OutboundAudioPacer(send_fn, sample_rate=8000)
        await pacer.start()

        await pacer.stop()
        await pacer.stop()  # Should not raise


class TestWaitForResponseDone:
    async def test_wait_completes_after_flush(self) -> None:
        """wait_for_response_done() resolves after response is flushed."""
        received: list[bytes] = []

        async def send_fn(audio: bytes) -> None:
            received.append(audio)

        pacer = OutboundAudioPacer(send_fn, sample_rate=8000, prebuffer_ms=0)
        await pacer.start()

        pacer.push(_make_audio(20))
        pacer.end_of_response()
        await asyncio.wait_for(pacer.wait_for_response_done(), timeout=1.0)

        await pacer.stop()
        assert len(received) >= 1

    async def test_interrupt_unblocks_waiter(self) -> None:
        """interrupt() unblocks a pending wait_for_response_done()."""

        async def send_fn(audio: bytes) -> None:
            pass

        pacer = OutboundAudioPacer(send_fn, sample_rate=8000, prebuffer_ms=150)
        await pacer.start()

        # Push a lot of audio so pacing takes a long time
        for _ in range(100):
            pacer.push(_make_audio(20))
        pacer.end_of_response()

        # Start waiting, then interrupt before it can finish
        wait_task = asyncio.create_task(pacer.wait_for_response_done())
        await asyncio.sleep(0.05)
        pacer.interrupt()

        # Should unblock quickly
        await asyncio.wait_for(wait_task, timeout=1.0)

        await pacer.stop()

    async def test_no_response_returns_immediately(self) -> None:
        """wait_for_response_done() returns immediately when no response in flight."""

        async def send_fn(audio: bytes) -> None:
            pass

        pacer = OutboundAudioPacer(send_fn, sample_rate=8000)
        await pacer.start()

        # No end_of_response() called, event is set by default
        await asyncio.wait_for(pacer.wait_for_response_done(), timeout=0.1)

        await pacer.stop()


class TestEmptyAudio:
    async def test_empty_audio_skipped(self) -> None:
        """Empty bytes should not be sent."""
        received: list[bytes] = []

        async def send_fn(audio: bytes) -> None:
            received.append(audio)

        pacer = OutboundAudioPacer(send_fn, sample_rate=8000, prebuffer_ms=0)
        await pacer.start()

        pacer.push(b"")  # Should be ignored by push()
        pacer.push(b"")  # Should be ignored by push()
        pacer.end_of_response()
        await asyncio.sleep(0.1)

        await pacer.stop()

        # No audio should have been sent
        assert len(received) == 0
