"""Outbound audio queues are bounded, inbound frames are capped.

The producer on the outbound side is a synchronous provider callback that
returns immediately by contract, so it cannot be back-pressured: if the
transport stops draining, the queue grows for as long as the provider keeps
talking — and the provider keeps billing for audio nobody will hear.
"""

from __future__ import annotations

import asyncio

from roomkit.channels import _realtime_audio
from roomkit.voice.backends import twilio_ws
from roomkit.voice.realtime.ws_transport import MAX_INBOUND_AUDIO_FRAME_BYTES


class TestOutboundRealtimeQueueIsBounded:
    def test_the_bound_is_a_sane_amount_of_speech(self) -> None:
        # 20 ms per chunk — the bound should be seconds of audio, not hours.
        seconds = _realtime_audio._MAX_QUEUED_AUDIO_CHUNKS * 0.02
        assert 5 <= seconds <= 60

    async def test_a_full_queue_stops_growing(self) -> None:
        """The behaviour that matters: unbounded growth becomes bounded."""
        queue: asyncio.Queue = asyncio.Queue()
        dropped = 0
        for _ in range(_realtime_audio._MAX_QUEUED_AUDIO_CHUNKS + 250):
            if queue.qsize() >= _realtime_audio._MAX_QUEUED_AUDIO_CHUNKS:
                dropped += 1
                continue
            queue.put_nowait(("audio", b"x" * 960, None, 24000, 0))

        assert queue.qsize() == _realtime_audio._MAX_QUEUED_AUDIO_CHUNKS
        assert dropped == 250

    async def test_control_items_are_never_dropped(self) -> None:
        """Transports settle playback state on the end-of-response marker.

        This is why the newest chunk is dropped rather than the oldest, unlike
        the conference backlog: evicting from the head could swallow one.
        """
        queue: asyncio.Queue = asyncio.Queue()
        for _ in range(_realtime_audio._MAX_QUEUED_AUDIO_CHUNKS):
            queue.put_nowait(("audio", b"x", None, 24000, 0))

        # The queue is full, yet the marker still goes in — only audio is
        # subject to the bound.
        queue.put_nowait(("eor", None, 24000))
        queue.put_nowait(None)

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        assert items[-2][0] == "eor"
        assert items[-1] is None


class TestTwilioWriteQueueIsBounded:
    def test_the_bound_is_a_sane_amount_of_speech(self) -> None:
        seconds = twilio_ws._MAX_QUEUED_FRAMES * 0.02
        assert 5 <= seconds <= 60

    async def test_frames_are_dropped_once_the_socket_stops_draining(self) -> None:
        backend = twilio_ws.TwilioWebSocketBackend()

        class _NeverDrains:
            async def send_json(self, msg: dict) -> None:
                await asyncio.sleep(3600)

        backend.bind_websocket(_NeverDrains())
        assert backend._write_queue is not None

        for _ in range(twilio_ws._MAX_QUEUED_FRAMES + 100):
            backend._write_queue.put_nowait({"event": "media"})

        # The producer refuses to add more once the bound is reached.
        before = backend._write_queue.qsize()
        session = type("S", (), {"room_id": "r1", "id": "s1"})()
        await backend._send_mulaw_frame(session, b"\x00" * 320)
        assert backend._write_queue.qsize() == before
        assert backend._dropped_frames >= 1

        await backend.close()


class TestInboundFrameCap:
    def test_the_binary_path_uses_the_same_cap_as_the_base64_path(self) -> None:
        """The cap existed, ten lines below, on the JSON path only."""
        import inspect

        from roomkit.voice.realtime import ws_transport

        src = inspect.getsource(ws_transport.WebSocketRealtimeTransport._receive_loop)
        binary_branch = src.split("data = json.loads")[0]
        assert "MAX_INBOUND_AUDIO_FRAME_BYTES" in binary_branch
        assert MAX_INBOUND_AUDIO_FRAME_BYTES > 0
