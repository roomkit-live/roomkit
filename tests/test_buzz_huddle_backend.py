"""Tests for BuzzHuddleBackend (Buzz huddle voice transport).

These tests carry no ``buzzkit`` dependency: the backend duck-types the
huddle client, so a fake drives the whole inbound + outbound path without a
relay or the compiled wheel.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

import roomkit.voice.backends.buzz_huddle as buzz_huddle_module
from roomkit.voice.backends.buzz_huddle import BuzzHuddleBackend
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSession

# =============================================================================
# Helpers
# =============================================================================


@dataclass(frozen=True)
class FakeAudio:
    """Duck-typed buzzkit.HuddleAudio."""

    pcm: bytes
    is_dtx: bool = False
    peer_index: int = 0
    pubkey: str = "ab" * 32


@dataclass(frozen=True)
class FakePeerJoined:
    """Duck-typed roster event (no ``pcm`` attribute)."""

    pubkey: str
    peer_index: int


@dataclass
class FakeHuddleClient:
    """Duck-typed buzzkit.HuddleClient."""

    channel_id: str = "2a29484c-aac0-4ceb-93f4-9cee196348cb"
    peers: dict[int, str] = field(default_factory=dict)
    sent: list[bytes] = field(default_factory=list)
    queued: int = 0
    flushes: int = 0
    left: bool = False

    def __post_init__(self) -> None:
        self._events: asyncio.Queue = asyncio.Queue()

    # -- what the backend calls --------------------------------------------
    def send_pcm(self, pcm: bytes) -> None:
        self.sent.append(pcm)
        self.queued += 1

    def clear_queue(self) -> int:
        dropped, self.queued = self.queued, 0
        return dropped

    def flush_pcm(self) -> None:
        self.flushes += 1

    @property
    def queued_frames(self) -> int:
        return self.queued

    async def leave(self) -> None:
        self.left = True
        await self._events.put(None)

    async def events(self):
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    # -- test controls -------------------------------------------------------
    def feed(self, event) -> None:
        self._events.put_nowait(event)

    def end(self) -> None:
        self._events.put_nowait(None)


def make_session(session_id: str = "s1") -> VoiceSession:
    return VoiceSession(id=session_id, room_id="r", participant_id="p", channel_id="c")


@pytest.fixture(autouse=True)
def force_buzzkit_present(monkeypatch) -> None:
    """The backend never touches buzzkit APIs directly — only the guard."""
    monkeypatch.setattr(buzz_huddle_module, "HAS_BUZZKIT", True)


class FakePacer:
    """Deterministic stand-in for OutboundAudioPacer.

    Forwards each push straight to send_fn (no prebuffer/headroom timing —
    that behaviour is covered by the pacer's own tests) so the backend's
    wiring can be asserted synchronously.
    """

    def __init__(self, send_fn, **_kw) -> None:
        self._send_fn = send_fn
        self.pushed: list[bytes] = []
        self.interrupts = 0
        self.ends = 0

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def push(self, pcm: bytes) -> None:
        self.pushed.append(pcm)
        asyncio.ensure_future(self._send_fn(pcm))

    def interrupt(self) -> None:
        self.interrupts += 1

    def end_of_response(self) -> None:
        self.ends += 1


@pytest.fixture(autouse=True)
def fake_pacer(monkeypatch) -> None:
    """Swap the real (timed) pacer for the deterministic fake in all tests."""
    monkeypatch.setattr(
        "roomkit.voice.realtime.pacer.OutboundAudioPacer", FakePacer
    )


def make_backend(**kw) -> BuzzHuddleBackend:
    """Silence fill is disabled by default so assertions stay deterministic."""
    kw.setdefault("silence_fill", False)
    return BuzzHuddleBackend(**kw)


async def accept_fake(backend: BuzzHuddleBackend) -> tuple[VoiceSession, FakeHuddleClient]:
    session = make_session()
    client = FakeHuddleClient()
    await backend.accept(session, client)
    return session, client


async def settle() -> None:
    """Let the receive task process queued events."""
    for _ in range(3):
        await asyncio.sleep(0)


# =============================================================================
# Tests
# =============================================================================


class TestConstruction:
    def test_requires_buzzkit(self, monkeypatch) -> None:
        monkeypatch.setattr(buzz_huddle_module, "HAS_BUZZKIT", False)
        with pytest.raises(ImportError, match=r"roomkit\[buzz\]"):
            BuzzHuddleBackend()

    def test_name_and_capabilities(self) -> None:
        backend = make_backend()
        assert backend.name == "BuzzHuddleBackend"
        assert backend.capabilities & VoiceCapability.INTERRUPTION
        assert backend.auto_connect is False


class TestAccept:
    async def test_rejects_non_huddle_client(self) -> None:
        backend = make_backend()
        with pytest.raises(TypeError, match="HuddleClient"):
            await backend.accept(make_session(), object())

    async def test_tracks_session_and_channel(self) -> None:
        backend = make_backend()
        session, client = await accept_fake(backend)
        assert backend.get_session(session.id) is session
        assert backend.list_sessions("r") == [session]
        assert session.metadata["buzz_channel_id"] == client.channel_id
        await backend.disconnect(session)


class TestInboundAudio:
    async def test_audio_fires_callbacks_with_resampled_bytes(self) -> None:
        backend = make_backend()
        received: list[tuple[VoiceSession, bytes]] = []
        backend.on_audio_received(lambda s, audio: received.append((s, audio)))

        session, client = await accept_fake(backend)
        client.feed(FakeAudio(pcm=b"\x01\x02" * 960))  # 20 ms @ 48 kHz
        await settle()

        assert len(received) == 1
        got_session, audio = received[0]
        assert got_session is session
        # 48 kHz -> 16 kHz: ~320 samples (streaming resamplers may hold a
        # few samples of filter delay on the first chunk).
        assert 0 < len(audio) <= 640 and len(audio) % 2 == 0
        await backend.disconnect(session)

    async def test_dtx_frames_are_dropped(self) -> None:
        backend = make_backend()
        received: list[bytes] = []
        backend.on_audio_received(lambda s, audio: received.append(audio))

        session, client = await accept_fake(backend)
        client.feed(FakeAudio(pcm=b"\x00\x00" * 960, is_dtx=True))
        client.feed(FakeAudio(pcm=b"\x01\x01" * 960))
        await settle()

        assert len(received) == 1, "DTX frame must be dropped"
        await backend.disconnect(session)

    async def test_roster_events_update_metadata(self) -> None:
        backend = make_backend()
        session, client = await accept_fake(backend)
        client.peers = {0: "ab" * 32, 2: "cd" * 32}
        client.feed(FakePeerJoined(pubkey="cd" * 32, peer_index=2))
        await settle()

        assert session.metadata["buzz_peers"] == {0: "ab" * 32, 2: "cd" * 32}
        await backend.disconnect(session)

    async def test_stream_end_fires_disconnect_callbacks(self) -> None:
        backend = make_backend()
        disconnected: list[VoiceSession] = []
        backend.on_client_disconnected(lambda s: disconnected.append(s))

        session, client = await accept_fake(backend)
        client.end()
        await settle()

        assert disconnected == [session]
        await backend.disconnect(session)


class TestOutboundAudio:
    async def test_send_audio_bytes_pushed_and_relayed_resampled(self) -> None:
        backend = make_backend()
        session, client = await accept_fake(backend)
        pacer = backend._pacers[session.id]
        await backend.send_audio(session, b"\x05\x06" * 100)  # 100 samples @ 24 kHz
        await settle()
        # Pushed to the pacer (resampled 24 kHz -> 48 kHz, ~2x samples) and
        # relayed on to the client by the fake pacer.
        assert len(pacer.pushed) == 1
        assert 0 < len(pacer.pushed[0]) <= 400 and len(pacer.pushed[0]) % 2 == 0
        assert client.sent == pacer.pushed
        await backend.disconnect(session)

    async def test_send_audio_stream(self) -> None:
        backend = make_backend()
        session, _client = await accept_fake(backend)
        pacer = backend._pacers[session.id]

        async def chunks():
            yield AudioChunk(data=b"\x01\x00" * 240, sample_rate=24000)
            yield AudioChunk(data=b"\x02\x00" * 240, sample_rate=24000)

        await backend.send_audio(session, chunks())
        assert len(pacer.pushed) == 2
        await backend.disconnect(session)

    async def test_send_audio_unknown_session_is_noop(self) -> None:
        backend = make_backend()
        await backend.send_audio(make_session("ghost"), b"\x00")  # must not raise

    async def test_interrupt_pacer_and_client(self) -> None:
        backend = make_backend()
        session, client = await accept_fake(backend)
        pacer = backend._pacers[session.id]
        await backend.send_audio(session, b"\x00" * 1920)
        await settle()
        assert backend.is_playing(session)
        backend.interrupt(session)
        assert pacer.interrupts == 1
        assert not backend.is_playing(session)
        assert client.queued == 0
        await backend.disconnect(session)

    async def test_cancel_audio_returns_true(self) -> None:
        backend = make_backend()
        session, _client = await accept_fake(backend)
        assert await backend.cancel_audio(session) is True
        await backend.disconnect(session)

    async def test_end_of_response_signals_pacer(self) -> None:
        backend = make_backend()
        session, _client = await accept_fake(backend)
        pacer = backend._pacers[session.id]
        backend.end_of_response(session)
        assert pacer.ends == 1
        await backend.disconnect(session)


class TestSilenceFill:
    async def test_streams_silence_frames_while_idle(self) -> None:
        backend = make_backend(silence_fill=True)
        received: list[bytes] = []
        backend.on_audio_received(lambda s, audio: received.append(audio))

        session, _client = await accept_fake(backend)
        await asyncio.sleep(0.15)
        await backend.disconnect(session)

        assert received, "idle session must produce silence frames"
        # 20 ms of silence at the provider input rate (16 kHz -> 640 bytes).
        assert all(audio == b"\x00" * 640 for audio in received)

    async def test_disabled_produces_nothing_while_idle(self) -> None:
        backend = make_backend()  # silence_fill=False
        received: list[bytes] = []
        backend.on_audio_received(lambda s, audio: received.append(audio))

        session, _client = await accept_fake(backend)
        await asyncio.sleep(0.1)
        await backend.disconnect(session)

        assert received == []


class TestLifecycle:
    async def test_disconnect_leaves_huddle_and_forgets_session(self) -> None:
        backend = make_backend()
        session, client = await accept_fake(backend)
        await backend.disconnect(session)
        assert client.left is True
        assert backend.get_session(session.id) is None
        assert backend.list_sessions("r") == []

    async def test_close_disconnects_all_sessions(self) -> None:
        backend = make_backend()
        s1 = make_session("s1")
        s2 = make_session("s2")
        c1, c2 = FakeHuddleClient(), FakeHuddleClient()
        await backend.accept(s1, c1)
        await backend.accept(s2, c2)
        await backend.close()
        assert c1.left and c2.left
        assert backend.list_sessions("r") == []
