"""End-to-end: speech captured before a session exists reaches the provider.

This is the whole point of the shared capture source (RFC Section 12.12).  A
wake word is recognised only once the phrase is over, so by the time the
session opens the user has already said what they wanted.  The mark taken at
speech start, replayed on subscribe, lands in the channel's pre-connect buffer
and is flushed in order when the handshake completes.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from roomkit import RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.capture import MockCaptureSource
from roomkit.voice.realtime.mock import MockRealtimeProvider


def _mock_sounddevice() -> MagicMock:
    sd = MagicMock()
    sd.RawInputStream = MagicMock
    sd.CallbackStop = type("CallbackStop", (Exception,), {})
    return sd


def _backend(source: MockCaptureSource):  # noqa: ANN202 — test helper
    sd = _mock_sounddevice()
    with patch.dict("sys.modules", {"sounddevice": sd}):
        from roomkit.voice.backends.local import LocalAudioBackend

        backend = LocalAudioBackend(source=source, mute_mic_during_playback=False)
        backend._sd = sd
        return backend


@pytest.fixture
def source() -> MockCaptureSource:
    source = MockCaptureSource()
    source.start()
    return source


@pytest.fixture
def provider() -> MockRealtimeProvider:
    return MockRealtimeProvider()


@pytest.fixture
def channel(source: MockCaptureSource, provider: MockRealtimeProvider) -> RealtimeVoiceChannel:
    return RealtimeVoiceChannel(
        "rt-wake",
        provider=provider,
        transport=_backend(source),
        system_prompt="You are a test agent.",
    )


@pytest.fixture
async def room_id(channel: RealtimeVoiceChannel) -> str:
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "rt-wake")
    return room.id


async def _settle() -> None:
    """Let the loop drain the frames the capture thread scheduled."""
    for _ in range(5):
        await asyncio.sleep(0)
    await asyncio.sleep(0.01)


def _audio_sent(provider: MockRealtimeProvider) -> bytes:
    return b"".join(audio for _, audio in provider.sent_audio)


class TestBacklogReachesTheProvider:
    async def test_the_whole_utterance_arrives_in_order(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        source: MockCaptureSource,
        room_id: str,
    ) -> None:
        # The user starts talking; a detector marks the position at speech start.
        mark = source.mark()
        phrase = source.feed_blocks(6, fill=10)

        # Nothing is listening yet — no session exists.
        assert provider.sent_audio == []

        # The wake word matched. Open the session, replaying from the mark.
        await channel.start_session(room_id, "user-1", "conn", metadata={"capture_since": mark})
        await _settle()

        assert _audio_sent(provider) == b"".join(phrase)

    async def test_speech_before_the_mark_is_not_replayed(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        source: MockCaptureSource,
        room_id: str,
    ) -> None:
        """A stale tail of the previous conversation must not reach the model."""
        earlier = source.feed_blocks(4, fill=200)
        mark = source.mark()
        phrase = source.feed_blocks(3, fill=10)

        await channel.start_session(room_id, "user-1", "conn", metadata={"capture_since": mark})
        await _settle()

        sent = _audio_sent(provider)
        assert sent == b"".join(phrase)
        assert b"".join(earlier) not in sent

    async def test_live_audio_continues_after_the_replay(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        source: MockCaptureSource,
        room_id: str,
    ) -> None:
        mark = source.mark()
        phrase = source.feed_blocks(3, fill=10)

        await channel.start_session(room_id, "user-1", "conn", metadata={"capture_since": mark})
        await _settle()
        tail = source.feed_blocks(2, fill=90)
        await _settle()

        assert _audio_sent(provider) == b"".join(phrase + tail)

    async def test_without_a_mark_only_live_audio_is_sent(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        source: MockCaptureSource,
        room_id: str,
    ) -> None:
        before = source.feed_blocks(3, fill=200)

        await channel.start_session(room_id, "user-1", "conn")
        await _settle()
        live = source.feed_blocks(2, fill=10)
        await _settle()

        sent = _audio_sent(provider)
        assert sent == b"".join(live)
        assert b"".join(before) not in sent


class TestFailurePaths:
    async def test_a_failed_handshake_sends_nothing(
        self,
        source: MockCaptureSource,
        room_id: str,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
    ) -> None:
        """The retained audio is discarded with the session, never flushed."""
        mark = source.mark()
        source.feed_blocks(4, fill=10)

        async def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("handshake failed")

        provider.connect = _boom  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="handshake failed"):
            await channel.start_session(
                room_id, "user-1", "conn", metadata={"capture_since": mark}
            )
        await _settle()

        assert provider.sent_audio == []

    async def test_a_non_mark_in_metadata_is_ignored_not_fatal(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        source: MockCaptureSource,
        room_id: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Starting the call matters more than the backlog."""
        source.feed_blocks(2, fill=200)

        session = await channel.start_session(
            room_id, "user-1", "conn", metadata={"capture_since": "not-a-mark"}
        )
        await _settle()
        live = source.feed_blocks(1, fill=10)
        await _settle()

        assert session is not None
        assert _audio_sent(provider) == b"".join(live)
        assert "expected CaptureMark" in caplog.text


class TestSessionLifecycleAgainstTheSource:
    async def test_ending_a_session_leaves_the_source_running(
        self,
        channel: RealtimeVoiceChannel,
        source: MockCaptureSource,
        room_id: str,
    ) -> None:
        """A detector reattaching after the call must still find a live source."""
        session = await channel.start_session(room_id, "user-1", "conn")
        await channel.end_session(session)
        await _settle()

        assert source.started is True

        received: list[bytes] = []
        source.subscribe(lambda frame: received.append(frame.data), name="wakeword")
        sent = source.feed_blocks(2)
        assert received == sent

    async def test_the_session_stops_receiving_once_it_ends(
        self,
        channel: RealtimeVoiceChannel,
        provider: MockRealtimeProvider,
        source: MockCaptureSource,
        room_id: str,
    ) -> None:
        session = await channel.start_session(room_id, "user-1", "conn")
        await _settle()
        await channel.end_session(session)
        await _settle()

        before = len(provider.sent_audio)
        source.feed_blocks(3, fill=10)
        await _settle()

        assert len(provider.sent_audio) == before
