"""The one track the AI speaks on, in one conference.

Its own module because it owns state nothing else in the session reads — the
source, the track it is published on, the format both are pinned to, and whether
an utterance is open — and because the contract it keeps is the subtlest one in
this backend: what a format change means, what an empty chunk means, and what
becomes of an utterance when the session goes away underneath it.

One track per conference, heard by everyone. Per-participant audio is not a
thing here: the AI is synthesized once and published once.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from roomkit.conference._livekit_mapping import SAMPLE_WIDTH, require_publishable_pcm
from roomkit.voice.base import AudioChunk

logger = logging.getLogger("roomkit.conference.livekit")


class BotVoiceTrack:
    """The bot's outbound audio track, published on first need."""

    def __init__(
        self,
        *,
        rtc: Any,
        room: Any,
        identity: str,
        room_id: str,
        queue_ms: int,
    ) -> None:
        self._rtc = rtc
        self._room = room
        self._identity = identity
        self._room_id = room_id
        self._queue_ms = queue_ms
        self._lock = asyncio.Lock()
        self._source: Any | None = None
        self._track: Any | None = None
        self._format: tuple[int, int] | None = None
        self._utterance_open = False

    async def publish(self, chunk: AudioChunk) -> None:
        """Put one chunk of the AI's speech on the track.

        ``capture_frame`` waits when the source's queue is full, so this applies
        real backpressure: the framework is paced by the SFU rather than running
        ahead of it.

        A chunk carrying no audio only declares a boundary — there is nothing to
        play. The queue is deliberately *not* flushed on it: an utterance cut
        short and one that ended by itself both close this way, the two are
        indistinguishable from here, and flushing would truncate real speech in
        the second case. What bounds the audio still queued behind a barge-in is
        the queue's own size, because the interface has no way to say "stop
        talking".
        """
        require_publishable_pcm(chunk)
        if chunk.data:
            source = await self._ensure_source(chunk)
            await source.capture_frame(self._frame(chunk))
        self._utterance_open = not chunk.is_final

    def abandon_utterance(self) -> bool:
        """Take an in-flight utterance as ended, and say whether there was one.

        The two ways a session stops — the channel leaving, and the SFU dropping
        it — end an utterance the same way and for the same reason: the session a
        terminal chunk would name is gone, so no boundary can be published into
        it, any more than a crashed process could publish one (RFC section
        12.10.4). What differs is only how surprising it was, which is the
        caller's to say.
        """
        if not self._utterance_open:
            return False
        self._utterance_open = False
        return True

    async def close(self) -> None:
        """Unpublish the track and release the source. Idempotent."""
        source, self._source = self._source, None
        track, self._track = self._track, None
        self._format = None
        if track is not None:
            with contextlib.suppress(Exception):
                await self._room.local_participant.unpublish_track(track.sid)
        if source is not None:
            with contextlib.suppress(Exception):
                await source.aclose()

    def _frame(self, chunk: AudioChunk) -> Any:
        frame_align = SAMPLE_WIDTH * chunk.channels
        if len(chunk.data) % frame_align != 0:
            raise ValueError(
                f"a chunk of {len(chunk.data)} bytes is not a whole number of "
                f"{chunk.channels}-channel 16-bit samples, so LiveKit would publish a "
                "frame shifted by part of one"
            )
        return self._rtc.AudioFrame(
            data=chunk.data,
            sample_rate=chunk.sample_rate,
            num_channels=chunk.channels,
            samples_per_channel=len(chunk.data) // frame_align,
        )

    async def _ensure_source(self, chunk: AudioChunk) -> Any:
        """The audio source, created from the first chunk that carries audio.

        Taken from the chunk rather than configured, because the synthesizer's
        rate is the one fact available and resampling it here is exactly what a
        transport must not do. LiveKit's own encoder takes it from there.

        A later chunk in another format is refused rather than reinterpreted: an
        ``rtc.AudioSource`` is fixed at construction, and republishing the track
        mid-conversation to follow a format change would drop the bot's voice
        out of the conference for as long as renegotiation takes.
        """
        wanted = (chunk.sample_rate, chunk.channels)
        async with self._lock:
            if self._source is not None:
                self._require_same_format(wanted)
                return self._source
            source = self._rtc.AudioSource(
                chunk.sample_rate, chunk.channels, queue_size_ms=self._queue_ms
            )
            track = self._rtc.LocalAudioTrack.create_audio_track(f"{self._identity}-voice", source)
            options = self._rtc.TrackPublishOptions(source=self._rtc.TrackSource.SOURCE_MICROPHONE)
            await self._room.local_participant.publish_track(track, options)
            self._source, self._track, self._format = source, track, wanted
            logger.debug(
                "Conference bot %s published its voice track in room %s at %d Hz / %d ch",
                self._identity,
                self._room_id,
                chunk.sample_rate,
                chunk.channels,
            )
            return source

    def _require_same_format(self, wanted: tuple[int, int]) -> None:
        if self._format == wanted:
            return
        published = self._format or ("?", "?")
        raise ValueError(
            f"the bot's track in room {self._room_id!r} publishes {published[0]} Hz / "
            f"{published[1]} ch and this chunk is {wanted[0]} Hz / {wanted[1]} ch. A "
            "source's format is fixed once published, and resampling belongs to the lane."
        )
