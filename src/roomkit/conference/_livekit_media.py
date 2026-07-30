"""Frames from a subscribed LiveKit track into the framework's own types.

Split out from the session that starts them because it is the one part of a bot
connection that touches no session state: a track goes in, framework frames come
out, and nothing is decided along the way that anything else depends on. Which
also means the format declaration — the whole point of having a real backend —
can be read here without a conference to hold it.

One task per subscribed track, so a lane doing its work inline delays that
track's frames and nobody else's. That is the isolation RFC section 12.10.4 makes
checkable from outside.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from roomkit.conference._livekit_mapping import SAMPLE_WIDTH, codec_for_buffer_type
from roomkit.conference.models import ConferenceTrack
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger("roomkit.conference.livekit")

AudioSink = Callable[[ConferenceTrack, AudioFrame], Awaitable[None]]
VideoSink = Callable[[ConferenceTrack, VideoFrame], Awaitable[None]]


async def pump_audio(
    *,
    rtc: Any,
    track: Any,
    record: ConferenceTrack,
    sink: AudioSink,
    sample_rate: int,
    channels: int,
) -> None:
    """Deliver a track's decoded audio, in the format it was asked for.

    The rate and channel count are what this backend requested of LiveKit's
    decoder, and every frame *declares* them. Nothing is resampled: normalising
    to what a recognizer wants belongs to the lane, and a transport that did it
    would hide from the pipeline the one thing this backend exists to hand it —
    audio the framework did not manufacture.

    ``timestamp_ms`` counts the samples that have gone by, so it is a clock that
    advances with the audio rather than with the wall. It starts at zero when the
    subscription does, which is the only origin available here: a track the
    framework unsubscribed and took again is a new stream to this pump.
    """
    stream = rtc.AudioStream.from_track(
        track=track, sample_rate=sample_rate, num_channels=channels
    )
    delivered = 0
    try:
        async for event in stream:
            frame = event.frame
            await sink(
                record,
                AudioFrame(
                    data=bytes(frame.data),
                    sample_rate=frame.sample_rate,
                    channels=frame.num_channels,
                    sample_width=SAMPLE_WIDTH,
                    timestamp_ms=delivered * 1000 / frame.sample_rate,
                ),
            )
            delivered += frame.samples_per_channel
    finally:
        with contextlib.suppress(Exception):
            await stream.aclose()


async def pump_video(
    *,
    rtc: Any,
    track: Any,
    record: ConferenceTrack,
    sink: VideoSink,
) -> None:
    """Deliver a track's video as raw I420.

    One layout is requested rather than taking whatever the decoder emits, so
    that what arrives is the same shape whichever codec the publisher
    negotiated. Each frame still declares its own: a decoder that answered with
    something else must not be described as I420.
    """
    stream = rtc.VideoStream(track=track, format=rtc.VideoBufferType.I420)
    sequence = 0
    try:
        async for event in stream:
            frame = event.frame
            try:
                codec = codec_for_buffer_type(rtc.VideoBufferType.Name(frame.type))
            except ValueError:
                logger.warning(
                    "Dropping a video frame on conference track %s in room %s",
                    record.id,
                    record.room_id,
                    exc_info=True,
                )
                continue
            await sink(
                record,
                VideoFrame(
                    data=bytes(frame.data),
                    codec=codec,
                    width=frame.width,
                    height=frame.height,
                    timestamp_ms=event.timestamp_us / 1000,
                    sequence=sequence,
                ),
            )
            sequence += 1
    finally:
        with contextlib.suppress(Exception):
            await stream.aclose()
