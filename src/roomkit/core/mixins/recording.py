"""RecordingMixin — room-level media recording wiring."""

from __future__ import annotations

import array
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from roomkit.channels.voice import VoiceChannel
    from roomkit.recorder._room_recorder_manager import RoomRecorderManager
    from roomkit.voice.base import VoiceSession


@runtime_checkable
class RecordingHost(Protocol):
    """Contract: capabilities a host class must provide for RecordingMixin.

    Attributes provided by the host's ``__init__``:
        _room_recorder_mgr: Manager that coordinates room-level media recorders.
    """

    _room_recorder_mgr: RoomRecorderManager


class RecordingMixin:
    """Room-level audio and video recording tap wiring.

    Host contract: :class:`RecordingHost`.
    """

    _room_recorder_mgr: RoomRecorderManager

    @staticmethod
    def _make_audio_track(
        session_id: str,
        channel_id: str,
        participant_id: str | None,
        sample_rate: int = 16000,
    ) -> Any:
        """Create a RecordingTrack for audio."""
        from roomkit.recorder.base import RecordingTrack

        return RecordingTrack(
            id=f"audio:{session_id}",
            kind="audio",
            channel_id=channel_id,
            participant_id=participant_id,
            codec="pcm_s16le",
            sample_rate=sample_rate,
        )

    @staticmethod
    def _make_video_track(
        session_id: str,
        channel_id: str,
        participant_id: str | None,
    ) -> Any:
        """Create a RecordingTrack for video."""
        from roomkit.recorder.base import RecordingTrack

        return RecordingTrack(
            id=f"video:{session_id}",
            kind="video",
            channel_id=channel_id,
            participant_id=participant_id,
        )

    def _wire_audio_recording(
        self,
        room_id: str,
        channel_id: str,
        session: VoiceSession,
        channel: VoiceChannel,
    ) -> None:
        """Wire room-level audio recording tap on a VoiceChannel.

        Recording is opt-out: if the room has recorders, audio is recorded
        by default.  Set ``ChannelRecordingConfig(audio=False)`` on the
        channel to disable.
        """
        if not self._room_recorder_mgr.has_recorders(room_id):
            return
        if channel._recording is not None and not channel._recording.audio:
            return

        sample_rate = session.sample_rate if hasattr(session, "sample_rate") else 16000
        mgr = self._room_recorder_mgr

        track = self._make_audio_track(
            session.id,
            channel_id,
            session.participant_id,
            sample_rate=sample_rate,
        )
        mgr.on_track_added(room_id, track)

        # Ring buffer for outbound (TTS) audio.  The inbound tap runs on
        # the mic clock (~every 20 ms) and mixes in any pending outbound
        # samples so both directions share a single track / PTS timeline.
        outbound_buf = array.array("h")  # signed 16-bit ring buffer
        buf_lock = threading.Lock()
        max_outbound_samples = sample_rate * 5  # 5 s cap

        def _inbound_tap(sess: VoiceSession, frame: Any) -> None:
            data = frame.data
            n_samples = len(data) // 2

            with buf_lock:
                take = min(n_samples, len(outbound_buf))
                if take == 0:
                    # No outbound pending — pass inbound as-is (fast path)
                    mgr.on_data(room_id, track, data, time.monotonic() * 1000)
                    return
                out_samples = outbound_buf[:take]
                del outbound_buf[:take]

            # Mix inbound + outbound using array.array (avoids per-sample struct calls)
            in_arr = array.array("h", data)
            for i in range(take):
                in_arr[i] = max(-32768, min(32767, in_arr[i] + out_samples[i]))
            mgr.on_data(room_id, track, in_arr.tobytes(), time.monotonic() * 1000)

        def _outbound_tap(sess: VoiceSession, data: bytes, sample_rate: int) -> None:
            samples = array.array("h", data)
            with buf_lock:
                outbound_buf.extend(samples)
                overflow = len(outbound_buf) - max_outbound_samples
                if overflow > 0:
                    del outbound_buf[:overflow]

        channel.add_media_tap(_inbound_tap)
        channel.add_outbound_media_tap(_outbound_tap)

    def _make_video_recording_tap(
        self,
        room_id: str,
        track: Any,
    ) -> Any:
        """Build a video recording tap closure for the given room and track.

        The returned callable updates *track* metadata from the frame
        (codec, dimensions) and feeds data to the room recorder manager.
        """
        mgr = self._room_recorder_mgr

        def _video_tap(sess: Any, frame: Any) -> None:
            if not track.codec and hasattr(frame, "codec"):
                track.codec = frame.codec
            # Only copy dimensions from raw (uncompressed) frames —
            # encoded frames have meaningless defaults (640x480).
            # The recorder probes actual dimensions from the bitstream.
            if (
                track.width is None
                and hasattr(frame, "width")
                and not getattr(frame, "is_encoded", False)
            ):
                track.width = frame.width
                track.height = frame.height
            mgr.on_data(room_id, track, frame.data, time.monotonic() * 1000)

        return _video_tap

    def _wire_video_recording(
        self,
        room_id: str,
        channel_id: str,
        session: Any,
        channel: Any,
    ) -> None:
        """Wire room-level video recording tap on a VideoChannel.

        Recording is opt-out: if the room has recorders, video is recorded
        by default.  Set ``ChannelRecordingConfig(video=False)`` on the
        channel to disable.
        """
        if not self._room_recorder_mgr.has_recorders(room_id):
            return
        if channel._recording is not None and not channel._recording.video:
            return

        track = self._make_video_track(session.id, channel_id, session.participant_id)
        self._room_recorder_mgr.on_track_added(room_id, track)
        channel.add_media_tap(self._make_video_recording_tap(room_id, track))

    def _wire_backend_video_recording(
        self,
        room_id: str,
        channel_id: str,
        session: VoiceSession,
        backend: Any,
    ) -> None:
        """Wire room-level video recording tap directly on a VideoBackend.

        Used for combined A/V backends (e.g. SIPVideoBackend) where
        video frames come from the backend rather than a VideoChannel.
        """
        if not self._room_recorder_mgr.has_recorders(room_id):
            return

        track = self._make_video_track(session.id, channel_id, session.participant_id)
        self._room_recorder_mgr.on_track_added(room_id, track)
        backend.add_video_tap(self._make_video_recording_tap(room_id, track))

    def _wire_av_video_recording(
        self,
        room_id: str,
        channel_id: str,
        session: VoiceSession,
        channel: Any,
    ) -> None:
        """Wire room-level video recording via AudioVideoChannel tap.

        Recording is opt-out: if the room has recorders, video is recorded
        by default.  Set ``ChannelRecordingConfig(video=False)`` on the
        channel to disable.
        """
        if not self._room_recorder_mgr.has_recorders(room_id):
            return
        if channel._recording is not None and not channel._recording.video:
            return

        track = self._make_video_track(session.id, channel_id, session.participant_id)
        self._room_recorder_mgr.on_track_added(room_id, track)
        channel.add_video_media_tap(self._make_video_recording_tap(room_id, track))
