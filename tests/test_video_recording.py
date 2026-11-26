"""Tests for video recording — VideoRecorder ABC, Mock, and channel integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from roomkit import (
    RoomKit,
    VideoChannel,
)
from roomkit.video import MockVideoBackend, VideoFrame, VideoSession, VideoSessionState
from roomkit.video.pipeline import VideoPipelineConfig
from roomkit.video.recorder import (
    MockVideoRecorder,
    OpenCVVideoRecorder,
    PyAVVideoRecorder,
    VideoRecorder,
    VideoRecordingConfig,
)

# The modules themselves, only to inject faults into what they wrap.
from roomkit.video.recorder import opencv as opencv_recorder
from roomkit.video.recorder import pyav as pyav_recorder
from roomkit.voice.pipeline import RecordingEncryption


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


class TestMockVideoRecorder:
    def test_start_stop(self) -> None:
        from roomkit.video.base import VideoSession

        recorder = MockVideoRecorder()
        session = VideoSession(id="s1", room_id="r1", participant_id="u1", channel_id="v1")
        config = VideoRecordingConfig(storage="/tmp/test")

        handle = recorder.start(session, config)
        assert handle.state == "recording"
        assert handle.session_id == "s1"
        assert handle.id

        result = recorder.stop(handle)
        assert handle.state == "stopped"
        assert result.frame_count == 0
        assert result.id == handle.id

    def test_tap_frame(self) -> None:
        from roomkit.video.base import VideoSession

        recorder = MockVideoRecorder()
        session = VideoSession(id="s1", room_id="r1", participant_id="u1", channel_id="v1")
        handle = recorder.start(session, VideoRecordingConfig())

        frame = VideoFrame(data=b"\x00" * 100, codec="h264")
        recorder.tap_frame(handle, frame)
        recorder.tap_frame(handle, frame)

        result = recorder.stop(handle)
        assert result.frame_count == 2

    def test_multiple_recordings(self) -> None:
        from roomkit.video.base import VideoSession

        recorder = MockVideoRecorder()
        s1 = VideoSession(id="s1", room_id="r1", participant_id="u1", channel_id="v1")
        s2 = VideoSession(id="s2", room_id="r1", participant_id="u2", channel_id="v1")

        h1 = recorder.start(s1, VideoRecordingConfig())
        h2 = recorder.start(s2, VideoRecordingConfig())

        assert h1.id != h2.id
        assert len(recorder.recordings) == 2


class TestVideoChannelRecording:
    @pytest.mark.parametrize(
        ("recording_config", "started"), [(VideoRecordingConfig(), 1), (None, 0)]
    )
    async def test_recording_starts_on_bind(
        self, kit: RoomKit, recording_config: VideoRecordingConfig | None, started: int
    ) -> None:
        """The recorder starts for a session that binds, given a recording
        config; without one, it is not started (RFC §12.8.4)."""
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        ch = VideoChannel(
            "video-1",
            backend=backend,
            pipeline=VideoPipelineConfig(recorder=recorder, recording_config=recording_config),
        )
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        await kit.connect_video("r1", "user-1", "video-1")

        assert len(recorder.recordings) == started

    async def test_recording_stops_on_unbind(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        ch = VideoChannel(
            "video-1",
            backend=backend,
            pipeline=VideoPipelineConfig(
                recorder=recorder,
                recording_config=VideoRecordingConfig(),
            ),
        )
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")
        await kit.disconnect_video(session)

        assert session.id not in ch._recording_handles
        # Recording was stopped
        rec = list(recorder.recordings.values())[0]
        assert rec.handle.state == "stopped"

    async def test_frames_tapped_to_recorder(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        ch = VideoChannel(
            "video-1",
            backend=backend,
            pipeline=VideoPipelineConfig(
                recorder=recorder,
                recording_config=VideoRecordingConfig(),
            ),
        )
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")

        # Send frames
        for i in range(5):
            frame = VideoFrame(data=b"\x00" * 100, codec="h264", timestamp_ms=float(i * 100))
            await backend.simulate_video_received(session, frame)

        rec = list(recorder.recordings.values())[0]
        assert len(rec.frames) == 5

    async def test_no_recorder_no_error(self, kit: RoomKit) -> None:
        """Channel works fine without a recorder."""
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")
        frame = VideoFrame(data=b"\x00" * 100, codec="h264")
        await backend.simulate_video_received(session, frame)
        await kit.disconnect_video(session)

    async def test_close_stops_recordings(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        ch = VideoChannel(
            "video-1",
            backend=backend,
            pipeline=VideoPipelineConfig(
                recorder=recorder,
                recording_config=VideoRecordingConfig(),
            ),
        )
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        await kit.connect_video("r1", "user-1", "video-1")
        await ch.close()

        assert ch._recording_handles == {}
        rec = list(recorder.recordings.values())[0]
        assert rec.handle.state == "stopped"


# -- Encryption at rest (RFC §17.6, RMK-568) ---------------------------------

_CIPHER_HEADER = b"XOR1"


class _XorEncryption(RecordingEncryption):
    """Stands for the integrator's cipher: writes ``<path>.enc``, deletes the plaintext."""

    def __init__(self) -> None:
        self.plaintexts: list[bytes] = []

    @property
    def name(self) -> str:
        return "xor"

    def encrypt_file(self, path: str) -> str:
        source = Path(path)
        plaintext = source.read_bytes()
        self.plaintexts.append(plaintext)
        target = source.with_name(source.name + ".enc")
        target.write_bytes(_CIPHER_HEADER + bytes(b ^ 0x5A for b in plaintext))
        source.unlink()
        return str(target)


class _Faults:
    """How many of the next files fail as they are closed, once written."""

    def __init__(self) -> None:
        self.remaining = 0

    def after_close(self) -> None:
        if self.remaining:
            self.remaining -= 1
            raise OSError("disk full")


class _FileVideoWriter:
    """``cv2.VideoWriter`` as the recorder drives it: each frame's bytes go to the file."""

    def __init__(self, path: str, faults: _Faults) -> None:
        self._file = open(path, "wb")  # noqa: SIM115 - released by release()
        self._faults = faults

    def isOpened(self) -> bool:  # noqa: N802
        return not self._file.closed

    def write(self, frame: Any) -> None:
        self._file.write(frame.tobytes())

    def release(self) -> None:
        self._file.close()
        self._faults.after_close()


class _FileWritingCV2:
    """The part of ``cv2`` the OpenCV recorder uses, writing real files."""

    COLOR_RGB2BGR = 4

    def __init__(self, faults: _Faults) -> None:
        self._faults = faults

    def VideoWriter_fourcc(self, *args: str) -> int:  # noqa: N802
        return 0

    def VideoWriter(  # noqa: N802
        self, path: str, fourcc: int, fps: float, size: tuple[int, int]
    ) -> _FileVideoWriter:
        return _FileVideoWriter(path, self._faults)

    def cvtColor(self, src: Any, code: int) -> Any:  # noqa: N802
        return src


@dataclass
class _FileRecorder:
    recorder: VideoRecorder
    codec: str
    faults: _Faults

    def fail_closing(self, count: int) -> None:
        """Make the next *count* files raise as they are closed."""
        self.faults.remaining = count


@pytest.fixture(params=["opencv", "pyav"])
def file_recorder(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> _FileRecorder:
    """A recorder that writes files, the codec to ask it for, and its faults."""
    faults = _Faults()
    if request.param == "pyav":
        pytest.importorskip("av", reason="av (PyAV) not installed")
        flush_and_close: Callable[[Any], None] = pyav_recorder._flush_and_close

        def failing_flush_and_close(active: Any) -> None:
            flush_and_close(active)
            faults.after_close()

        monkeypatch.setattr(pyav_recorder, "_flush_and_close", failing_flush_and_close)
        return _FileRecorder(PyAVVideoRecorder(), "libx264", faults)
    monkeypatch.setattr(opencv_recorder, "_import_cv2", lambda: _FileWritingCV2(faults))
    return _FileRecorder(OpenCVVideoRecorder(), "mp4v", faults)


def _raw_frame(value: int) -> VideoFrame:
    return VideoFrame(data=bytes([value]) * 64 * 48 * 3, codec="raw_rgb24", width=64, height=48)


def _video_session(sid: str) -> VideoSession:
    return VideoSession(id=sid, room_id="r1", participant_id="u1", channel_id="video-1")


class TestRecordingEncryptionAtRest:
    async def test_session_video_configured_for_encryption_is_stored_as_ciphertext(
        self, file_recorder: _FileRecorder, tmp_path: Path
    ) -> None:
        """A video channel's recording is handed to ``encryption`` when it ends,
        and only the cipher's artifact stays on the storage (RFC §17.6)."""
        recorder, codec = file_recorder.recorder, file_recorder.codec
        cipher = _XorEncryption()
        config = VideoRecordingConfig(storage=str(tmp_path), codec=codec, encryption=cipher)
        kit, backend = RoomKit(), MockVideoBackend()
        kit.register_channel(
            VideoChannel(
                "video-1",
                backend=backend,
                pipeline=VideoPipelineConfig(recorder=recorder, recording_config=config),
            )
        )
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.join("r1", "video-1", participant_id="u1")
        for i in range(3):
            await backend.simulate_video_received(session, _raw_frame(i))
        await kit.leave(session)
        await kit.close()

        stored = sorted(tmp_path.iterdir())
        assert [p.suffix for p in stored] == [".enc"]
        assert len(cipher.plaintexts) == 1 and cipher.plaintexts[0]
        plaintext = cipher.plaintexts[0]
        assert stored[0].read_bytes() == _CIPHER_HEADER + bytes(b ^ 0x5A for b in plaintext)

    def test_result_names_the_ciphertext_and_close_encrypts_an_open_recording(
        self, file_recorder: _FileRecorder, tmp_path: Path
    ) -> None:
        """``stop()`` reports the encrypted artifact; ``close()`` encrypts a
        recording still open, which would otherwise stay in the clear."""
        recorder, codec = file_recorder.recorder, file_recorder.codec
        config = VideoRecordingConfig(
            storage=str(tmp_path), codec=codec, encryption=_XorEncryption()
        )
        stopped = recorder.start(_video_session("stopped-session"), config)
        left_open = recorder.start(_video_session("open-session"), config)
        for handle in (stopped, left_open):
            recorder.tap_frame(handle, _raw_frame(7))

        result = recorder.stop(stopped)
        recorder.close()

        assert result.url == stopped.path + ".enc"
        assert result.size_bytes == Path(result.url).stat().st_size
        assert result.metadata == {"encryption": "xor"}
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
            Path(h.path).name + ".enc" for h in (stopped, left_open)
        )

    @pytest.mark.parametrize(
        ("storage_encrypted_at_rest", "metadata"),
        [(False, {}), (True, {"encryption": "storage"})],
    )
    def test_without_encryption_the_file_stays_as_written(
        self,
        file_recorder: _FileRecorder,
        tmp_path: Path,
        storage_encrypted_at_rest: bool,
        metadata: dict[str, object],
    ) -> None:
        """No ``encryption``: the recording is stored as it was written, and the
        result claims encryption only when the storage was declared to apply it."""
        recorder, codec = file_recorder.recorder, file_recorder.codec
        config = VideoRecordingConfig(
            storage=str(tmp_path), codec=codec, storage_encrypted_at_rest=storage_encrypted_at_rest
        )
        handle = recorder.start(_video_session("s1"), config)
        recorder.tap_frame(handle, _raw_frame(7))

        result = recorder.stop(handle)
        recorder.close()

        assert result.url == handle.path
        assert [str(p) for p in tmp_path.iterdir()] == [handle.path]
        assert result.size_bytes == Path(handle.path).stat().st_size > 0
        assert result.metadata == metadata

    def test_a_file_that_fails_to_close_is_still_encrypted(
        self, file_recorder: _FileRecorder, tmp_path: Path
    ) -> None:
        """A recording whose file raises as it is closed still goes through the
        cipher: ``stop()`` raises, and no plaintext stays on the storage."""
        recorder, codec = file_recorder.recorder, file_recorder.codec
        config = VideoRecordingConfig(
            storage=str(tmp_path), codec=codec, encryption=_XorEncryption()
        )
        handle = recorder.start(_video_session("s1"), config)
        recorder.tap_frame(handle, _raw_frame(7))
        file_recorder.fail_closing(1)

        with pytest.raises(OSError, match="disk full"):
            recorder.stop(handle)

        assert [p.name for p in tmp_path.iterdir()] == [Path(handle.path).name + ".enc"]

    def test_close_goes_on_past_a_recording_that_fails(
        self,
        file_recorder: _FileRecorder,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``close()`` stores each open recording: the first one failing to close
        is logged, encrypted all the same, and the next one is encrypted too."""
        recorder, codec = file_recorder.recorder, file_recorder.codec
        config = VideoRecordingConfig(
            storage=str(tmp_path), codec=codec, encryption=_XorEncryption()
        )
        handles = [recorder.start(_video_session(sid), config) for sid in ("first", "second")]
        for handle in handles:
            recorder.tap_frame(handle, _raw_frame(7))
        file_recorder.fail_closing(1)

        recorder.close()

        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
            Path(h.path).name + ".enc" for h in handles
        )
        assert "disk full" in caplog.text


class TestVideoChannelRecordingFailures:
    """A recorder that fails leaves the channel's other sessions as configured."""

    @staticmethod
    async def _kit_recording(
        recorder: VideoRecorder, config: VideoRecordingConfig
    ) -> tuple[RoomKit, MockVideoBackend, VideoChannel]:
        kit, backend = RoomKit(), MockVideoBackend()
        channel = VideoChannel(
            "video-1",
            backend=backend,
            pipeline=VideoPipelineConfig(recorder=recorder, recording_config=config),
        )
        kit.register_channel(channel)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")
        return kit, backend, channel

    async def test_kit_close_encrypts_every_recording_when_one_fails_to_stop(
        self, file_recorder: _FileRecorder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``kit.close()`` stops each open recording and closes the recorder even
        when one ``stop()`` raises: the next ones would otherwise stay in the
        clear. The failure is raised once the channel is closed."""
        recorder = file_recorder.recorder
        config = VideoRecordingConfig(
            storage=str(tmp_path), codec=file_recorder.codec, encryption=_XorEncryption()
        )
        kit, backend, _ = await self._kit_recording(recorder, config)
        for participant in ("u1", "u2"):
            session = await kit.join("r1", "video-1", participant_id=participant)
            await backend.simulate_video_received(session, _raw_frame(7))
        # Either guard alone would still leave only ciphertext, since the
        # recorder's close() stores an open recording too: count the calls.
        calls: list[str] = []
        stop, close = recorder.stop, recorder.close
        monkeypatch.setattr(recorder, "stop", lambda h: calls.append("stop") or stop(h))
        monkeypatch.setattr(recorder, "close", lambda: calls.append("close") or close())
        file_recorder.fail_closing(1)

        with pytest.raises(ExceptionGroup) as raised:
            await kit.close()

        assert raised.group_contains(OSError, match="disk full")
        assert calls == ["stop", "stop", "close"]
        assert [p.suffix for p in tmp_path.iterdir()] == [".enc", ".enc"]

    async def test_a_recording_that_fails_to_stop_does_not_keep_its_session_bound(
        self, file_recorder: _FileRecorder, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``kit.leave()`` unbinds the session even when its recording's
        ``stop()`` raises: the failure is logged, the session receives no more
        frames, and the file is stored as configured."""
        config = VideoRecordingConfig(
            storage=str(tmp_path), codec=file_recorder.codec, encryption=_XorEncryption()
        )
        kit, backend, channel = await self._kit_recording(file_recorder.recorder, config)
        tapped: list[VideoFrame] = []
        channel.add_media_tap(lambda _session, frame: tapped.append(frame))
        session = await kit.join("r1", "video-1", participant_id="u1")
        await backend.simulate_video_received(session, _raw_frame(7))
        file_recorder.fail_closing(1)

        await kit.leave(session)
        await backend.simulate_video_received(session, _raw_frame(8))
        await kit.close()

        assert len(tapped) == 1
        assert "disk full" in caplog.text
        assert [p.suffix for p in tmp_path.iterdir()] == [".enc"]

    async def test_a_session_the_kit_made_for_a_refused_recording_is_hung_up(self) -> None:
        """A join whose recording the recorder refuses hangs up the session it
        made, as a join the full bridge refuses (RFC §12.8.9, §12.8.10)."""

        class _Refusing(MockVideoRecorder):
            def start(self, session: VideoSession, config: VideoRecordingConfig) -> Any:
                raise ValueError("storage refused")

        kit, backend, _ = await self._kit_recording(_Refusing(), VideoRecordingConfig())

        with pytest.raises(ValueError, match="storage refused"):
            await kit.join("r1", "video-1", participant_id="u1")

        made = next(s for s in backend.list_sessions("r1") if s.participant_id == "u1")
        assert made.state == VideoSessionState.ENDED
        await kit.close()

    @pytest.mark.parametrize(
        ("storage", "fmt", "refusal"),
        [
            ("a/../b", "mp4", "must not contain '..'"),
            ("rec", "mp4/../../escaped", "plain file extension"),
        ],
    )
    async def test_a_refused_recording_refuses_the_session_before_binding_it(
        self, file_recorder: _FileRecorder, tmp_path: Path, storage: str, fmt: str, refusal: str
    ) -> None:
        """A recording the recorder refuses to start fails ``kit.join()`` before
        the session is bound to the room: it receives no frames, and nothing is
        written. Refused: a ``storage`` with '..', and a ``format`` that would
        name a file outside the storage (RFC §12.11, RMK-2067)."""
        config = VideoRecordingConfig(
            storage=str(tmp_path / storage), format=fmt, codec=file_recorder.codec
        )
        kit, backend, channel = await self._kit_recording(file_recorder.recorder, config)
        tapped: list[VideoFrame] = []
        channel.add_media_tap(lambda _session, frame: tapped.append(frame))
        session = await backend.connect("r1", "u1", "video-1")

        with pytest.raises(ValueError, match=refusal):
            await kit.join("r1", "video-1", session=session)
        await backend.simulate_video_received(session, _raw_frame(7))
        await kit.close()

        assert tapped == []
        assert list(tmp_path.rglob("*")) == []
