"""roomkit.voice.testing.wav: PCMAudio, the WAV round trip, framing, synthetics."""

from __future__ import annotations

from array import array

import pytest

from roomkit.voice.testing import PCMAudio, pcm_frames, read_wav, silence, tone, write_wav


class TestPCMAudio:
    def test_duration_and_frame_bytes(self) -> None:
        clip = silence(1000)

        assert clip.duration_ms == 1000
        assert len(clip.data) == 32_000
        assert clip.frame_bytes(20) == 640

    def test_rejects_misaligned_data(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            PCMAudio(data=b"\x00")

    @pytest.mark.parametrize(
        ("field", "value"),
        [("sample_rate", 0), ("channels", 3), ("sample_width", 3)],
    )
    def test_rejects_an_invalid_format(self, field: str, value: int) -> None:
        with pytest.raises(ValueError, match=field):
            PCMAudio(data=bytes(48), **{field: value})

    def test_concatenation_keeps_the_format_and_refuses_a_mismatch(self) -> None:
        clip = tone(100) + silence(100)

        assert clip.duration_ms == 200
        assert clip.sample_rate == 16000
        with pytest.raises(ValueError, match="different formats"):
            tone(100) + silence(100, sample_rate=8000)


class TestWavRoundTrip:
    def test_write_then_read_is_identity(self, tmp_path) -> None:
        clip = tone(250)

        path = write_wav(tmp_path / "nested" / "tone.wav", clip)

        assert path.exists()
        assert read_wav(path) == clip

    def test_stereo_and_rate_survive(self, tmp_path) -> None:
        clip = PCMAudio(data=bytes(8000 * 4), sample_rate=8000, channels=2)

        back = read_wav(write_wav(tmp_path / "stereo.wav", clip))

        assert back == clip
        assert back.duration_ms == 1000


class TestFrames:
    def test_one_second_at_16k_is_fifty_frames_of_640_bytes(self) -> None:
        frames = pcm_frames(tone(1000))

        assert len(frames) == 50
        assert {len(f.data) for f in frames} == {640}
        assert [f.timestamp_ms for f in frames[:3]] == [0.0, 20.0, 40.0]
        assert frames[0].sample_rate == 16000
        assert frames[0].channels == 1
        assert frames[0].sample_width == 2

    def test_the_last_frame_is_padded_with_silence(self) -> None:
        clip = tone(30)  # 960 bytes: one whole frame and a half

        frames = pcm_frames(clip)

        assert len(frames) == 2
        assert frames[1].data[:320] == clip.data[640:]
        assert frames[1].data[320:] == bytes(320)

    def test_eight_bit_padding_is_unsigned_silence(self) -> None:
        clip = PCMAudio(data=b"\x80" * 100, sample_rate=8000, sample_width=1)

        frames = pcm_frames(clip)  # 160 bytes per 20 ms frame

        assert len(frames) == 1
        assert frames[0].data[100:] == b"\x80" * 60

    def test_frame_ms_is_honoured(self) -> None:
        assert len(pcm_frames(tone(1000), frame_ms=10)) == 100

    def test_a_frame_shorter_than_one_sample_is_refused(self) -> None:
        with pytest.raises(ValueError, match="frame_ms"):
            pcm_frames(silence(20, sample_rate=100), frame_ms=1)


class TestSynthetics:
    def test_tone_is_loud_and_silence_is_flat(self) -> None:
        loud = array("h", tone(100, amplitude=0.5).data)
        flat = array("h", silence(100).data)

        assert max(abs(s) for s in loud) > 16_000
        assert not any(flat)

    def test_amplitude_out_of_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="amplitude"):
            tone(10, amplitude=1.5)
