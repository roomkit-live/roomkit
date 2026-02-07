"""Tests for EnergyVADProvider."""

from __future__ import annotations

import struct

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.vad.base import VADEventType
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider, _rms_int16

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(
    amplitude: int = 0,
    n_samples: int = 320,
    sample_rate: int = 16000,
) -> AudioFrame:
    """Create an AudioFrame filled with a constant int16 value."""
    data = struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))
    return AudioFrame(data=data, sample_rate=sample_rate)


def _silence(n_samples: int = 320, sample_rate: int = 16000) -> AudioFrame:
    return _make_frame(amplitude=0, n_samples=n_samples, sample_rate=sample_rate)


def _speech(
    amplitude: int = 1000,
    n_samples: int = 320,
    sample_rate: int = 16000,
) -> AudioFrame:
    return _make_frame(amplitude=amplitude, n_samples=n_samples, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Unit: _rms_int16
# ---------------------------------------------------------------------------


class TestRmsInt16:
    def test_silence(self) -> None:
        data = struct.pack("<4h", 0, 0, 0, 0)
        assert _rms_int16(data) == 0.0

    def test_constant_signal(self) -> None:
        data = struct.pack("<4h", 100, 100, 100, 100)
        assert _rms_int16(data) == 100.0

    def test_empty(self) -> None:
        assert _rms_int16(b"") == 0.0

    def test_odd_byte_count(self) -> None:
        # 5 bytes → 2 full samples (4 bytes), last byte ignored
        data = struct.pack("<2h", 200, 200) + b"\x00"
        assert _rms_int16(data) == 200.0


# ---------------------------------------------------------------------------
# Basic transitions
# ---------------------------------------------------------------------------


class TestBasicTransitions:
    def test_silence_produces_no_events(self) -> None:
        vad = EnergyVADProvider(energy_threshold=300)
        for _ in range(100):
            assert vad.process(_silence()) is None

    def test_speech_start(self) -> None:
        vad = EnergyVADProvider(energy_threshold=300)
        event = vad.process(_speech(1000))
        assert event is not None
        assert event.type == VADEventType.SPEECH_START

    def test_speech_end_after_silence(self) -> None:
        vad = EnergyVADProvider(
            energy_threshold=300,
            silence_threshold_ms=100,
            min_speech_duration_ms=0,
        )
        # Start speech
        event = vad.process(_speech())
        assert event is not None
        assert event.type == VADEventType.SPEECH_START

        # Continue speaking for a few frames
        for _ in range(5):
            assert vad.process(_speech()) is None

        # Silence until SPEECH_END (100ms = 5 frames at 20ms each)
        end_event = None
        for _ in range(10):
            ev = vad.process(_silence())
            if ev is not None:
                end_event = ev
                break

        assert end_event is not None
        assert end_event.type == VADEventType.SPEECH_END
        assert end_event.audio_bytes is not None
        assert len(end_event.audio_bytes) > 0
        assert end_event.duration_ms is not None
        assert end_event.duration_ms > 0


# ---------------------------------------------------------------------------
# Audio accumulation
# ---------------------------------------------------------------------------


class TestAudioAccumulation:
    def test_accumulated_audio_contains_speech_frames(self) -> None:
        vad = EnergyVADProvider(
            energy_threshold=300,
            silence_threshold_ms=60,
            min_speech_duration_ms=0,
            speech_pad_ms=0,
        )
        # Start speech
        vad.process(_speech(1000))
        # 3 more speech frames
        for _ in range(3):
            vad.process(_speech(1000))

        # Enough silence to end
        end_event = None
        for _ in range(10):
            ev = vad.process(_silence())
            if ev is not None:
                end_event = ev
                break

        assert end_event is not None
        # 1 start frame + 3 speech frames + silence frames until end
        assert len(end_event.audio_bytes) >= 320 * 2 * 4  # at least 4 frames


# ---------------------------------------------------------------------------
# Min speech duration filtering
# ---------------------------------------------------------------------------


class TestMinSpeechDuration:
    def test_short_speech_discarded(self) -> None:
        vad = EnergyVADProvider(
            energy_threshold=300,
            silence_threshold_ms=40,  # 2 frames
            min_speech_duration_ms=500,  # much longer than we'll speak
            speech_pad_ms=0,
        )
        # Single speech frame → SPEECH_START
        event = vad.process(_speech())
        assert event is not None
        assert event.type == VADEventType.SPEECH_START

        # Immediate silence → should eventually discard (no SPEECH_END)
        events = []
        for _ in range(20):
            ev = vad.process(_silence())
            if ev is not None:
                events.append(ev)

        # No SPEECH_END because speech was too short
        assert all(e.type != VADEventType.SPEECH_END for e in events)

    def test_long_enough_speech_emitted(self) -> None:
        vad = EnergyVADProvider(
            energy_threshold=300,
            silence_threshold_ms=60,
            min_speech_duration_ms=100,  # 5 frames at 20ms
            speech_pad_ms=0,
        )
        # Start
        vad.process(_speech())
        # 10 more frames (220ms total > 100ms min)
        for _ in range(10):
            vad.process(_speech())

        end_event = None
        for _ in range(20):
            ev = vad.process(_silence())
            if ev is not None:
                end_event = ev
                break

        assert end_event is not None
        assert end_event.type == VADEventType.SPEECH_END


# ---------------------------------------------------------------------------
# Pre-roll buffer
# ---------------------------------------------------------------------------


class TestPreRoll:
    def test_pre_roll_included_in_audio(self) -> None:
        vad = EnergyVADProvider(
            energy_threshold=300,
            silence_threshold_ms=60,
            min_speech_duration_ms=0,
            speech_pad_ms=100,  # 5 frames worth
        )
        # Feed 10 silence frames (they go into pre-roll buffer)
        for _ in range(10):
            vad.process(_silence())

        # Now speech
        vad.process(_speech())

        # End with silence
        end_event = None
        for _ in range(20):
            ev = vad.process(_silence())
            if ev is not None:
                end_event = ev
                break

        assert end_event is not None
        # Audio should include pre-roll frames + the speech frame + silence frames
        # Pre-roll is ~100ms = 5 frames of 20ms each = 5*640 = 3200 bytes
        # Plus speech frame = 640 bytes, plus some silence frames
        assert len(end_event.audio_bytes) > 320 * 2  # more than just speech frame


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_state(self) -> None:
        vad = EnergyVADProvider(energy_threshold=300)

        # Start speaking
        event = vad.process(_speech())
        assert event is not None
        assert event.type == VADEventType.SPEECH_START

        # Reset mid-speech
        vad.reset()

        # Should be back to idle — next speech frame triggers SPEECH_START again
        event = vad.process(_speech())
        assert event is not None
        assert event.type == VADEventType.SPEECH_START


# ---------------------------------------------------------------------------
# name property
# ---------------------------------------------------------------------------


class TestName:
    def test_name(self) -> None:
        vad = EnergyVADProvider()
        assert vad.name == "EnergyVADProvider"


# ---------------------------------------------------------------------------
# Multiple utterances
# ---------------------------------------------------------------------------


class TestMultipleUtterances:
    def test_two_utterances(self) -> None:
        vad = EnergyVADProvider(
            energy_threshold=300,
            silence_threshold_ms=60,
            min_speech_duration_ms=0,
            speech_pad_ms=0,
        )

        events = []

        # First utterance
        events.append(vad.process(_speech()))
        for _ in range(5):
            vad.process(_speech())
        for _ in range(20):
            ev = vad.process(_silence())
            if ev is not None:
                events.append(ev)
                break

        # Second utterance
        events.append(vad.process(_speech()))
        for _ in range(5):
            vad.process(_speech())
        for _ in range(20):
            ev = vad.process(_silence())
            if ev is not None:
                events.append(ev)
                break

        types = [e.type for e in events if e is not None]
        assert types == [
            VADEventType.SPEECH_START,
            VADEventType.SPEECH_END,
            VADEventType.SPEECH_START,
            VADEventType.SPEECH_END,
        ]
