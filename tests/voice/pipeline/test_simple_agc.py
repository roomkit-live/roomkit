"""Tests for the built-in PCM16 automatic gain controller."""

from __future__ import annotations

import math
import struct

import pytest

from roomkit import SimpleAGCProvider as RootSimpleAGCProvider
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.agc import AGCConfig, SimpleAGCProvider


def _frame(
    value: int,
    *,
    samples: int = 320,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> AudioFrame:
    data = struct.pack(f"<{samples * channels}h", *([value] * samples * channels))
    return AudioFrame(
        data=data,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )


def _rms(frame: AudioFrame) -> float:
    samples = struct.unpack(f"<{len(frame.data) // 2}h", frame.data)
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


class TestSimpleAGCProvider:
    def test_exported_from_root_package(self) -> None:
        assert RootSimpleAGCProvider is SimpleAGCProvider

    def test_quiet_speech_is_amplified_up_to_max_gain(self) -> None:
        provider = SimpleAGCProvider(
            AGCConfig(
                target_level_dbfs=-12.0,
                max_gain_db=12.0,
                attack_ms=0.0,
                release_ms=0.0,
            )
        )
        source = _frame(1000)

        result = provider.process(source, "alice")

        assert _rms(result) == pytest.approx(_rms(source) * 10 ** (12 / 20), rel=0.01)
        assert result.metadata["gain_applied_db"] == pytest.approx(12.0)

    def test_loud_speech_is_attenuated_toward_target(self) -> None:
        provider = SimpleAGCProvider(
            AGCConfig(target_level_dbfs=-12.0, attack_ms=0.0, release_ms=0.0)
        )

        result = provider.process(_frame(30000), "alice")

        result_dbfs = 20 * math.log10(_rms(result) / 32768.0)
        assert result_dbfs == pytest.approx(-12.0, abs=0.1)
        assert result.metadata["gain_applied_db"] < 0

    def test_limiter_prevents_int16_clipping(self) -> None:
        provider = SimpleAGCProvider(
            AGCConfig(target_level_dbfs=0.0, max_gain_db=30.0, attack_ms=0.0)
        )

        result = provider.process(_frame(20000), "alice")
        samples = struct.unpack(f"<{len(result.data) // 2}h", result.data)

        assert max(samples) <= 32767
        assert min(samples) >= -32768

    def test_gain_state_is_isolated_per_stream(self) -> None:
        provider = SimpleAGCProvider(
            AGCConfig(target_level_dbfs=-12.0, max_gain_db=20.0, attack_ms=0.0, release_ms=0.0)
        )

        provider.process(_frame(500), "quiet")
        provider.process(_frame(30000), "loud")

        assert provider._streams["quiet"].gain_db > 0
        assert provider._streams["loud"].gain_db < 0

    def test_silence_is_not_amplified(self) -> None:
        provider = SimpleAGCProvider(
            AGCConfig(target_level_dbfs=-3.0, max_gain_db=30.0, attack_ms=0.0)
        )

        result = provider.process(_frame(0), "alice")

        assert result.data == _frame(0).data
        assert result.metadata["gain_applied_db"] == 0.0

    def test_silence_is_not_amplified_after_gain_has_adapted(self) -> None:
        provider = SimpleAGCProvider(
            AGCConfig(target_level_dbfs=-12.0, max_gain_db=20.0, attack_ms=0.0)
        )
        provider.process(_frame(1000), "alice")

        silence = _frame(0)
        result = provider.process(silence, "alice")

        assert provider._streams["alice"].gain_db > 0
        assert result.data == silence.data
        assert result.metadata["gain_applied_db"] == 0.0

    def test_attack_and_release_use_actual_frame_duration(self) -> None:
        provider = SimpleAGCProvider(
            AGCConfig(
                target_level_dbfs=-12.0,
                max_gain_db=30.0,
                attack_ms=100.0,
                release_ms=200.0,
            )
        )

        quiet = provider.process(_frame(1000, samples=320), "alice")
        medium = provider.process(_frame(6000, samples=320), "alice")

        quiet_level = 20 * math.log10(1000 / 32768)
        quiet_target_gain = -12.0 - quiet_level
        expected_attack = quiet_target_gain * (1 - math.exp(-20 / 100))
        medium_level = 20 * math.log10(6000 / 32768)
        medium_target_gain = -12.0 - medium_level
        expected_release = expected_attack + (1 - math.exp(-20 / 200)) * (
            medium_target_gain - expected_attack
        )
        assert quiet.metadata["gain_applied_db"] == pytest.approx(expected_attack)
        assert medium.metadata["gain_applied_db"] == pytest.approx(expected_release)

    def test_reset_drops_only_one_stream(self) -> None:
        provider = SimpleAGCProvider(AGCConfig(attack_ms=0.0))
        provider.process(_frame(1000), "alice")
        provider.process(_frame(1000), "bob")

        provider.reset("alice")

        assert set(provider._streams) == {"bob"}

    def test_unsupported_pcm_format_is_not_misinterpreted(self) -> None:
        provider = SimpleAGCProvider()
        frame = AudioFrame(
            data=b"\x01" * 320,
            sample_rate=16000,
            channels=1,
            sample_width=1,
        )

        result = provider.process(frame, "alice")

        assert result is frame
        assert provider._streams == {}

    def test_process_after_close_does_not_resurrect_state(self) -> None:
        provider = SimpleAGCProvider()
        provider.close()
        frame = _frame(1000)

        assert provider.process(frame, "alice") is frame
        assert provider._streams == {}


class TestAGCConfigValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"target_level_dbfs": 1.0}, "target_level_dbfs"),
            ({"max_gain_db": -1.0}, "max_gain_db"),
            ({"attack_ms": -1.0}, "attack_ms"),
            ({"release_ms": -1.0}, "release_ms"),
        ],
    )
    def test_invalid_values_fail_fast(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            AGCConfig(**kwargs)
