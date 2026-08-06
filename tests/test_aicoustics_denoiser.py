"""Tests for AICousticsDenoiserProvider."""

from __future__ import annotations

import importlib
import struct
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from roomkit.voice.audio_frame import AudioFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Quail frame size: 160 samples (10 ms at 16 kHz).
_FRAME_SIZE = 160


def _mock_aic_module() -> MagicMock:
    """Create a MagicMock that stands in for the aic_sdk module."""
    mock = MagicMock()
    # ProcessorConfig.optimal() returns a config with num_frames.
    config_mock = MagicMock()
    config_mock.num_frames = _FRAME_SIZE
    mock.ProcessorConfig.optimal.return_value = config_mock
    # Processor() returns a processor whose process() returns zeros.
    processor_mock = MagicMock()
    processor_mock.process.side_effect = lambda x: np.zeros_like(x)
    processor_mock.context.return_value = MagicMock()
    mock.Processor.return_value = processor_mock
    # Model.download() returns a path.
    mock.Model.download.return_value = "/tmp/models/quail"
    return mock


def _frame(
    n_samples: int = _FRAME_SIZE,
    value: int = 0,
    sample_rate: int = 16000,
    timestamp_ms: float | None = None,
) -> AudioFrame:
    """Create a PCM-16 mono AudioFrame with *n_samples* samples."""
    data = struct.pack(f"<{n_samples}h", *([value] * n_samples))
    return AudioFrame(
        data=data,
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
        timestamp_ms=timestamp_ms,
    )


def _make_provider(
    aic_mock: MagicMock | None = None,
    **config_kwargs: Any,
) -> Any:
    """Create an AICousticsDenoiserProvider with mocked aic_sdk."""
    if aic_mock is None:
        aic_mock = _mock_aic_module()
    with patch.dict("sys.modules", {"aic_sdk": aic_mock}):
        import roomkit.voice.pipeline.denoiser.aicoustics as dn_mod

        importlib.reload(dn_mod)
        from roomkit.voice.pipeline.denoiser.aicoustics import (
            AICousticsDenoiserConfig,
            AICousticsDenoiserProvider,
        )

        cfg = AICousticsDenoiserConfig(**config_kwargs)
        provider = AICousticsDenoiserProvider(cfg)
        provider._aic = aic_mock
        return provider


def _provider_that_returns(samples_2d: np.ndarray[Any, Any]) -> Any:
    """Create a provider whose processor.process() returns given array."""
    aic = _mock_aic_module()
    processor = aic.Processor.return_value
    processor.process.side_effect = lambda _: samples_2d
    return _make_provider(aic)


# ---------------------------------------------------------------------------
# Process — basic denoising
# ---------------------------------------------------------------------------


class TestProcess:
    def test_process_returns_new_frame(self) -> None:
        provider = _make_provider()

        frame_in = _frame(_FRAME_SIZE, value=100)
        frame_out = provider.process(frame_in, "s1")

        assert isinstance(frame_out, AudioFrame)
        assert frame_out is not frame_in
        assert len(frame_out.data) == len(frame_in.data)
        assert frame_out.sample_rate == frame_in.sample_rate
        assert frame_out.channels == frame_in.channels
        assert frame_out.sample_width == frame_in.sample_width

    def test_process_converts_denoised_samples(self) -> None:
        # Processor returns 0.5 for all samples.
        out_array = np.full((1, _FRAME_SIZE), 0.5, dtype=np.float32)
        provider = _provider_that_returns(out_array)

        frame_in = _frame(_FRAME_SIZE, value=1000)
        frame_out = provider.process(frame_in, "s1")

        out_samples = struct.unpack(f"<{_FRAME_SIZE}h", frame_out.data)
        expected = int(0.5 * 32767)
        for s in out_samples:
            assert s == expected

    def test_process_preserves_timestamp(self) -> None:
        provider = _make_provider()

        frame_in = _frame(_FRAME_SIZE, timestamp_ms=42.5)
        frame_out = provider.process(frame_in, "s1")

        assert frame_out.timestamp_ms == 42.5

    def test_process_copies_metadata(self) -> None:
        provider = _make_provider()

        frame_in = _frame(_FRAME_SIZE)
        frame_in.metadata["source"] = "test"
        frame_out = provider.process(frame_in, "s1")

        assert frame_out.metadata["source"] == "test"
        # Mutation of output must not affect input.
        frame_out.metadata["extra"] = True
        assert "extra" not in frame_in.metadata

    def test_process_error_returns_original(self) -> None:
        aic = _mock_aic_module()
        aic.Processor.return_value.process.side_effect = RuntimeError("boom")
        provider = _make_provider(aic)

        frame = _frame(_FRAME_SIZE, value=500)
        result = provider.process(frame, "s1")

        # Should return the original frame on error.
        assert result is frame


# ---------------------------------------------------------------------------
# Frame buffering
# ---------------------------------------------------------------------------


class TestFrameBuffering:
    def test_partial_frame_uses_fixed_delay_instead_of_reemitting_raw_audio(self) -> None:
        """A buffered fragment is represented by delay, not raw duplicate audio."""
        provider = _make_provider()

        half = _FRAME_SIZE // 2
        frame = _frame(half, value=100)
        result = provider.process(frame, "s1")

        assert result is not frame
        assert len(result.data) == len(frame.data)
        assert result.data == b"\x00" * len(frame.data)

    def test_exact_chunk_processed(self) -> None:
        """Frames exactly matching the SDK chunk size are processed."""
        aic = _mock_aic_module()
        processor = aic.Processor.return_value
        provider = _make_provider(aic)

        frame = _frame(_FRAME_SIZE, value=100)
        provider.process(frame, "s1")

        processor.process.assert_called_once()

    def test_two_halves_produce_output(self) -> None:
        """Two half-frames should eventually trigger processing."""
        aic = _mock_aic_module()
        processor = aic.Processor.return_value
        provider = _make_provider(aic)

        half = _FRAME_SIZE // 2
        frame1 = _frame(half, value=100)
        frame2 = _frame(half, value=200)

        provider.process(frame1, "s1")
        assert processor.process.call_count == 0

        provider.process(frame2, "s1")
        assert processor.process.call_count == 1

    def test_irregular_chunks_preserve_sample_order_with_one_chunk_delay(self) -> None:
        aic = _mock_aic_module()
        processor = aic.Processor.return_value
        processor.process.side_effect = lambda samples: samples
        provider = _make_provider(aic)
        half = _FRAME_SIZE // 2

        first = provider.process(_frame(half, value=100), "s1")
        second = provider.process(_frame(half, value=200), "s1")
        third = provider.process(_frame(_FRAME_SIZE, value=300), "s1")

        assert first.data + second.data == b"\x00" * (_FRAME_SIZE * 2)
        delayed = struct.unpack(f"<{_FRAME_SIZE}h", third.data)
        assert all(abs(sample - 100) <= 1 for sample in delayed[:half])
        assert all(abs(sample - 200) <= 1 for sample in delayed[half:])

    def test_stereo_pcm_is_deinterleaved_and_reinterleaved_correctly(self) -> None:
        aic = _mock_aic_module()
        processor = aic.Processor.return_value
        left = np.linspace(-0.5, 0.5, _FRAME_SIZE, dtype=np.float32)
        right = np.linspace(0.5, -0.5, _FRAME_SIZE, dtype=np.float32)
        expected = np.stack([left, right])

        def identity(samples: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            assert np.allclose(samples, expected, atol=1 / 32768)
            return samples

        processor.process.side_effect = identity
        provider = _make_provider(aic, num_channels=2)
        interleaved = expected.T.reshape(-1)
        pcm = bytes(np.clip(interleaved * 32767, -32767, 32767).astype(np.int16).tobytes())
        frame = AudioFrame(pcm, 16000, 2, 2)

        result = provider.process(frame, "stereo")

        assert result.data == frame.data

    def test_double_chunk_processed_twice(self) -> None:
        """A frame with 2x the chunk size triggers two process() calls."""
        aic = _mock_aic_module()
        processor = aic.Processor.return_value
        provider = _make_provider(aic)

        frame = _frame(_FRAME_SIZE * 2, value=100)
        provider.process(frame, "s1")

        assert processor.process.call_count == 2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_model(self) -> None:
        with patch.dict("sys.modules", {"aic_sdk": _mock_aic_module()}):
            import roomkit.voice.pipeline.denoiser.aicoustics as dn_mod

            importlib.reload(dn_mod)
            from roomkit.voice.pipeline.denoiser.aicoustics import (
                AICousticsDenoiserConfig,
            )

            assert AICousticsDenoiserConfig().model == "quail-vf-2.0-l-16khz"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"enhancement_level": 1.1}, "enhancement_level"),
            ({"num_channels": 3}, "num_channels"),
            ({"sample_rate": 0}, "sample_rate"),
        ],
    )
    def test_invalid_config_fails_fast(self, kwargs: dict[str, Any], match: str) -> None:
        with patch.dict("sys.modules", {"aic_sdk": _mock_aic_module()}):
            from roomkit.voice.pipeline.denoiser.aicoustics import AICousticsDenoiserConfig

            with pytest.raises(ValueError, match=match):
                AICousticsDenoiserConfig(**kwargs)

    def test_config_flows_to_sdk(self) -> None:
        aic = _mock_aic_module()
        provider = _make_provider(
            aic,
            model="quail-2.0-l-16khz",
            model_dir="/tmp/custom",
            license_key="test-key-123",
            enhancement_level=0.6,
        )

        # Trigger lazy init.
        provider.process(_frame(_FRAME_SIZE), "s1")

        aic.Model.download.assert_called_once_with("quail-2.0-l-16khz", "/tmp/custom")
        aic.ProcessorConfig.optimal.assert_called_once_with(
            "/tmp/models/quail",
            num_channels=1,
        )
        # License key is passed to Processor(), not ProcessorConfig.
        aic.Processor.assert_called_once_with(
            "/tmp/models/quail",
            "test-key-123",
            aic.ProcessorConfig.optimal.return_value,
        )
        # Enhancement level set via context.
        context = aic.Processor.return_value.context.return_value
        context.set_parameter.assert_called_once_with("enhancement_level", 0.6)


# ---------------------------------------------------------------------------
# Lazy initialization
# ---------------------------------------------------------------------------


class TestLazyInit:
    def test_processor_not_created_until_first_process(self) -> None:
        provider = _make_provider()

        assert provider._streams == {}

        provider.process(_frame(_FRAME_SIZE), "s1")
        assert provider._streams["s1"].processor is not None

    def test_ensure_processor_only_creates_once(self) -> None:
        aic = _mock_aic_module()
        provider = _make_provider(aic)

        provider.process(_frame(_FRAME_SIZE), "s1")
        provider.process(_frame(_FRAME_SIZE), "s1")

        aic.Processor.assert_called_once()

    def test_each_stream_gets_its_own_processor(self) -> None:
        aic = _mock_aic_module()
        provider = _make_provider(aic)

        provider.process(_frame(_FRAME_SIZE), "alice")
        provider.process(_frame(_FRAME_SIZE), "bob")

        assert aic.Processor.call_count == 2
        assert set(provider._streams) == {"alice", "bob"}


# ---------------------------------------------------------------------------
# Name property
# ---------------------------------------------------------------------------


class TestName:
    def test_name(self) -> None:
        provider = _make_provider()
        assert provider.name == "aicoustics"


# ---------------------------------------------------------------------------
# Lifecycle — reset / close
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_reset_clears_buffer(self) -> None:
        provider = _make_provider()

        # Buffer a partial frame.
        half = _FRAME_SIZE // 2
        provider.process(_frame(half, value=100), "s1")
        assert len(provider._streams["s1"].buffer) > 0

        provider.reset("s1")
        assert "s1" not in provider._streams

    def test_reset_recreates_processor(self) -> None:
        aic = _mock_aic_module()
        provider = _make_provider(aic)

        # Initialize.
        provider.process(_frame(_FRAME_SIZE), "s1")
        assert provider._streams["s1"].processor is not None

        provider.reset("s1")
        # The next frame builds a fresh processor, clearing the model's
        # recurrent state.
        provider.process(_frame(_FRAME_SIZE), "s1")
        assert aic.Processor.call_count == 2

    def test_reset_leaves_other_streams_alone(self) -> None:
        provider = _make_provider()

        half = _FRAME_SIZE // 2
        provider.process(_frame(half, value=100), "alice")
        provider.process(_frame(half, value=100), "bob")

        provider.reset("alice")

        assert "alice" not in provider._streams
        assert len(provider._streams["bob"].buffer) > 0

    def test_close_releases_every_stream(self) -> None:
        provider = _make_provider()

        provider.process(_frame(_FRAME_SIZE), "alice")
        provider.process(_frame(_FRAME_SIZE), "bob")

        provider.close()
        assert provider._streams == {}

    def test_double_close(self) -> None:
        provider = _make_provider()

        provider.close()
        provider.close()  # Must not raise.

    def test_reset_before_init_is_safe(self) -> None:
        provider = _make_provider()
        provider.reset("s1")  # Must not raise.

    def test_process_after_close_does_not_recreate_processor(self) -> None:
        aic = _mock_aic_module()
        provider = _make_provider(aic)
        provider.close()
        frame = _frame()

        assert provider.process(frame, "s1") is frame
        aic.Processor.assert_not_called()
        assert provider._streams == {}


class TestFormatValidation:
    def test_wrong_sample_rate_is_bypassed_without_processor(self) -> None:
        aic = _mock_aic_module()
        provider = _make_provider(aic)
        frame = _frame(sample_rate=24000)

        assert provider.process(frame, "s1") is frame
        aic.Processor.assert_not_called()
        assert provider._streams == {}


# ---------------------------------------------------------------------------
# Import error
# ---------------------------------------------------------------------------


class TestImportError:
    def test_import_error_when_aic_sdk_not_installed(self) -> None:
        with patch.dict("sys.modules", {"aic_sdk": None}):
            import roomkit.voice.pipeline.denoiser.aicoustics as dn_mod

            importlib.reload(dn_mod)

            from roomkit.voice.pipeline.denoiser.aicoustics import (
                AICousticsDenoiserConfig,
                AICousticsDenoiserProvider,
            )

            with pytest.raises(ImportError, match="aic-sdk is required"):
                AICousticsDenoiserProvider(AICousticsDenoiserConfig())


# ---------------------------------------------------------------------------
# Environment variable license key
# ---------------------------------------------------------------------------


class TestEnvLicenseKey:
    def test_defaults_to_env_var(self) -> None:
        with patch.dict("sys.modules", {"aic_sdk": _mock_aic_module()}):
            import roomkit.voice.pipeline.denoiser.aicoustics as dn_mod

            importlib.reload(dn_mod)
            from roomkit.voice.pipeline.denoiser.aicoustics import (
                AICousticsDenoiserConfig,
            )

            with patch.dict("os.environ", {"AIC_SDK_LICENSE": "env-key-456"}):
                cfg = AICousticsDenoiserConfig()
                assert cfg._resolved_license_key == "env-key-456"

    def test_explicit_key_overrides_env(self) -> None:
        with patch.dict("sys.modules", {"aic_sdk": _mock_aic_module()}):
            import roomkit.voice.pipeline.denoiser.aicoustics as dn_mod

            importlib.reload(dn_mod)
            from roomkit.voice.pipeline.denoiser.aicoustics import (
                AICousticsDenoiserConfig,
            )

            with patch.dict("os.environ", {"AIC_SDK_LICENSE": "env-key"}):
                cfg = AICousticsDenoiserConfig(license_key="explicit-key")
                assert cfg._resolved_license_key == "explicit-key"

    def test_empty_when_no_key(self) -> None:
        with patch.dict("sys.modules", {"aic_sdk": _mock_aic_module()}):
            import roomkit.voice.pipeline.denoiser.aicoustics as dn_mod

            importlib.reload(dn_mod)
            from roomkit.voice.pipeline.denoiser.aicoustics import (
                AICousticsDenoiserConfig,
            )

            with patch.dict("os.environ", {}, clear=True):
                cfg = AICousticsDenoiserConfig()
                assert cfg._resolved_license_key == ""
