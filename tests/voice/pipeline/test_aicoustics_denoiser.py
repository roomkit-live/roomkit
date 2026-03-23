"""Tests for the ai|coustics denoiser provider."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from roomkit.voice.audio_frame import AudioFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_aic_sdk():
    """Build a fake aic_sdk module."""
    processor = MagicMock()
    context = MagicMock()
    processor.context.return_value = context
    processor.process = MagicMock(return_value=[[0.0] * 160])

    processor_config = MagicMock()
    processor_config.num_frames = 160

    mod = SimpleNamespace(
        Model=SimpleNamespace(download=MagicMock(return_value="/fake/model.onnx")),
        ProcessorConfig=SimpleNamespace(optimal=MagicMock(return_value=processor_config)),
        Processor=MagicMock(return_value=processor),
    )
    return mod, processor


def _make_provider(mock_mod, config=None):
    """Reload module and construct AICousticsDenoiserProvider with aic_sdk mocked."""
    with patch.dict(sys.modules, {"aic_sdk": mock_mod}):
        import roomkit.voice.pipeline.denoiser.aicoustics as aic_mod

        importlib.reload(aic_mod)
        if config is not None:
            return aic_mod.AICousticsDenoiserProvider(config), aic_mod
        return aic_mod.AICousticsDenoiserProvider(), aic_mod


def _make_frame(n_bytes: int = 320) -> AudioFrame:
    return AudioFrame(
        data=b"\x01\x00" * (n_bytes // 2),
        sample_rate=16000,
        channels=1,
        sample_width=2,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAICousticsDenoiserConfig:
    def test_defaults(self):
        mock_mod, _ = _make_mock_aic_sdk()
        with patch.dict(sys.modules, {"aic_sdk": mock_mod}):
            import roomkit.voice.pipeline.denoiser.aicoustics as aic_mod

            importlib.reload(aic_mod)
            config = aic_mod.AICousticsDenoiserConfig()

        assert config.model == "quail-vf-2.0-l-16khz"
        assert config.enhancement_level == 0.8
        assert config.num_channels == 1

    def test_license_from_env(self):
        mock_mod, _ = _make_mock_aic_sdk()
        with (
            patch.dict(sys.modules, {"aic_sdk": mock_mod}),
            patch.dict("os.environ", {"AIC_SDK_LICENSE": "test-key"}),
        ):
            import roomkit.voice.pipeline.denoiser.aicoustics as aic_mod

            importlib.reload(aic_mod)
            config = aic_mod.AICousticsDenoiserConfig()

        assert config._resolved_license_key == "test-key"


class TestAICousticsDenoiserProviderConstructor:
    def test_defaults(self):
        mock_mod, _ = _make_mock_aic_sdk()
        provider, _ = _make_provider(mock_mod)

        assert provider.name == "aicoustics"
        assert provider._processor is None  # lazy init

    def test_import_error(self):
        """Missing aic_sdk raises ImportError."""
        with patch.dict(sys.modules, {"aic_sdk": None}):
            import roomkit.voice.pipeline.denoiser.aicoustics as aic_mod

            importlib.reload(aic_mod)
            with pytest.raises((ImportError, ModuleNotFoundError)):
                aic_mod.AICousticsDenoiserProvider()


class TestAICousticsDenoiserProviderProcess:
    def test_process_lazy_init(self):
        """First process() call triggers lazy initialization."""
        mock_mod, processor = _make_mock_aic_sdk()
        provider, _ = _make_provider(mock_mod)
        assert provider._processor is None

        frame = _make_frame(n_bytes=320)
        result = provider.process(frame)

        assert provider._processor is not None
        assert isinstance(result, AudioFrame)

    def test_process_short_frame_passthrough(self):
        """Frame smaller than chunk size passes through."""
        mock_mod, processor = _make_mock_aic_sdk()
        provider, _ = _make_provider(mock_mod)

        # Force initialization
        with patch.dict(sys.modules, {"aic_sdk": mock_mod}):
            provider._ensure_processor()

        small_frame = _make_frame(n_bytes=100)
        result = provider.process(small_frame)
        assert result is small_frame


class TestAICousticsDenoiserProviderReset:
    def test_reset_clears_buffer(self):
        mock_mod, _ = _make_mock_aic_sdk()
        provider, _ = _make_provider(mock_mod)

        provider._buffer = b"\x00" * 100
        with patch.dict(sys.modules, {"aic_sdk": mock_mod}):
            provider.reset()
        assert provider._buffer == b""


class TestAICousticsDenoiserProviderClose:
    def test_close_releases_resources(self):
        mock_mod, _ = _make_mock_aic_sdk()
        provider, _ = _make_provider(mock_mod)

        with patch.dict(sys.modules, {"aic_sdk": mock_mod}):
            provider._ensure_processor()
        assert provider._processor is not None

        provider.close()
        assert provider._processor is None
        assert provider._buffer == b""
        assert provider._frame_size == 0
