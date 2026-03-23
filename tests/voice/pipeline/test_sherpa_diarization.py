"""Tests for the sherpa-onnx speaker diarization provider."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_sherpa_onnx():
    """Build a fake sherpa_onnx module."""
    extractor = MagicMock()
    extractor.dim = 192

    manager = MagicMock()
    manager.all_speakers = []
    manager.add = MagicMock(return_value=True)
    manager.remove = MagicMock(return_value=True)
    manager.search = MagicMock(return_value="")

    mod = SimpleNamespace(
        SpeakerEmbeddingExtractorConfig=MagicMock(),
        SpeakerEmbeddingExtractor=MagicMock(return_value=extractor),
        SpeakerEmbeddingManager=MagicMock(return_value=manager),
    )
    return mod, extractor, manager


def _make_provider(mock_mod, **config_kwargs):
    """Reload module and construct SherpaOnnxDiarizationProvider."""
    with patch.dict(sys.modules, {"sherpa_onnx": mock_mod}):
        import roomkit.voice.pipeline.diarization.sherpa_onnx as diar_mod

        importlib.reload(diar_mod)
        config = diar_mod.SherpaOnnxDiarizationConfig(
            model=config_kwargs.pop("model", "/fake/model.onnx"),
            **config_kwargs,
        )
        return diar_mod.SherpaOnnxDiarizationProvider(config), diar_mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSherpaOnnxDiarizationConfig:
    def test_defaults(self):
        mock_mod, _, _ = _make_mock_sherpa_onnx()
        with patch.dict(sys.modules, {"sherpa_onnx": mock_mod}):
            import roomkit.voice.pipeline.diarization.sherpa_onnx as diar_mod

            importlib.reload(diar_mod)
            config = diar_mod.SherpaOnnxDiarizationConfig(model="/fake/model.onnx")

        assert config.model == "/fake/model.onnx"
        assert config.num_threads == 1
        assert config.search_threshold == 0.5
        assert config.min_speech_ms == 500


class TestSherpaOnnxDiarizationProviderConstructor:
    def test_constructor(self):
        mock_mod, _, _ = _make_mock_sherpa_onnx()
        provider, _ = _make_provider(mock_mod)
        assert provider.name == "SherpaOnnxDiarizationProvider"


class TestSherpaOnnxDiarizationProviderEnroll:
    def test_enroll_speaker(self):
        mock_mod, _, manager = _make_mock_sherpa_onnx()
        provider, _ = _make_provider(mock_mod)

        embedding = [0.1] * 192
        result = provider.enroll_speaker("alice", embedding)
        assert result is True
        manager.add.assert_called_once_with("alice", embedding)

    def test_remove_speaker(self):
        mock_mod, _, manager = _make_mock_sherpa_onnx()
        provider, _ = _make_provider(mock_mod)

        result = provider.remove_speaker("alice")
        assert result is True
        manager.remove.assert_called_once_with("alice")


class TestSherpaOnnxDiarizationProviderReset:
    def test_reset_clears_state(self):
        mock_mod, _, _ = _make_mock_sherpa_onnx()
        provider, _ = _make_provider(mock_mod)

        provider._speech_buffer.extend(b"\x00" * 100)
        provider._in_speech = True
        provider._last_speaker_id = "bob"

        provider.reset()

        assert len(provider._speech_buffer) == 0
        assert provider._in_speech is False
        assert provider._last_speaker_id == ""


class TestSherpaOnnxDiarizationProviderClose:
    def test_close_clears_buffer(self):
        mock_mod, _, _ = _make_mock_sherpa_onnx()
        provider, _ = _make_provider(mock_mod)

        provider._speech_buffer.extend(b"\x00" * 200)
        provider.close()
        assert len(provider._speech_buffer) == 0
