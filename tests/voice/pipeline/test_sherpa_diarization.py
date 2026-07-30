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
    def test_reset_clears_only_that_stream(self):
        mock_mod, _, _ = _make_mock_sherpa_onnx()
        provider, _ = _make_provider(mock_mod)

        alice = provider._state_for("alice")
        alice.speech_buffer.extend(b"\x00" * 100)
        alice.in_speech = True
        alice.last_speaker_id = "bob"

        other = provider._state_for("carol")
        other.speech_buffer.extend(b"\x00" * 50)
        other.in_speech = True

        provider.reset("alice")

        assert "alice" not in provider._streams
        # Carol kept accumulating — one speaker leaving does not truncate
        # another's utterance.
        assert len(provider._streams["carol"].speech_buffer) == 50
        assert provider._streams["carol"].in_speech is True

    def test_reset_keeps_enrolled_speakers(self):
        """Enrolment belongs to the room, not to a stream."""
        mock_mod, _, manager = _make_mock_sherpa_onnx()
        provider, _ = _make_provider(mock_mod)

        provider.enroll_speaker("alice", [0.1, 0.2])
        provider.reset("s1")

        assert "alice" in provider._enrolled_embeddings
        manager.remove.assert_not_called()


class TestSherpaOnnxDiarizationProviderClose:
    def test_close_clears_every_stream(self):
        mock_mod, _, _ = _make_mock_sherpa_onnx()
        provider, _ = _make_provider(mock_mod)

        provider._state_for("alice").speech_buffer.extend(b"\x00" * 200)
        provider._state_for("bob").speech_buffer.extend(b"\x00" * 200)

        provider.close()
        assert provider._streams == {}


class TestSherpaOnnxDiarizationProviderClearSpeakers:
    def test_clear_speakers_forgets_all_enrollments(self):
        mock_mod, _, manager = _make_mock_sherpa_onnx()
        manager.all_speakers = ["alice", "bob"]
        provider, _ = _make_provider(mock_mod)
        provider._enrolled_embeddings = {"alice": [0.1] * 192, "bob": [0.2] * 192}

        provider.clear_speakers()

        assert manager.remove.call_count == 2
        manager.remove.assert_any_call("alice")
        manager.remove.assert_any_call("bob")
        assert provider._enrolled_embeddings == {}
