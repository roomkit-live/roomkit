"""Tests for the SmartTurnDetector (smart-turn ONNX model)."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from roomkit.voice.pipeline.turn.base import TurnContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# We need real numpy for the detector's PCM conversion and sigmoid.
# Import it once; if unavailable, skip the whole module.
np = pytest.importorskip("numpy")


def _make_mock_ort():
    """Build a fake onnxruntime module."""
    mod = MagicMock()
    mod.SessionOptions = MagicMock(return_value=MagicMock())
    return mod


def _make_mock_transformers():
    """Build a fake transformers module."""
    fe = MagicMock()
    # Return dict with input_features key
    fe.return_value = {"input_features": np.zeros((1, 80, 3000), dtype=np.float32)}
    mod = MagicMock()
    mod.WhisperFeatureExtractor = MagicMock(return_value=fe)
    return mod


def _make_provider(ort_mod, tr_mod=None, **config_kwargs):
    """Reload module and construct SmartTurnDetector with deps mocked."""
    if tr_mod is None:
        tr_mod = _make_mock_transformers()

    with patch.dict(
        sys.modules,
        {"onnxruntime": ort_mod, "transformers": tr_mod},
    ):
        import roomkit.voice.pipeline.turn.smart_turn as st_mod

        importlib.reload(st_mod)
        config = st_mod.SmartTurnConfig(
            model_path=config_kwargs.pop("model_path", "/fake/model.onnx"),
            **config_kwargs,
        )
        return st_mod.SmartTurnDetector(config), st_mod


def _make_session(ort_mod, logit: float):
    """Create an ONNX InferenceSession mock that returns *logit*."""
    session = MagicMock()
    input_info = MagicMock()
    input_info.name = "input_features"
    session.get_inputs.return_value = [input_info]
    session.run.return_value = [np.array([[logit]], dtype=np.float32)]
    ort_mod.InferenceSession = MagicMock(return_value=session)
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSmartTurnConfig:
    def test_defaults(self):
        ort_mod = _make_mock_ort()
        _, st_mod = _make_provider(ort_mod)

        config = st_mod.SmartTurnConfig(model_path="/fake/model.onnx")
        assert config.threshold == 0.5
        assert config.num_threads == 1
        assert config.provider == "cpu"
        assert config.fallback_on_no_audio is True
        assert config.max_consecutive_failures == 3


class TestSmartTurnDetectorConstructor:
    def test_constructor(self):
        ort_mod = _make_mock_ort()
        detector, _ = _make_provider(ort_mod)

        assert detector.name == "SmartTurnDetector"
        assert detector._session is None  # lazy init

    def test_empty_model_path_raises(self):
        ort_mod = _make_mock_ort()
        with pytest.raises(ValueError, match="model_path"):
            _make_provider(ort_mod, model_path="")


class TestSmartTurnDetectorEvaluateNoAudio:
    def test_no_audio_fallback_true(self):
        ort_mod = _make_mock_ort()
        detector, _ = _make_provider(ort_mod, fallback_on_no_audio=True)

        ctx = TurnContext(audio_bytes=None)
        decision = detector.evaluate(ctx)
        assert decision.is_complete is True
        assert decision.confidence == 0.0

    def test_no_audio_fallback_false(self):
        ort_mod = _make_mock_ort()
        detector, _ = _make_provider(ort_mod, fallback_on_no_audio=False)

        ctx = TurnContext(audio_bytes=None)
        decision = detector.evaluate(ctx)
        assert decision.is_complete is False


class TestSmartTurnDetectorEvaluateWithAudio:
    def test_evaluate_complete_turn(self):
        """Logit > 0 -> sigmoid > 0.5 -> turn complete."""
        ort_mod = _make_mock_ort()
        tr_mod = _make_mock_transformers()
        detector, st_mod = _make_provider(ort_mod, tr_mod)

        _make_session(ort_mod, logit=2.0)

        # 8s of 16kHz int16 PCM
        audio = (b"\x01\x00") * 128000
        ctx = TurnContext(audio_bytes=audio, audio_sample_rate=16000)

        with patch.dict(sys.modules, {"onnxruntime": ort_mod, "transformers": tr_mod}):
            decision = detector.evaluate(ctx)

        assert decision.is_complete == True  # noqa: E712  (np.bool_ vs bool)
        assert detector._consecutive_failures == 0

    def test_evaluate_incomplete_turn(self):
        """Logit < 0 -> sigmoid < 0.5 -> turn incomplete."""
        ort_mod = _make_mock_ort()
        tr_mod = _make_mock_transformers()
        detector, st_mod = _make_provider(ort_mod, tr_mod)

        _make_session(ort_mod, logit=-3.0)

        audio = (b"\x01\x00") * 128000
        ctx = TurnContext(audio_bytes=audio, audio_sample_rate=16000)

        with patch.dict(sys.modules, {"onnxruntime": ort_mod, "transformers": tr_mod}):
            decision = detector.evaluate(ctx)

        assert decision.is_complete == False  # noqa: E712  (np.bool_ vs bool)


class TestSmartTurnDetectorFailureCounter:
    def test_consecutive_failures_fail_open_then_closed(self):
        """After max failures, switch from fail-open to fail-closed."""
        ort_mod = _make_mock_ort()
        tr_mod = _make_mock_transformers()
        detector, st_mod = _make_provider(
            ort_mod,
            tr_mod,
            max_consecutive_failures=2,
        )

        # Make InferenceSession constructor raise to simulate failure
        ort_mod.InferenceSession = MagicMock(side_effect=RuntimeError("ONNX load fail"))

        audio = b"\x01\x00" * 1000
        ctx = TurnContext(audio_bytes=audio)

        with patch.dict(sys.modules, {"onnxruntime": ort_mod, "transformers": tr_mod}):
            # First failure: fail-open
            d1 = detector.evaluate(ctx)
            assert d1.is_complete is True
            assert detector._consecutive_failures == 1

            # Second failure: >= max -> fail-closed
            d2 = detector.evaluate(ctx)
            assert d2.is_complete is False
            assert detector._consecutive_failures == 2


class TestSmartTurnDetectorClose:
    def test_close_releases_resources(self):
        ort_mod = _make_mock_ort()
        detector, _ = _make_provider(ort_mod)

        detector._session = MagicMock()
        detector._feature_extractor = MagicMock()
        detector._consecutive_failures = 5

        detector.close()
        assert detector._session is None
        assert detector._feature_extractor is None
        assert detector._consecutive_failures == 0
