"""Tests for VideoEffectTransform (video/pipeline/transform/effects.py)."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _build_mock_cv2() -> MagicMock:
    """Build a mock cv2 module with required constants."""
    cv2 = MagicMock()
    cv2.COLOR_RGB2GRAY = 7
    cv2.COLOR_GRAY2RGB = 8
    cv2.COLOR_RGB2BGR = 4
    cv2.COLOR_BGR2RGB = 5
    cv2.COLOR_BGR2GRAY = 6
    cv2.INTER_LINEAR = 1
    cv2.INTER_NEAREST = 0
    cv2.ADAPTIVE_THRESH_MEAN_C = 0
    cv2.THRESH_BINARY = 0
    arr = MagicMock()
    arr.tobytes.return_value = b"\x00" * (4 * 4 * 3)
    arr.shape = (4, 4, 3)
    arr.astype.return_value = arr
    cv2.cvtColor.return_value = arr
    cv2.GaussianBlur.return_value = arr
    cv2.resize.return_value = arr
    cv2.Canny.return_value = arr
    cv2.medianBlur.return_value = arr
    cv2.adaptiveThreshold.return_value = arr
    cv2.bilateralFilter.return_value = arr
    cv2.bitwise_and.return_value = arr
    cv2.divide.return_value = arr
    cv2.transform.return_value = arr
    return cv2


def _build_mock_numpy() -> MagicMock:
    """Build a mock numpy module."""
    np = MagicMock()
    np.uint8 = "uint8"
    np.float32 = "float32"
    np.float64 = "float64"
    arr = MagicMock()
    arr.tobytes.return_value = b"\x00" * (4 * 4 * 3)
    arr.shape = (4, 4, 3)
    arr.astype.return_value = arr
    arr.copy.return_value = arr
    arr.__sub__ = lambda self, other: arr
    np.frombuffer.return_value = MagicMock(reshape=MagicMock(return_value=arr))
    np.clip.return_value = arr
    np.array.return_value = arr
    return np


class TestVideoEffectTransform:
    def _reload_module(self, mock_cv2: MagicMock, mock_np: MagicMock) -> SimpleNamespace:
        with patch.dict(sys.modules, {"cv2": mock_cv2, "numpy": mock_np}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.video.pipeline.transform.effects")
            importlib.reload(mod)
            return mod  # type: ignore[return-value]

    def test_constructor_valid_effects(self) -> None:
        mock_cv2 = _build_mock_cv2()
        mock_np = _build_mock_numpy()
        mod = self._reload_module(mock_cv2, mock_np)
        valid_effects = [
            "grayscale",
            "sepia",
            "invert",
            "blur",
            "cartoon",
            "edges",
            "sketch",
            "pixelate",
        ]
        for effect in valid_effects:
            t = mod.VideoEffectTransform(effect)
            assert t._effect == effect

    def test_invalid_effect_raises(self) -> None:
        mock_cv2 = _build_mock_cv2()
        mock_np = _build_mock_numpy()
        mod = self._reload_module(mock_cv2, mock_np)
        with pytest.raises(ValueError, match="Unknown effect"):
            mod.VideoEffectTransform("nonexistent")

    def test_name_format(self) -> None:
        mock_cv2 = _build_mock_cv2()
        mock_np = _build_mock_numpy()
        mod = self._reload_module(mock_cv2, mock_np)
        t = mod.VideoEffectTransform("grayscale")
        assert t.name == "effect:grayscale"

    def test_passthrough_for_non_raw_rgb24(self) -> None:
        mock_cv2 = _build_mock_cv2()
        mock_np = _build_mock_numpy()
        mod = self._reload_module(mock_cv2, mock_np)
        t = mod.VideoEffectTransform("grayscale")
        frame = SimpleNamespace(
            data=b"\x00" * 100,
            codec="h264",
            width=10,
            height=10,
        )
        result = t.transform(frame)
        assert result is frame  # Should pass through unchanged
