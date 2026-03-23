"""Tests for AnamAvatarProvider (providers/anam/avatar.py)."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from roomkit.providers.anam.config import AnamConfig


def _build_mock_anam() -> MagicMock:
    """Build a mock anam module."""
    mock_anam = MagicMock()
    mock_anam.PersonaConfig = MagicMock(return_value=SimpleNamespace())
    mock_anam.AnamClient = MagicMock(return_value=MagicMock())
    mock_anam.AgentAudioInputConfig = MagicMock(return_value=SimpleNamespace())
    return mock_anam


class TestAnamAvatarProvider:
    def _make_config(self) -> AnamConfig:
        return AnamConfig(api_key="test-key", avatar_id="av-1")

    def test_constructor(self) -> None:
        mock_anam = _build_mock_anam()
        with patch.dict(sys.modules, {"anam": mock_anam}):
            # Reset the module-level global
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.providers.anam.avatar")
            importlib.reload(mod)

            provider = mod.AnamAvatarProvider(self._make_config())
            assert provider.name == "anam-cloud"
            assert provider.fps == 25
            assert provider.is_started is False
            assert provider.is_async is True

    def test_custom_fps(self) -> None:
        mock_anam = _build_mock_anam()
        with patch.dict(sys.modules, {"anam": mock_anam}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.providers.anam.avatar")
            importlib.reload(mod)

            provider = mod.AnamAvatarProvider(self._make_config(), video_fps=30)
            assert provider.fps == 30

    def test_on_video_callback_registration(self) -> None:
        mock_anam = _build_mock_anam()
        with patch.dict(sys.modules, {"anam": mock_anam}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.providers.anam.avatar")
            importlib.reload(mod)

            provider = mod.AnamAvatarProvider(self._make_config())
            cb = MagicMock()
            provider.on_video(cb)
            assert cb in provider._video_callbacks

    async def test_stop_cleanup(self) -> None:
        mock_anam = _build_mock_anam()
        with patch.dict(sys.modules, {"anam": mock_anam}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.providers.anam.avatar")
            importlib.reload(mod)

            provider = mod.AnamAvatarProvider(self._make_config())
            # Stop without starting should be safe
            await provider.stop()
            assert provider.is_started is False
            assert provider._session is None
            assert provider._audio_stream is None
            assert provider._client is None

    def test_feed_audio_when_not_started(self) -> None:
        mock_anam = _build_mock_anam()
        with patch.dict(sys.modules, {"anam": mock_anam}):
            importlib.invalidate_caches()
            mod = importlib.import_module("roomkit.providers.anam.avatar")
            importlib.reload(mod)

            provider = mod.AnamAvatarProvider(self._make_config())
            result = provider.feed_audio(b"\x00" * 100)
            assert result == []
