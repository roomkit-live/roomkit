"""Tests for screen input clipboard paste."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from roomkit.video.vision.screen_input import _clipboard_paste


class TestClipboardPaste:
    @patch("roomkit.video.vision.screen_input.platform")
    @patch("roomkit.video.vision.screen_input.subprocess")
    @patch("roomkit.video.vision.screen_input._get_pyautogui")
    def test_macos_uses_pbcopy(self, mock_pag_fn, mock_subprocess, mock_platform) -> None:
        mock_platform.system.return_value = "Darwin"
        mock_pag = MagicMock()
        mock_pag_fn.return_value = mock_pag

        _clipboard_paste("hello world")

        mock_subprocess.run.assert_called_once()
        call_args = mock_subprocess.run.call_args
        assert call_args[0][0] == ["pbcopy"]
        assert call_args[1]["input"] == b"hello world"
        mock_pag.hotkey.assert_called_once_with("command", "v")

    @patch("roomkit.video.vision.screen_input.platform")
    @patch("roomkit.video.vision.screen_input.subprocess")
    @patch("roomkit.video.vision.screen_input._get_pyautogui")
    def test_linux_uses_xclip(self, mock_pag_fn, mock_subprocess, mock_platform) -> None:
        mock_platform.system.return_value = "Linux"
        mock_pag = MagicMock()
        mock_pag_fn.return_value = mock_pag

        _clipboard_paste("test text")

        mock_subprocess.run.assert_called_once()
        call_args = mock_subprocess.run.call_args
        assert call_args[0][0] == ["xclip", "-selection", "clipboard"]
        mock_pag.hotkey.assert_called_once_with("ctrl", "v")

    @patch("roomkit.video.vision.screen_input.platform")
    @patch("roomkit.video.vision.screen_input.subprocess")
    @patch("roomkit.video.vision.screen_input._get_pyautogui")
    def test_windows_uses_clip(self, mock_pag_fn, mock_subprocess, mock_platform) -> None:
        mock_platform.system.return_value = "Windows"
        mock_pag = MagicMock()
        mock_pag_fn.return_value = mock_pag

        _clipboard_paste("windows text")

        mock_subprocess.run.assert_called_once()
        call_args = mock_subprocess.run.call_args
        assert call_args[0][0] == ["clip"]
        mock_pag.hotkey.assert_called_once_with("ctrl", "v")

    @patch("roomkit.video.vision.screen_input.platform")
    @patch("roomkit.video.vision.screen_input.subprocess")
    @patch("roomkit.video.vision.screen_input._get_pyautogui")
    def test_fallback_to_typewrite(self, mock_pag_fn, mock_subprocess, mock_platform) -> None:
        """Should fall back to typewrite when clipboard fails."""
        import subprocess as real_subprocess

        mock_platform.system.return_value = "Darwin"
        mock_subprocess.run.side_effect = real_subprocess.SubprocessError("fail")
        mock_subprocess.SubprocessError = real_subprocess.SubprocessError
        mock_pag = MagicMock()
        mock_pag_fn.return_value = mock_pag

        _clipboard_paste("fallback text")

        mock_pag.typewrite.assert_called_once_with("fallback text", interval=0.02)
