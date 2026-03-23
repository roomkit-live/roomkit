"""Tests for ScreenInputTools (video/vision/screen_input.py)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


class TestScreenInputTools:
    def test_constructor(self) -> None:
        from roomkit.video.vision.screen_input import ScreenInputTools

        tools = ScreenInputTools()
        assert tools._vision is None
        assert tools._monitor == 1

    def test_constructor_with_vision(self) -> None:
        from roomkit.video.vision.screen_input import ScreenInputTools

        mock_vision = MagicMock()
        tools = ScreenInputTools(vision=mock_vision, monitor=2)
        assert tools._vision is mock_vision
        assert tools._monitor == 2

    def test_definitions_without_vision(self) -> None:
        from roomkit.video.vision.screen_input import ScreenInputTools

        tools = ScreenInputTools()
        defs = tools.definitions
        names = [d["name"] for d in defs]
        assert "type_text" in names
        assert "press_key" in names
        assert "scroll" in names
        assert "click_element" not in names

    def test_definitions_with_vision(self) -> None:
        from roomkit.video.vision.screen_input import ScreenInputTools

        mock_vision = MagicMock()
        tools = ScreenInputTools(vision=mock_vision)
        defs = tools.definitions
        names = [d["name"] for d in defs]
        assert "click_element" in names

    async def test_handler_type_text(self) -> None:
        mock_pag = MagicMock()
        mock_pag.FAILSAFE = True
        with (
            patch.dict(sys.modules, {"pyautogui": mock_pag}),
            patch("roomkit.video.vision.screen_input._clipboard_paste") as mock_paste,
        ):
            from roomkit.video.vision.screen_input import ScreenInputTools

            tools = ScreenInputTools()
            result = await tools.handler("type_text", {"text": "hello"})
            assert "hello" in result
            mock_paste.assert_called_once_with("hello")

    async def test_handler_press_key(self) -> None:
        mock_pag = MagicMock()
        mock_pag.FAILSAFE = True
        with (
            patch.dict(sys.modules, {"pyautogui": mock_pag}),
            patch("roomkit.video.vision.screen_input._get_pyautogui", return_value=mock_pag),
        ):
            from roomkit.video.vision.screen_input import ScreenInputTools

            tools = ScreenInputTools()
            result = await tools.handler("press_key", {"key": "enter"})
            assert "enter" in result
            mock_pag.press.assert_called_once_with("enter")

    async def test_handler_scroll(self) -> None:
        mock_pag = MagicMock()
        mock_pag.FAILSAFE = True
        with (
            patch.dict(sys.modules, {"pyautogui": mock_pag}),
            patch("roomkit.video.vision.screen_input._get_pyautogui", return_value=mock_pag),
        ):
            from roomkit.video.vision.screen_input import ScreenInputTools

            tools = ScreenInputTools()
            result = await tools.handler("scroll", {"clicks": -3})
            assert "-3" in result
            mock_pag.scroll.assert_called_once_with(-3)

    async def test_handler_unknown_tool(self) -> None:
        from roomkit.video.vision.screen_input import ScreenInputTools

        tools = ScreenInputTools()
        result = await tools.handler("nonexistent", {})
        assert "Unknown tool" in result
