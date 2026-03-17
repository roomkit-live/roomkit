"""Tests for screen_input.py: _parse_json_response, _build_press_key_tool, _get_scale_factor."""

from __future__ import annotations

from unittest.mock import patch

from roomkit.video.vision.screen_input import (
    _build_press_key_tool,
    _get_scale_factor,
    _parse_json_response,
)

# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------


def test_parse_json_clean() -> None:
    raw = (
        '{"found": true, "cx": 100, "cy": 200,'
        ' "box": {"x1": 90, "y1": 190, "x2": 110, "y2": 210},'
        ' "label": "OK"}'
    )
    result = _parse_json_response(raw)
    assert result is not None
    assert result["found"] is True
    assert result["cx"] == 100
    assert result["label"] == "OK"


def test_parse_json_embedded_in_text() -> None:
    raw = (
        'Here is the result: {"found": true, "cx": 50,'
        ' "cy": 60, "box": {}, "label": "btn"} extra text'
    )
    result = _parse_json_response(raw)
    assert result is not None
    assert result["cx"] == 50


def test_parse_json_fallback_with_label() -> None:
    raw = 'garbled "found": true, "cx": 42, "cy": 99, broken json "label": "Search"'
    result = _parse_json_response(raw)
    assert result is not None
    assert result["found"] is True
    assert result["cx"] == 42
    assert result["cy"] == 99
    assert result["label"] == "Search"


def test_parse_json_fallback_without_label() -> None:
    raw = '"found": false, "cx": 0, "cy": 0'
    result = _parse_json_response(raw)
    assert result is not None
    assert result["found"] is False
    assert result["label"] == ""


def test_parse_json_garbage() -> None:
    result = _parse_json_response("totally random text with no json")
    assert result is None


# ---------------------------------------------------------------------------
# _build_press_key_tool
# ---------------------------------------------------------------------------


def test_build_press_key_tool_darwin() -> None:
    with patch("roomkit.video.vision.screen_input.platform") as mock_platform:
        mock_platform.system.return_value = "Darwin"
        tool = _build_press_key_tool()
    assert tool["name"] == "press_key"
    assert "command" in tool["description"]


def test_build_press_key_tool_linux() -> None:
    with patch("roomkit.video.vision.screen_input.platform") as mock_platform:
        mock_platform.system.return_value = "Linux"
        tool = _build_press_key_tool()
    assert tool["name"] == "press_key"
    assert "ctrl" in tool["description"]


# ---------------------------------------------------------------------------
# _get_scale_factor DPI guard
# ---------------------------------------------------------------------------


def test_dpi_guard_called_once_on_windows() -> None:
    """SetProcessDpiAwareness should only be called once."""
    import roomkit.video.vision.screen_input as mod

    original = mod._dpi_initialized
    try:
        mod._dpi_initialized = False
        with patch(
            "roomkit.video.vision.screen_input.platform",
        ) as mock_platform:
            mock_platform.system.return_value = "Windows"
            with patch(
                "roomkit.video.vision.screen_input.ctypes",
                create=True,
            ):
                _get_scale_factor()
                assert mod._dpi_initialized is True

                # Second call — guard prevents re-calling SetProcessDpiAwareness
                _get_scale_factor()
                assert mod._dpi_initialized is True
    finally:
        mod._dpi_initialized = original


def test_scale_factor_linux_with_gdk_scale() -> None:
    with patch("roomkit.video.vision.screen_input.platform") as mock_platform:
        mock_platform.system.return_value = "Linux"
        with patch.dict("os.environ", {"GDK_SCALE": "2"}):
            sx, sy = _get_scale_factor()
    assert sx == 0.5
    assert sy == 0.5
