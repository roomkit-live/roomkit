"""Tests for WebSocketAvatarProvider (video/avatar/websocket.py)."""

from __future__ import annotations

import importlib
import struct
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_frame_bytes(w: int, h: int) -> bytes:
    """Build a valid wire-format frame: 2B width BE + 2B height BE + RGB24 data."""
    header = struct.pack("!HH", w, h)
    pixel_data = b"\x00" * (w * h * 3)
    return header + pixel_data


def _fake_ws_modules(
    fake_httpx: SimpleNamespace | None = None,
) -> dict[str, SimpleNamespace]:
    """Build the fake module dict for patching websocket imports."""
    if fake_httpx is None:
        fake_httpx = SimpleNamespace(Client=MagicMock)
    fake_ws_mod = SimpleNamespace(
        sync=SimpleNamespace(
            client=SimpleNamespace(connect=MagicMock),
        ),
    )
    return {
        "httpx": fake_httpx,
        "websockets": fake_ws_mod,
        "websockets.sync": fake_ws_mod.sync,
        "websockets.sync.client": fake_ws_mod.sync.client,
    }


class TestWebSocketAvatarProvider:
    def test_constructor(self) -> None:
        with patch.dict(sys.modules, _fake_ws_modules()):
            importlib.invalidate_caches()
            from roomkit.video.avatar.websocket import WebSocketAvatarProvider

            provider = WebSocketAvatarProvider(
                "http://localhost:8765",
                fps=25,
            )
            assert provider.name == "websocket"
            assert provider.fps == 25
            assert provider.is_started is False

    def test_parse_frame_valid(self) -> None:
        with patch.dict(sys.modules, _fake_ws_modules()):
            importlib.invalidate_caches()
            from roomkit.video.avatar.websocket import WebSocketAvatarProvider

            provider = WebSocketAvatarProvider("http://localhost:8765")
            data = _make_frame_bytes(2, 2)
            frame = provider._parse_frame(data)
            assert frame is not None
            assert frame.width == 2
            assert frame.height == 2
            assert frame.codec == "raw_rgb24"

    def test_parse_frame_too_small(self) -> None:
        with patch.dict(sys.modules, _fake_ws_modules()):
            importlib.invalidate_caches()
            from roomkit.video.avatar.websocket import WebSocketAvatarProvider

            provider = WebSocketAvatarProvider("http://localhost:8765")
            result = provider._parse_frame(b"\x00\x01")
            assert result is None

    async def test_stop_when_not_started(self) -> None:
        fake_httpx = SimpleNamespace(
            Client=MagicMock,
            AsyncClient=MagicMock,
        )
        with patch.dict(sys.modules, _fake_ws_modules(fake_httpx)):
            importlib.invalidate_caches()
            from roomkit.video.avatar.websocket import WebSocketAvatarProvider

            provider = WebSocketAvatarProvider("http://localhost:8765")
            await provider.stop()
            assert provider.is_started is False
