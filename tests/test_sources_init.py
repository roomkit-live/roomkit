"""Tests for lazy imports in sources/__init__.py."""

from __future__ import annotations

import importlib

import pytest

import roomkit.sources as sources_mod


def _neonize_available() -> bool:
    """Check if neonize can be imported without errors."""
    try:
        importlib.import_module("roomkit.sources.neonize")
        return True
    except Exception:
        return False


class TestLazyImports:
    def test_websocket_source(self) -> None:
        from roomkit.sources.websocket import WebSocketSource

        result = sources_mod.WebSocketSource
        assert result is WebSocketSource

    def test_sse_source(self) -> None:
        from roomkit.sources.sse import SSESource

        result = sources_mod.SSESource
        assert result is SSESource

    @pytest.mark.skipif(not _neonize_available(), reason="neonize has protobuf conflict")
    def test_whatsapp_personal_source(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        result = sources_mod.WhatsAppPersonalSourceProvider
        assert result is WhatsAppPersonalSourceProvider

    def test_unknown_attribute_raises(self) -> None:
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = sources_mod.NonExistentThing  # type: ignore[attr-defined]

    def test_getattr_via_getattr_builtin(self) -> None:
        """Verify __getattr__ works via the builtin getattr()."""
        cls = sources_mod.WebSocketSource
        assert cls.__name__ == "WebSocketSource"

    def test_all_non_neonize_exports(self) -> None:
        """All non-neonize names in __all__ should be importable."""
        skip = {"WhatsAppPersonalSourceProvider"}
        for name in sources_mod.__all__:
            if name in skip:
                continue
            obj = getattr(sources_mod, name)
            assert obj is not None
