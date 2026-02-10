"""Tests for public API surface."""

from __future__ import annotations

import roomkit


class TestPublicAPI:
    def test_version_string(self) -> None:
        assert isinstance(roomkit.__version__, str)
        assert roomkit.__version__ == "0.4.6"

    def test_all_names_importable(self) -> None:
        for name in roomkit.__all__:
            obj = getattr(roomkit, name)
            assert obj is not None, f"{name} is None"

    def test_core_classes_available(self) -> None:
        assert roomkit.RoomKit is not None
        assert roomkit.Channel is not None
        assert roomkit.Room is not None
        assert roomkit.RoomEvent is not None

    def test_subpackage_imports(self) -> None:
        from roomkit.channels import base
        from roomkit.core import hooks
        from roomkit.identity import mock
        from roomkit.models import enums
        from roomkit.store import memory

        assert enums is not None
        assert hooks is not None
        assert base is not None
        assert memory is not None
        assert mock is not None

    def test_exception_classes(self) -> None:
        assert issubclass(roomkit.RoomNotFoundError, Exception)
        assert issubclass(roomkit.ChannelNotFoundError, Exception)
        assert issubclass(roomkit.ChannelNotRegisteredError, Exception)
