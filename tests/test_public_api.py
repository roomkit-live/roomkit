"""Tests for public API surface."""

from __future__ import annotations

import re

import roomkit


class TestPublicAPI:
    def test_version_string(self) -> None:
        assert isinstance(roomkit.__version__, str)
        # PEP 440 release identifier: ``MAJOR.MINOR.PATCH`` with an
        # optional pre-release suffix (``a``/``b``/``rc`` + digits) and
        # an optional post/dev tag. Pinning the exact string here would
        # tie every release commit to a test edit — keep this loose so
        # the assertion catches a missing or empty version, not a bump.
        assert re.match(
            r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?$",
            roomkit.__version__,
        ), f"unexpected __version__ {roomkit.__version__!r}"

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
