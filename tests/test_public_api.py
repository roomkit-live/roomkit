"""Tests for public API surface."""

from __future__ import annotations

import importlib
import inspect
import json
import pydoc
import re
import sys
from pathlib import Path
from types import ModuleType

import pydantic
import pytest

import roomkit
from roomkit.voice import realtime as voice_realtime
from roomkit.voice import tts as voice_tts
from tests import api_signatures
from tests.api_signatures import (
    breaks_since_snapshot,
    call_breaks,
    drift_since_snapshot,
    names_since_snapshot,
    object_breaks,
)

_PACKAGE_ROOT = Path(roomkit.__file__).parent
_EXPORTED = Path(__file__).with_name("public_api_names.json")


class TestPublicAPI:
    def test_text_history_marker_values(self) -> None:
        from roomkit.models import STANDALONE, TURNS_KEY

        assert roomkit.SUPERSEDED == "superseded"
        assert STANDALONE == "standalone"
        assert TURNS_KEY == "turns"

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

    def test_every_name_a_package_exports_resolves(self) -> None:
        """A name in ``__all__`` that does not resolve fails only whoever imports
        it, so the list of roomkit and of each of its subpackages is read here,
        once.

        A package, or a lazily imported name, that needs an optional dependency
        the environment lacks is left out; a missing roomkit module is a failure.
        """
        unresolved: dict[str, list[str]] = {}
        for init in sorted(_PACKAGE_ROOT.rglob("__init__.py")):
            package = ".".join(("roomkit", *init.parent.relative_to(_PACKAGE_ROOT).parts))
            try:
                module = importlib.import_module(package)
            except ModuleNotFoundError as missing:
                _raise_unless_optional(missing)
                continue
            names = [n for n in getattr(module, "__all__", ()) if not _resolves(module, n)]
            if names:
                unresolved[package] = names

        assert unresolved == {}

    def test_no_name_a_package_exported_is_dropped(self) -> None:
        """Every name a package exported stays exported: an application imports
        it from there. tests/public_api_names.json lists them, written by
        `make api-snapshot` with the signatures; the commit that breaks an
        import on purpose runs it.

        A package whose optional dependency the environment lacks is left out,
        as test_every_name_a_package_exports_resolves leaves it out.
        """
        dropped: dict[str, list[str]] = {}
        for package, names in json.loads(_EXPORTED.read_text()).items():
            try:
                module = importlib.import_module(package)
            except ModuleNotFoundError as missing:
                _raise_unless_optional(missing)
                continue
            gone = sorted(set(names) - set(getattr(module, "__all__", ())))
            if gone:
                dropped[package] = gone

        assert dropped == {}

    def test_no_signature_change_breaks_a_caller(self) -> None:
        """A call that worked on a public class or function still works:
        tests/public_api_signatures.json holds their signatures, read as
        tests/api_signatures.py says. The commit that breaks a call on purpose
        runs `make api-snapshot` and carries its BREAKING entry.
        """
        assert breaks_since_snapshot() == []

    def test_the_snapshot_holds_the_whole_public_surface(self) -> None:
        """roomkit-rs pins both files as Python's public API and mirrors what
        they hold, so a name, member or parameter the code adds, or a change no
        caller notices, is written there by the commit that makes it: `make
        api-snapshot`. Only a break carries a BREAKING entry.

        A name whose optional dependency the environment lacks is left out, as
        the two tests above leave it out.
        """
        assert drift_since_snapshot() == []
        assert names_since_snapshot() == []

    def test_channels_importable_from_channels_package(self) -> None:
        """Every channel resolves from ``roomkit.channels``, factories included.

        The transport factories always lived here; the channel classes only in
        the top-level package, so a reader had no rule telling one from the
        other and half the documented imports raised ImportError.
        """
        import roomkit.channels as channels

        for name in (
            "ACPChannel",
            "AIChannel",
            "CLIChannel",
            "EmailChannel",
            "HTTPChannel",
            "RCSChannel",
            "RealtimeVoiceChannel",
            "SMSChannel",
            "TransportChannel",
            "VoiceChannel",
            "WebSocketChannel",
        ):
            obj = getattr(channels, name, None)
            assert obj is not None, f"roomkit.channels.{name} is missing"
            assert obj is getattr(roomkit, name), f"{name} differs from the top-level export"


def _resolves(module: ModuleType, name: str) -> bool:
    """Whether ``name`` resolves on ``module``; an absent optional dependency counts."""
    try:
        getattr(module, name)
    except ImportError as missing:
        # A lazy name whose optional dependency is absent, or installed without
        # what the module imports from it (a module's own ImportError).
        _raise_unless_optional(missing)
    except AttributeError:
        return False
    return True


def _raise_unless_optional(missing: ImportError) -> None:
    if (missing.name or "").startswith("roomkit"):
        raise missing


@pytest.mark.parametrize(
    ("old", "new", "named"),
    [
        (["a", "b=1"], ["a", "b=1", "*", "c=None"], None),
        (["a"], ["a", "b=None"], None),
        (["a"], ["a=None"], None),
        (["*", "a"], ["a"], None),
        (["a", "/"], ["b", "/"], None),
        (["a"], ["a", "**kwargs"], None),
        (["a", "b=1"], ["a"], "'b'"),
        (["a"], ["b"], "'a'"),
        (["a=200"], ["a=50"], "'a'"),
        (["a=None"], ["a"], "'a'"),
        (["a"], ["a", "b"], "'b'"),
        (["a"], ["a", "*", "b"], "'b'"),
        (["a", "b=None"], ["a", "*", "b=None"], "'b'"),
        (["a", "b=None"], ["a", "c=None", "b=None"], "'b'"),
        (["a"], ["a", "/"], "'a'"),
        (["*", "a=None"], [], "'a'"),
        (["a", "**kwargs"], ["a"], "**kwargs"),
        (["*args"], [], "*args"),
    ],
    ids=[
        "keyword added with a default",
        "positional appended with a default",
        "required gains a default",
        "keyword-only becomes positional",
        "positional-only renamed",
        "kwargs added",
        "removed",
        "renamed",
        "default changed",
        "default dropped",
        "required added",
        "required keyword-only added",
        "made keyword-only",
        "inserted before another",
        "made positional-only",
        "keyword-only removed",
        "kwargs removed",
        "args removed",
    ],
)
def test_a_signature_change_breaks_only_a_call_that_worked(
    old: list[str], new: list[str], named: str | None
) -> None:
    """Each parameter is matched by its position when a caller can pass it by
    position, by its name otherwise; an addition no caller can notice passes."""
    found = call_breaks(old, new)
    if named is None:
        assert found == []
    else:
        assert len(found) == 1 and named in found[0], found


@pytest.mark.parametrize(
    ("old", "new", "broken"),
    [
        ({"run": ["def", "x"]}, {"run": ["def", "x"], "stop": ["def"]}, False),
        ({"A": ["member", "'a'"]}, {"A": ["member", "'a'"], "B": ["member", "'b'"]}, False),
        ({"run": ["def", "x"]}, {}, True),
        ({"run": ["def", "x"]}, {"run": ["async def", "x"]}, True),
        ({"size": ["property"]}, {"size": ["def"]}, True),
        ({"A": ["member", "'a'"]}, {"A": ["member", "'b'"]}, True),
        ({"run": ["def", "x"]}, None, True),
    ],
    ids=[
        "method added",
        "enum member added",
        "method removed",
        "method made async",
        "property made a method",
        "enum value changed",
        "no longer exported",
    ],
)
def test_a_member_change_breaks_a_caller(
    old: dict[str, list[str]], new: dict[str, list[str]] | None, broken: bool
) -> None:
    assert bool(object_breaks("roomkit.X", old, new)) is broken


def test_a_lazy_name_whose_optional_dependency_fails_to_import_counts_as_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fish Audio's module raises a plain ImportError of its own when msgpack
    is missing; the name resolves as one whose dependency is absent."""
    monkeypatch.setitem(sys.modules, "msgpack", None)
    for module in ("roomkit.voice.tts.fish_audio", "roomkit.voice.tts._fish_audio_ws"):
        monkeypatch.delitem(sys.modules, module, raising=False)

    assert _resolves(voice_tts, "FishAudioTTSProvider")


def test_help_on_a_lazy_package_does_not_import_a_name_whose_extra_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """help() and inspect.getmembers() read every name dir() lists; a lazy name
    is not listed, so neither imports the FastRTC transport, whose extra
    (fastapi) is set aside, nor raises."""
    _set_aside_fastapi(monkeypatch)

    assert "FastRTCRealtimeTransport" not in dir(voice_realtime)
    pydoc.render_doc(voice_realtime)
    inspect.getmembers(voice_realtime)


def _set_aside_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fastapi", None)
    for module in list(sys.modules):
        if module.startswith(("roomkit.webrtc", "roomkit.voice.realtime.fastrtc")):
            monkeypatch.delitem(sys.modules, module)


def test_a_snapshot_rewrite_keeps_a_missing_extra_and_drops_a_gone_roomkit_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Written without fastapi, the snapshot keeps the FastRTC transport it
    pinned; a roomkit module removed on purpose drops out of it."""
    snapshot = tmp_path / "signatures.json"
    pinned = "roomkit.voice.realtime.FastRTCRealtimeTransport"
    snapshot.write_text(json.dumps({"roomkit.gone.Thing": {}, pinned: {}}))
    monkeypatch.setattr(api_signatures, "SNAPSHOT", snapshot)
    monkeypatch.setattr(api_signatures, "build", dict)
    _set_aside_fastapi(monkeypatch)

    api_signatures.write()

    assert list(json.loads(snapshot.read_text())) == [pinned]


def test_the_snapshot_reads_a_dependency_s_own_import_error_as_a_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "msgpack", None)
    for module in ("roomkit.voice.tts.fish_audio", "roomkit.voice.tts._fish_audio_ws"):
        monkeypatch.delitem(sys.modules, module, raising=False)

    assert api_signatures._resolve(voice_tts, "FishAudioTTSProvider") is api_signatures._OPTIONAL


class _Hooked(pydantic.BaseModel):
    __module__ = "roomkit.models"  # read as one of roomkit's own models

    size: int = 0

    def model_post_init(self, context: object, /) -> None: ...


class _HookOffered(_Hooked):
    """Subclasses extend ``model_post_init`` to finish their setup."""


@pytest.mark.parametrize(
    ("model", "pinned"),
    [(_Hooked, False), (_HookOffered, True)],
    ids=["an override", "named by the docstring"],
)
def test_a_pydantic_hook_is_pinned_only_where_the_model_offers_it(
    model: type[pydantic.BaseModel], pinned: bool
) -> None:
    """No caller calls model_post_init: an override is pinned only when the
    model's docstring names it, and the model's constructor still is."""
    members = api_signatures.describe(model)

    assert ("model_post_init" in members) is pinned
    assert members["__init__"] == ["def", "*", "size=0"]
