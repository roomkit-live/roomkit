"""Tests for the release-gate model catalog check (``scripts/check_models.py``).

The script is what stops a release shipping a stale catalog, so its own logic
has to be trustworthy — a check that cries wolf gets skipped, and one that
never fires is decoration. Every test here runs offline against a synthetic
upstream payload; the network path is exercised only through a stubbed opener.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from roomkit.providers.ai.base import ModelInfo

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_models.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_models", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves the string annotations that
    # `from __future__ import annotations` produces via sys.modules, and raises
    # on a module that isn't there yet.
    sys.modules["check_models"] = module
    spec.loader.exec_module(module)
    return module


check_models = _load_script()


def _upstream(model_id: str, *, context: int = 200_000, created: int = 1_000) -> dict[str, Any]:
    """One upstream record, shaped like OpenRouter's ``/api/v1/models`` items."""
    return {
        "id": model_id,
        "name": model_id,
        "created": created,
        "context_length": context,
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
    }


def _catalog(monkeypatch: pytest.MonkeyPatch, *models: ModelInfo) -> str:
    """Register a throwaway module exposing ``MODELS``, and return its name."""
    name = f"_fake_catalog_{len(models)}_{id(models)}"
    module = ModuleType(name)
    module.MODELS = list(models)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)
    return name


def _check(module_name: str, upstream: list[dict[str, Any]], *, track_new: bool = True) -> Any:
    return check_models.check_catalog("fake", module_name, "vendor/", False, track_new, upstream)


# --- id normalization ----------------------------------------------------------


def test_normalize_folds_vendor_punctuation() -> None:
    # Anthropic writes claude-opus-4-8; the mirror writes claude-opus-4.8.
    assert check_models.normalize("claude-opus-4-8") == check_models.normalize("claude-opus-4.8")


def test_is_alias_covers_latest_and_dated_snapshots() -> None:
    assert check_models.is_alias("mistral-large-latest")
    assert check_models.is_alias("claude-haiku-4-5-20251001")
    assert not check_models.is_alias("claude-opus-5")


def test_family_takes_the_leading_token() -> None:
    assert check_models.family("gemini-3.6-flash") == "gemini"
    assert check_models.family("gemma-4-31b-it") == "gemma"


# --- findings ------------------------------------------------------------------


def test_drift_reports_a_disagreeing_context_window(monkeypatch: pytest.MonkeyPatch) -> None:
    name = _catalog(monkeypatch, ModelInfo(id="acme-1", context_window=128_000))
    found = _check(name, [_upstream("vendor/acme-1", context=200_000)])
    assert found.drift and "128,000" in found.drift[0] and "200,000" in found.drift[0]


def test_matching_window_is_not_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    name = _catalog(monkeypatch, ModelInfo(id="acme-1", context_window=200_000))
    assert not _check(name, [_upstream("vendor/acme-1")]).errors


def test_unknown_window_is_never_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    # context_window=None means "unknown", which is a valid state, not a wrong one.
    name = _catalog(monkeypatch, ModelInfo(id="acme-1"))
    assert not _check(name, [_upstream("vendor/acme-1")]).drift


def test_missing_reports_a_newer_upstream_model_in_a_known_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = _catalog(monkeypatch, ModelInfo(id="acme-1", context_window=200_000))
    found = _check(
        name,
        [_upstream("vendor/acme-1", created=100), _upstream("vendor/acme-2", created=500)],
    )
    assert found.missing and "acme-2" in found.missing[0]


def test_missing_ignores_an_unrelated_family(monkeypatch: pytest.MonkeyPatch) -> None:
    # A vendor's open-weights line lives in the same namespace as its hosted
    # API models; only one of them belongs in the catalog.
    name = _catalog(monkeypatch, ModelInfo(id="acme-1", context_window=200_000))
    found = _check(
        name,
        [_upstream("vendor/acme-1", created=100), _upstream("vendor/other-9", created=500)],
    )
    assert not found.missing


def test_missing_ignores_older_upstream_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # An id the catalog deliberately omits is only interesting if it postdates
    # everything the catalog knows — otherwise it was a judgement call.
    name = _catalog(monkeypatch, ModelInfo(id="acme-2", context_window=200_000))
    found = _check(
        name,
        [_upstream("vendor/acme-2", created=500), _upstream("vendor/acme-1", created=100)],
    )
    assert not found.missing


def test_missing_is_off_for_partial_catalogs(monkeypatch: pytest.MonkeyPatch) -> None:
    name = _catalog(monkeypatch, ModelInfo(id="acme-1", context_window=200_000))
    found = _check(
        name,
        [_upstream("vendor/acme-1", created=100), _upstream("vendor/acme-2", created=500)],
        track_new=False,
    )
    assert not found.missing


def test_routing_lane_suffixes_are_not_missing_models(monkeypatch: pytest.MonkeyPatch) -> None:
    name = _catalog(monkeypatch, ModelInfo(id="acme-1", context_window=200_000))
    found = _check(
        name,
        [_upstream("vendor/acme-1", created=100), _upstream("vendor/acme-1-fast", created=500)],
    )
    assert not found.missing


def test_gone_warns_on_an_id_upstream_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    name = _catalog(monkeypatch, ModelInfo(id="acme-1", context_window=200_000))
    found = _check(name, [_upstream("vendor/acme-9")])
    assert found.gone and "acme-1" in found.gone[0]
    # A candidate retirement is a warning, never a release blocker: a lagging
    # mirror looks exactly like a real removal.
    assert not found.errors


def test_gone_stays_quiet_for_deprecated_ids_and_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = _catalog(
        monkeypatch,
        ModelInfo(id="acme-old", deprecated=True),
        ModelInfo(id="acme-latest"),
        ModelInfo(id="acme-1-20250101"),
    )
    assert not _check(name, [_upstream("vendor/acme-9")]).gone


# --- deliberate divergences ----------------------------------------------------


def test_deliberate_divergence_suppresses_drift_and_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(check_models.DELIBERATE, "acme-1", "roomkit sends no beta header")
    name = _catalog(monkeypatch, ModelInfo(id="acme-1", context_window=128_000))
    found = _check(name, [_upstream("vendor/acme-1", context=200_000)])
    assert not found.errors
    assert found.expected == 1


def test_mirror_only_slug_suppresses_missing_and_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(check_models.MIRROR_ONLY, "vendor/acme-2", "not a vendor id")
    name = _catalog(monkeypatch, ModelInfo(id="acme-1", context_window=200_000))
    found = _check(
        name,
        [_upstream("vendor/acme-1", created=100), _upstream("vendor/acme-2", created=500)],
    )
    assert not found.errors
    assert found.expected == 1


def test_every_recorded_divergence_carries_a_reason() -> None:
    # An exception without a reason is indistinguishable from silencing an
    # inconvenient finding, so the reason is the price of the entry.
    for reason in (*check_models.DELIBERATE.values(), *check_models.MIRROR_ONLY.values()):
        assert reason.strip()


# --- upstream fetch ------------------------------------------------------------


def test_fetch_keeps_only_text_output_models(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "data": [
            _upstream("vendor/chat"),
            {
                "id": "vendor/image",
                "created": 1,
                "context_length": 32_000,
                "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
            },
        ]
    }
    _stub_urlopen(monkeypatch, payload)
    assert [m["id"] for m in check_models.fetch_upstream()] == ["vendor/chat"]


def test_fetch_rejects_an_empty_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_urlopen(monkeypatch, {"data": []})
    with pytest.raises(ValueError):
        check_models.fetch_upstream()


def test_unreachable_upstream_exits_two_not_one(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exit 2 is load-bearing: release.sh warns and continues on it, and blocks
    # only on 1. A network blip must not read as catalog drift.
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(check_models.urllib.request, "urlopen", _boom)
    monkeypatch.setattr(sys, "argv", ["check_models.py"])
    assert check_models.main() == 2


def _stub_urlopen(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    class _Response:
        def read(self) -> bytes:
            return json.dumps(payload).encode()

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_exc: Any) -> None:
            return None

    monkeypatch.setattr(check_models.urllib.request, "urlopen", lambda *_a, **_k: _Response())
