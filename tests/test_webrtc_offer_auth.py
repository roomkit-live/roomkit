"""Tests for HTTP-layer auth on the WebRTC /webrtc/offer endpoint.

Covers the P0 fix: unauthenticated WebRTC offers must be rejected before any
peer connection is created, and an anonymous endpoint must be opt-in.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

# fastapi is a peer dependency (the app provides it); skip when absent.
pytest.importorskip("fastapi")

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from roomkit.voice.backends._webrtc_auth import register_webrtc_offer_auth  # noqa: E402


class _FakeStream:
    """Stand-in for the vendored FastRTC Stream — records offer() calls."""

    def __init__(self) -> None:
        self.offer_calls: list[Any] = []

    async def offer(self, body: Any) -> dict[str, Any]:
        self.offer_calls.append(body)
        return {"status": "ok", "webrtc_id": body.webrtc_id}


def _backend() -> SimpleNamespace:
    return SimpleNamespace(_webrtc_auth_meta={})


def _client(auth: Any, *, allow_anonymous: bool = False) -> tuple[TestClient, _FakeStream, Any]:
    app = FastAPI()
    stream = _FakeStream()
    backend = _backend()
    register_webrtc_offer_auth(
        app, "/fastrtc", stream, backend, auth, allow_anonymous=allow_anonymous
    )
    return TestClient(app), stream, backend


_OFFER = {"type": "offer", "sdp": "v=0", "webrtc_id": "abc"}
_ICE = {
    "type": "ice-candidate",
    "candidate": {"candidate": "candidate:1 1 udp 1 1.2.3.4 5 typ host"},
    "webrtc_id": "abc",
}


def test_anonymous_without_optin_raises() -> None:
    app = FastAPI()
    with pytest.raises(ValueError, match="without authentication|allow_anonymous"):
        register_webrtc_offer_auth(
            app, "/fastrtc", _FakeStream(), _backend(), None, allow_anonymous=False
        )


def test_anonymous_optin_registers_no_guard() -> None:
    # With allow_anonymous=True and no auth, the helper adds no guarded route
    # (the vendored stream.mount would provide the plain one).
    app = FastAPI()
    register_webrtc_offer_auth(
        app, "/fastrtc", _FakeStream(), _backend(), None, allow_anonymous=True
    )
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/fastrtc/webrtc/offer" not in paths


def test_offer_rejected_when_auth_returns_none() -> None:
    async def reject(_request: Request) -> None:
        return None

    client, stream, backend = _client(reject)
    resp = client.post("/fastrtc/webrtc/offer", json=_OFFER)
    assert resp.status_code == 403
    # No peer connection work: stream.offer never called.
    assert stream.offer_calls == []
    assert backend._webrtc_auth_meta == {}


def test_offer_rejected_when_auth_raises() -> None:
    async def boom(_request: Request) -> dict[str, Any]:
        raise RuntimeError("nope")

    client, stream, _ = _client(boom)
    resp = client.post("/fastrtc/webrtc/offer", json=_OFFER)
    assert resp.status_code == 403
    assert stream.offer_calls == []


def test_offer_allowed_when_auth_succeeds() -> None:
    async def ok(request: Request) -> dict[str, Any]:
        return {"tenant": request.headers.get("x-tenant", "t1")}

    client, stream, backend = _client(ok)
    resp = client.post("/fastrtc/webrtc/offer", json=_OFFER, headers={"x-tenant": "acme"})
    assert resp.status_code == 200
    assert resp.json()["webrtc_id"] == "abc"
    # Delegated to the stream exactly once, and auth meta stashed for the factory.
    assert len(stream.offer_calls) == 1
    assert backend._webrtc_auth_meta["abc"] == {"tenant": "acme"}


def test_ice_candidate_is_also_authenticated() -> None:
    async def reject(_request: Request) -> None:
        return None

    client, stream, _ = _client(reject)
    resp = client.post("/fastrtc/webrtc/offer", json=_ICE)
    assert resp.status_code == 403
    assert stream.offer_calls == []
