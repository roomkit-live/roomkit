"""HTTP-layer authentication for the WebRTC ``/webrtc/offer`` endpoint.

The vendored FastRTC ``Stream`` runs its connection auth callback only when a
WebSocket object is present. WebRTC connections arrive over an HTTP
``POST /webrtc/offer`` with no WebSocket, so without this gate that path is
unauthenticated and an ``RTCPeerConnection`` is allocated for any caller — an
auth bypass and a denial-of-service surface.

:func:`register_webrtc_offer_auth` closes that hole: it registers an
authenticated ``/webrtc/offer`` route that runs the ``auth`` callback against
the HTTP request *before* delegating to the stream, so an unauthorized offer is
rejected before any peer connection is created. It must be called *before*
``stream.mount()`` so the authenticated route takes precedence over the
vendored one.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request
from pydantic import BaseModel

from roomkit.voice.auth import AuthCallback

logger = logging.getLogger("roomkit.voice.webrtc_auth")


class _OfferBody(BaseModel):
    """Request body for ``/webrtc/offer`` — mirrors the vendored ``Stream.Body``.

    Declared here rather than imported from ``roomkit.webrtc.stream`` so the
    auth layer does not pull in aiortc/numpy.
    """

    sdp: str | None = None
    candidate: dict[str, Any] | None = None
    type: str
    webrtc_id: str


def register_webrtc_offer_auth(
    app: Any,
    path: str,
    stream: Any,
    backend: Any,
    auth: AuthCallback | None,
    *,
    allow_anonymous: bool,
) -> None:
    """Gate ``POST {path}/webrtc/offer`` with *auth* before peer creation.

    Call this BEFORE ``stream.mount(app, path)`` so the authenticated route is
    matched ahead of the vendored one. Both the SDP offer and subsequent
    ICE-candidate POSTs flow through this endpoint, so both are authenticated.

    Args:
        app: FastAPI application.
        path: Mount prefix (e.g. ``/fastrtc``).
        stream: The vendored FastRTC ``Stream`` instance to delegate to.
        backend: The FastRTC backend; authenticated metadata is stashed on its
            ``_webrtc_auth_meta`` dict keyed by ``webrtc_id`` so the session
            factory can read it via :data:`roomkit.voice.auth.auth_context`.
        auth: Async auth callback receiving the HTTP ``Request``; returns a
            metadata dict to accept or ``None`` to reject.
        allow_anonymous: When *auth* is ``None`` this must be ``True`` to expose
            the endpoint without authentication; otherwise a ``ValueError`` is
            raised at mount time to prevent an implicit anonymous WebRTC
            endpoint on the public internet.

    Raises:
        ValueError: if *auth* is ``None`` and *allow_anonymous* is ``False``.
    """
    if auth is None:
        if not allow_anonymous:
            raise ValueError(
                f"The WebRTC offer endpoint at {path}/webrtc/offer would be "
                "exposed WITHOUT authentication. Pass an `auth` callback, or "
                "set allow_anonymous=True to explicitly allow unauthenticated "
                "access (only safe on a trusted/local network)."
            )
        logger.warning(
            "WebRTC offer endpoint at %s/webrtc/offer is UNAUTHENTICATED (allow_anonymous=True)",
            path,
        )
        return

    async def _authenticated_offer(body: _OfferBody, request: Request) -> Any:
        try:
            meta = await auth(request)
        except Exception:
            logger.exception("WebRTC auth error for webrtc_id=%s", body.webrtc_id)
            raise HTTPException(status_code=403, detail="auth_error") from None
        if meta is None:
            logger.warning("WebRTC auth rejected for webrtc_id=%s", body.webrtc_id)
            raise HTTPException(status_code=403, detail="unauthorized")
        backend._webrtc_auth_meta[body.webrtc_id] = meta
        return await stream.offer(body)

    app.post(f"{path}/webrtc/offer")(_authenticated_offer)
