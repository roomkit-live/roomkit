"""Twilio webhook signature verification, shared by the SMS and RCS providers.

Twilio signs every webhook the same way whatever the channel — both SMS and
RCS arrive from the Messages API — so the check belongs in one place. It lived
inline in the SMS provider, which is why RCS shipped without one at all: there
was nothing to reuse, only something to copy.

Modelled on ``providers/telnyx/_signature.py``, which already does this for its
two providers.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from urllib.parse import unquote_plus

__all__ = ["verify_twilio_signature"]


def verify_twilio_signature(
    payload: bytes,
    signature: str,
    *,
    auth_token: str,
    url: str | None,
) -> bool:
    """Whether ``X-Twilio-Signature`` matches this request.

    Twilio builds the signed string from the full request URL followed by
    every POST parameter, sorted by name, concatenated as ``key + value``;
    the result is HMAC-SHA1 with the account's auth token, base64-encoded.

    Args:
        payload: Raw request body (form-encoded).
        signature: Value of the ``X-Twilio-Signature`` header.
        auth_token: The account auth token, in the clear.
        url: The exact URL Twilio called, including scheme, host and query.
            Required — the URL is part of the signed string, so verification
            is impossible without it and returns ``False`` rather than
            pretending. Behind a proxy this must be the *public* URL, not the
            one the application sees.

    Returns:
        ``True`` only when the signature verifies.
    """
    if not url or not signature:
        return False

    try:
        pairs = payload.decode().split("&")
        params = {k: unquote_plus(v) for k, v in (p.split("=", 1) for p in pairs if "=" in p)}
    except Exception:
        return False

    validation_string = url
    for key in sorted(params):
        validation_string += key + params[key]

    expected = base64.b64encode(
        hmac.new(auth_token.encode(), validation_string.encode(), hashlib.sha1).digest()
    ).decode()

    return hmac.compare_digest(expected, signature)
