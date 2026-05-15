"""Shared Telnyx webhook signature verification.

Telnyx signs the concatenation ``timestamp|payload`` with an Ed25519
key. A correct verification also checks that ``timestamp`` is recent —
without freshness, a single captured request can be replayed forever.
"""

from __future__ import annotations

import base64
import time


def verify_telnyx_signature(
    *,
    payload: bytes,
    signature: str,
    timestamp: str | None,
    public_key: str,
    tolerance_seconds: int = 300,
    now: float | None = None,
) -> bool:
    """Verify a Telnyx webhook signature and timestamp freshness.

    Args:
        payload: Raw request body bytes.
        signature: Value of the ``Telnyx-Signature-Ed25519`` header.
        timestamp: Value of the ``Telnyx-Timestamp`` header (Unix
            seconds, as a string).
        public_key: Telnyx-provided base64 Ed25519 public key.
        tolerance_seconds: Maximum allowed clock skew between
            ``timestamp`` and the current wall clock. Default 300s
            (5 minutes). Set to a larger value only if your webhook
            ingest can buffer requests longer than that.
        now: Override the current wall clock (seconds since epoch).
            For tests only.

    Returns:
        True iff the signature is valid AND the timestamp is within
        ``tolerance_seconds`` of now. False otherwise.

    Raises:
        ImportError: If PyNaCl is not installed.
    """
    if not timestamp:
        return False

    try:
        ts_seconds = int(timestamp)
    except (TypeError, ValueError):
        return False

    current = time.time() if now is None else now
    if abs(current - ts_seconds) > tolerance_seconds:
        return False

    try:
        from nacl.signing import VerifyKey
    except ImportError as exc:
        raise ImportError(
            "PyNaCl is required for Telnyx signature verification. "
            "Install it with: pip install pynacl"
        ) from exc

    try:
        signed_payload = f"{timestamp}|".encode() + payload
        signature_bytes = base64.b64decode(signature)
        public_key_bytes = base64.b64decode(public_key)
        VerifyKey(public_key_bytes).verify(signed_payload, signature_bytes)
        return True
    except Exception:
        return False
