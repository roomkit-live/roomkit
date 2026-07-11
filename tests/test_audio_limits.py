"""Tests for inbound audio decode-size guards."""

from __future__ import annotations

import base64

from roomkit.voice._limits import MAX_INBOUND_AUDIO_FRAME_BYTES, b64_within_limit


def test_small_payload_within_limit() -> None:
    payload = base64.b64encode(b"x" * 1000).decode()
    assert b64_within_limit(payload, MAX_INBOUND_AUDIO_FRAME_BYTES) is True


def test_oversized_payload_rejected() -> None:
    big = base64.b64encode(b"x" * (MAX_INBOUND_AUDIO_FRAME_BYTES + 1)).decode()
    assert b64_within_limit(big, MAX_INBOUND_AUDIO_FRAME_BYTES) is False


def test_limit_computed_from_length_without_decoding() -> None:
    # A huge string is rejected purely from its length (no decode).
    huge = "A" * (MAX_INBOUND_AUDIO_FRAME_BYTES * 2)
    assert b64_within_limit(huge, MAX_INBOUND_AUDIO_FRAME_BYTES) is False
    # 4 base64 chars decode to 3 bytes.
    assert b64_within_limit("A" * 4, 3) is True
