"""Input-size guards for inbound audio decode paths (DoS hardening).

Server WebSocket transports receive base64-encoded audio from untrusted
clients and decode it. Without an upper bound a single frame can allocate an
arbitrarily large buffer. These helpers cap the payload before decoding.
"""

from __future__ import annotations

# Cap for a single inbound audio frame before base64 decode. Real frames are a
# few hundred bytes to a few KB (e.g. 20 ms of 8-48 kHz mu-law/PCM); 256 KiB is
# far above any legitimate frame and bounds the decode.
MAX_INBOUND_AUDIO_FRAME_BYTES = 256 * 1024


def b64_within_limit(payload: str, max_bytes: int) -> bool:
    """Return whether base64 *payload* decodes to at most *max_bytes*.

    Computed from the string length (base64 encodes 3 bytes as 4 chars), so an
    oversized payload is rejected without allocating its decoded buffer.
    """
    return (len(payload) * 3) // 4 <= max_bytes
