"""Pure-Python ITU-T G.711 mu-law codec (no C dependencies).

Replaces ``audioop.lin2ulaw`` / ``audioop.ulaw2lin`` which were removed
in Python 3.13.  Uses precomputed lookup tables for O(1) per-sample
encode and decode.
"""

from __future__ import annotations

import struct

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BIAS = 0x84
_CLIP = 32635

# ---------------------------------------------------------------------------
# Encode: PCM-16 → mu-law
# ---------------------------------------------------------------------------

_ENCODE_TABLE: bytes | None = None


def _build_encode_table() -> bytes:
    """Build 16384-entry lookup: 14-bit unsigned magnitude → 7-bit mu-law."""
    table = bytearray(16384)
    for i in range(16384):
        sample = min(i, _CLIP) + _BIAS
        exponent = 7
        mask = 0x4000
        while exponent > 0 and not (sample & mask):
            exponent -= 1
            mask >>= 1
        mantissa = (sample >> (exponent + 3)) & 0x0F
        table[i] = ~((exponent << 4) | mantissa) & 0x7F
    return bytes(table)


def pcm16_to_mulaw(pcm_data: bytes) -> bytes:
    """Encode PCM-16 LE bytes to mu-law bytes.

    Each 2-byte sample becomes 1 mu-law byte.
    """
    global _ENCODE_TABLE  # noqa: PLW0603
    if _ENCODE_TABLE is None:
        _ENCODE_TABLE = _build_encode_table()

    n = len(pcm_data) // 2
    samples = struct.unpack(f"<{n}h", pcm_data[: n * 2])
    out = bytearray(n)
    table = _ENCODE_TABLE
    for i, s in enumerate(samples):
        sign = 0x80 if s >= 0 else 0x00
        mag = min(-s if s < 0 else s, _CLIP)
        out[i] = table[mag >> 1] | sign
    return bytes(out)


# ---------------------------------------------------------------------------
# Decode: mu-law → PCM-16
# ---------------------------------------------------------------------------

_DECODE_TABLE: tuple[int, ...] | None = None


def _build_decode_table() -> tuple[int, ...]:
    """Build 256-entry lookup: mu-law byte → signed 16-bit PCM sample."""
    table = []
    for byte in range(256):
        complement = ~byte
        sign = complement & 0x80
        exponent = (complement >> 4) & 0x07
        mantissa = complement & 0x0F
        sample = ((mantissa << 3) + _BIAS) << exponent
        sample -= _BIAS
        table.append(-sample if sign else sample)
    return tuple(table)


def mulaw_to_pcm16(mulaw_data: bytes) -> bytes:
    """Decode mu-law bytes to PCM-16 LE bytes.

    Each 1-byte mu-law sample becomes 2 PCM bytes.
    """
    global _DECODE_TABLE  # noqa: PLW0603
    if _DECODE_TABLE is None:
        _DECODE_TABLE = _build_decode_table()

    n = len(mulaw_data)
    out = bytearray(n * 2)
    table = _DECODE_TABLE
    for i in range(n):
        struct.pack_into("<h", out, i * 2, table[mulaw_data[i]])
    return bytes(out)
