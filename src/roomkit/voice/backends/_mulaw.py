"""ITU-T G.711 mu-law codec (NumPy lookup tables, no C dependencies).

Replaces ``audioop.lin2ulaw`` / ``audioop.ulaw2lin`` which were removed
in Python 3.13 — bit-exact with both, verified against the full 16-bit
sweep. Encode and decode are single fancy-index NumPy lookups.
"""

from __future__ import annotations

from typing import Any

from roomkit.voice.utils import _get_np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BIAS = 0x84
# Segment upper bounds in the biased 14-bit domain (Sun/CCITT g711.c).
_SEG_UEND = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)

# ---------------------------------------------------------------------------
# Encode: PCM-16 → mu-law
# ---------------------------------------------------------------------------

_ENCODE_TABLE: Any | None = None


def _linear2ulaw(pcm_val: int) -> int:
    """Canonical Sun/CCITT ``g711.c`` linear→mu-law for one 16-bit sample.

    The arithmetic ``>> 2`` runs on the *signed* value before the sign
    split — negatives round toward -inf, so ``-x`` and ``x`` may land in
    different segments. That asymmetry is part of the reference (CPython's
    ``audioop`` matched it bit for bit); a magnitude-indexed table cannot
    reproduce it, which is why the lookup below is indexed by the full
    unsigned 16-bit representation.
    """
    pcm_val >>= 2
    if pcm_val < 0:
        pcm_val = -pcm_val
        mask = 0x7F
    else:
        mask = 0xFF
    pcm_val = min(pcm_val, 8159) + (_BIAS >> 2)
    for seg, uend in enumerate(_SEG_UEND):
        if pcm_val <= uend:
            return ((seg << 4) | ((pcm_val >> (seg + 1)) & 0x0F)) ^ mask
    return 0x7F ^ mask


def _build_encode_table() -> Any:
    """Build 65536-entry lookup: uint16 sample representation → mu-law byte."""
    np = _get_np()
    table = np.empty(65536, dtype=np.uint8)
    for u in range(65536):
        table[u] = _linear2ulaw(u - 65536 if u >= 32768 else u)
    return table


def pcm16_to_mulaw(pcm_data: bytes) -> bytes:
    """Encode PCM-16 LE bytes to mu-law bytes.

    Each 2-byte sample becomes 1 mu-law byte. Bit-exact with the G.711
    reference (``audioop.lin2ulaw``). Vectorised: the per-sample table
    lookup runs as a single NumPy fancy-index instead of a Python loop,
    which holds the GIL long enough to starve realtime RTP pacing (a turn
    streams dozens of audio chunks through here on the event loop).
    """
    global _ENCODE_TABLE  # noqa: PLW0603
    if _ENCODE_TABLE is None:
        _ENCODE_TABLE = _build_encode_table()

    n = len(pcm_data) // 2
    if n == 0:
        return b""
    np = _get_np()
    idx = np.frombuffer(pcm_data[: n * 2], dtype="<u2")
    return _ENCODE_TABLE[idx].tobytes()


# ---------------------------------------------------------------------------
# Decode: mu-law → PCM-16
# ---------------------------------------------------------------------------

_DECODE_TABLE: Any | None = None


def _build_decode_table() -> Any:
    """Build 256-entry lookup: mu-law byte → signed 16-bit PCM sample."""
    np = _get_np()
    table = np.empty(256, dtype="<i2")
    for byte in range(256):
        complement = ~byte
        sign = complement & 0x80
        exponent = (complement >> 4) & 0x07
        mantissa = complement & 0x0F
        sample = ((mantissa << 3) + _BIAS) << exponent
        sample -= _BIAS
        table[byte] = -sample if sign else sample
    return table


def mulaw_to_pcm16(mulaw_data: bytes) -> bytes:
    """Decode mu-law bytes to PCM-16 LE bytes.

    Each 1-byte mu-law sample becomes 2 PCM bytes. Vectorised for the same
    reason as the encode path: the per-sample Python loop holds the GIL long
    enough to starve realtime RTP pacing, and a call streams dozens of
    frames through here on the event loop.
    """
    global _DECODE_TABLE  # noqa: PLW0603
    if _DECODE_TABLE is None:
        _DECODE_TABLE = _build_decode_table()

    if not mulaw_data:
        return b""
    np = _get_np()
    idx = np.frombuffer(mulaw_data, dtype=np.uint8)
    return _DECODE_TABLE[idx].tobytes()
