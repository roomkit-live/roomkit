"""Equivalence guard for the vectorised realtime audio DSP.

``pcm16_to_mulaw`` / ``mulaw_to_pcm16`` and ``rms_db`` run per audio chunk
on the realtime event loop and are vectorised with NumPy — a per-sample
Python loop there holds the GIL long enough to starve RTP pacing (audible
drop-outs). These tests pin the codec to the G.711 reference two ways:

- an independent scalar transcription of Sun/CCITT ``g711.c`` (the same
  algorithm CPython's ``audioop`` used), swept over every 16-bit value;
- hardcoded reference vectors generated with ``audioop`` on Python 3.12,
  so the ground truth survives ``audioop``'s removal in 3.13.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from roomkit.voice.backends._mulaw import mulaw_to_pcm16, pcm16_to_mulaw
from roomkit.voice.utils import rms_db

_SEG_UEND = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)


def _ref_linear2ulaw(pcm_val: int) -> int:
    """Independent scalar transcription of Sun/CCITT ``g711.c`` encode."""
    pcm_val >>= 2
    if pcm_val < 0:
        pcm_val = -pcm_val
        mask = 0x7F
    else:
        mask = 0xFF
    pcm_val = min(pcm_val, 8159) + (0x84 >> 2)
    for seg, uend in enumerate(_SEG_UEND):
        if pcm_val <= uend:
            return ((seg << 4) | ((pcm_val >> (seg + 1)) & 0x0F)) ^ mask
    return 0x7F ^ mask


def _ref_ulaw2linear(byte: int) -> int:
    """Independent scalar transcription of Sun/CCITT ``g711.c`` decode."""
    complement = ~byte
    sign = complement & 0x80
    exponent = (complement >> 4) & 0x07
    mantissa = complement & 0x0F
    sample = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return -sample if sign else sample


# (pcm sample, mu-law code) pairs generated with audioop.lin2ulaw on 3.12.
_ENCODE_VECTORS = [
    (0, 0xFF),
    (1, 0xFF),
    (-1, 0x7E),
    (2, 0xFF),
    (7, 0xFE),
    (8, 0xFE),
    (33, 0xFB),
    (100, 0xF2),
    (-100, 0x72),
    (500, 0xDC),
    (-500, 0x5C),
    (1000, 0xCE),
    (8158, 0x9F),
    (8159, 0x9F),
    (8160, 0x9F),
    (-8160, 0x1F),
    (32124, 0x80),
    (32767, 0x80),
    (-32767, 0x00),
    (-32768, 0x00),
]

# (mu-law code, pcm sample) pairs generated with audioop.ulaw2lin on 3.12.
_DECODE_VECTORS = [
    (0x00, -32124),
    (0x10, -15996),
    (0x20, -7932),
    (0x30, -3900),
    (0x40, -1884),
    (0x50, -876),
    (0x60, -372),
    (0x70, -120),
    (0x80, 32124),
    (0x90, 15996),
    (0xA0, 7932),
    (0xB0, 3900),
    (0xC0, 1884),
    (0xD0, 876),
    (0xE0, 372),
    (0xF0, 120),
]


def _pcm(seed: int, n: int) -> bytes:
    return np.random.default_rng(seed).integers(-32768, 32768, size=n, dtype=np.int16).tobytes()


class TestMulawEncode:
    def test_full_sweep_matches_the_reference(self) -> None:
        """Every 16-bit value encodes exactly as g711.c does."""
        vals = range(-32768, 32768)
        pcm = struct.pack("<65536h", *vals)
        encoded = pcm16_to_mulaw(pcm)
        for i, v in enumerate(vals):
            assert encoded[i] == _ref_linear2ulaw(v), f"pcm={v}"

    def test_audioop_reference_vectors(self) -> None:
        for pcm_val, expected in _ENCODE_VECTORS:
            got = pcm16_to_mulaw(struct.pack("<h", pcm_val))[0]
            assert got == expected, f"pcm={pcm_val}: 0x{got:02X} != 0x{expected:02X}"

    @pytest.mark.parametrize("n", [0, 1, 2, 7, 100, 4800, 24000])
    def test_arbitrary_lengths(self, n: int) -> None:
        pcm = _pcm(n + 1, n)
        encoded = pcm16_to_mulaw(pcm)
        assert len(encoded) == n
        samples = struct.unpack(f"<{n}h", pcm)
        for i in (0, n // 2, n - 1) if n else ():
            assert encoded[i] == _ref_linear2ulaw(samples[i])

    def test_negative_positive_asymmetry_is_preserved(self) -> None:
        """The reference's arithmetic >>2 rounds negatives differently —
        a magnitude-indexed table would erase this; the codec must not."""
        asymmetric = [
            v for v in range(1, 8192) if _ref_linear2ulaw(v) ^ 0x80 != _ref_linear2ulaw(-v)
        ]
        assert asymmetric, "reference lost its documented asymmetry"
        v = asymmetric[0]
        pcm = struct.pack("<2h", v, -v)
        encoded = pcm16_to_mulaw(pcm)
        assert encoded[0] == _ref_linear2ulaw(v)
        assert encoded[1] == _ref_linear2ulaw(-v)


class TestMulawDecode:
    def test_all_256_codes_match_the_reference(self) -> None:
        decoded = mulaw_to_pcm16(bytes(range(256)))
        for code in range(256):
            (got,) = struct.unpack_from("<h", decoded, code * 2)
            assert got == _ref_ulaw2linear(code), f"code=0x{code:02X}"

    def test_audioop_reference_vectors(self) -> None:
        for code, expected in _DECODE_VECTORS:
            (got,) = struct.unpack("<h", mulaw_to_pcm16(bytes([code])))
            assert got == expected, f"code=0x{code:02X}: {got} != {expected}"

    def test_empty(self) -> None:
        assert mulaw_to_pcm16(b"") == b""
        assert pcm16_to_mulaw(b"") == b""

    def test_decode_then_encode_is_stable(self) -> None:
        """Decoded quantisation levels re-encode to their own code (the two
        zero codes collapse — the G.711 property audioop also had)."""
        codes = bytes(range(256))
        rt = pcm16_to_mulaw(mulaw_to_pcm16(codes))
        mismatches = [c for c in range(256) if rt[c] != c]
        assert mismatches in ([], [0x7F]), f"unexpected instability: {mismatches[:8]}"


def _ref_rms_db(data: bytes) -> float:
    """Original pure-Python RMS-dB."""
    n = len(data) // 2
    if n == 0:
        return -60.0
    samples = struct.unpack(f"<{n}h", data[: n * 2])
    rms = math.sqrt(sum(s * s for s in samples) / n) / 32768.0
    if rms < 1e-10:
        return -60.0
    return max(-60.0, 20.0 * math.log10(rms))


@pytest.mark.parametrize("n", [0, 1, 100, 4800, 24000])
def test_rms_db_matches_reference(n: int) -> None:
    pcm = _pcm(n + 7, n)
    assert rms_db(pcm) == pytest.approx(_ref_rms_db(pcm), abs=1e-9)


def test_rms_db_silence_and_full_scale() -> None:
    assert rms_db(b"") == -60.0
    assert rms_db(struct.pack("<4h", 0, 0, 0, 0)) == -60.0
    full = struct.pack("<4h", 32767, -32768, 32767, -32768)
    assert rms_db(full) == pytest.approx(_ref_rms_db(full), abs=1e-9)
