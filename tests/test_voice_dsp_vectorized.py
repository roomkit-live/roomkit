"""Equivalence guard for the vectorised realtime audio DSP.

``pcm16_to_mulaw`` and ``rms_db`` run per audio chunk on the realtime event
loop and are vectorised with NumPy — a per-sample Python loop there holds the
GIL long enough to starve RTP pacing (audible drop-outs). These tests pin
byte-/value-exact equivalence against scalar reference implementations so the
vectorisation can never silently alter the audio.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from roomkit.voice.backends._mulaw import _CLIP, _build_encode_table, pcm16_to_mulaw
from roomkit.voice.utils import rms_db


def _ref_mulaw(pcm: bytes) -> bytes:
    """Original pure-Python G.711 encode."""
    table = _build_encode_table()
    n = len(pcm) // 2
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    out = bytearray(n)
    for i, s in enumerate(samples):
        sign = 0x80 if s >= 0 else 0x00
        mag = min(-s if s < 0 else s, _CLIP)
        out[i] = table[mag >> 1] | sign
    return bytes(out)


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


def _pcm(seed: int, n: int) -> bytes:
    return np.random.default_rng(seed).integers(-32768, 32768, size=n, dtype=np.int16).tobytes()


@pytest.mark.parametrize("n", [0, 1, 2, 7, 100, 4800, 24000])
def test_mulaw_byte_exact(n: int) -> None:
    pcm = _pcm(n + 1, n)
    assert pcm16_to_mulaw(pcm) == _ref_mulaw(pcm)


def test_mulaw_edge_values() -> None:
    # int16 min/max, zero, ±1, ±clip boundary.
    pcm = struct.pack("<8h", -32768, 32767, 0, -1, 1, _CLIP, -_CLIP, 12345)
    assert pcm16_to_mulaw(pcm) == _ref_mulaw(pcm)


@pytest.mark.parametrize("n", [0, 1, 100, 4800, 24000])
def test_rms_db_matches_reference(n: int) -> None:
    pcm = _pcm(n + 7, n)
    assert rms_db(pcm) == pytest.approx(_ref_rms_db(pcm), abs=1e-9)


def test_rms_db_silence_and_full_scale() -> None:
    assert rms_db(b"") == -60.0
    assert rms_db(struct.pack("<4h", 0, 0, 0, 0)) == -60.0
    full = struct.pack("<4h", 32767, -32768, 32767, -32768)
    assert rms_db(full) == pytest.approx(_ref_rms_db(full), abs=1e-9)
