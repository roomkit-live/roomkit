"""G.711 codecs with optional NumPy acceleration and a stdlib fallback."""

from __future__ import annotations

import struct
from functools import lru_cache
from typing import Any, Literal


def _encode_mulaw(sample: int) -> int:
    sample >>= 2
    mask = 0x7F if sample < 0 else 0xFF
    magnitude = min(abs(sample), 8159) + 33
    segment = max(0, magnitude.bit_length() - 6)
    if segment >= 8:
        return 0x7F ^ mask
    return ((segment << 4) | ((magnitude >> (segment + 1)) & 15)) ^ mask


def _decode_mulaw(code: int) -> int:
    code = ~code
    magnitude = (((code & 15) << 3) + 132) << ((code >> 4) & 7)
    magnitude -= 132
    return -magnitude if code & 128 else magnitude


def _encode_alaw(sample: int) -> int:
    sample >>= 3
    mask = 0x55 if sample < 0 else 0xD5
    magnitude = -sample - 1 if sample < 0 else sample
    segment = max(0, magnitude.bit_length() - 5)
    shift = 1 if segment < 2 else segment
    return ((segment << 4) | ((magnitude >> shift) & 15)) ^ mask


def _decode_alaw(code: int) -> int:
    code ^= 0x55
    segment = (code >> 4) & 7
    magnitude = (code & 15) << 4
    magnitude = magnitude + 8 if segment == 0 else (magnitude + 264) << (segment - 1)
    return magnitude if code & 128 else -magnitude


class _G711Codec:
    """Immutable lookup tables shared by transports and realtime providers.

    Tables match CPython 3.12 audioop over the full PCM16 domain. NumPy
    accelerates the hot path when available; neither codec requires it.
    """

    def __init__(self, law: Literal["mulaw", "alaw"]) -> None:
        encode = _encode_mulaw if law == "mulaw" else _encode_alaw
        decode = _decode_mulaw if law == "mulaw" else _decode_alaw
        self._encoded = bytes(encode(u - 65536 if u >= 32768 else u) for u in range(65536))
        self._decoded = tuple(struct.pack("<h", decode(code)) for code in range(256))
        self._np: Any = None
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            self._np = np
            self._encode_table = np.frombuffer(self._encoded, dtype=np.uint8)
            self._decode_table = np.frombuffer(b"".join(self._decoded), dtype="<i2")

    def encode(self, pcm: bytes) -> bytes:
        n = len(pcm) // 2
        if not n:
            return b""
        pcm = pcm[: n * 2]
        if self._np is not None:
            return self._encode_table[self._np.frombuffer(pcm, dtype="<u2")].tobytes()
        return bytes(self._encoded[value] for (value,) in struct.iter_unpack("<H", pcm))

    def decode(self, audio: bytes) -> bytes:
        if self._np is not None:
            return self._decode_table[self._np.frombuffer(audio, dtype=self._np.uint8)].tobytes()
        return b"".join(self._decoded[value] for value in audio)


@lru_cache(maxsize=2)
def _get_codec(law: Literal["mulaw", "alaw"]) -> _G711Codec:
    return _G711Codec(law)


def pcm16_to_mulaw(pcm_data: bytes) -> bytes:
    """Encode little-endian PCM16, ignoring an incomplete trailing sample."""
    return _get_codec("mulaw").encode(pcm_data)


def mulaw_to_pcm16(mulaw_data: bytes) -> bytes:
    """Decode mu-law into little-endian PCM16."""
    return _get_codec("mulaw").decode(mulaw_data)


def pcm16_to_alaw(pcm_data: bytes) -> bytes:
    """Encode little-endian PCM16, ignoring an incomplete trailing sample."""
    return _get_codec("alaw").encode(pcm_data)


def alaw_to_pcm16(alaw_data: bytes) -> bytes:
    """Decode A-law into little-endian PCM16."""
    return _get_codec("alaw").decode(alaw_data)
