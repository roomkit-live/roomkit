"""H.264 Annex B helpers shared by the encoder and the RTP-based video backends."""

from __future__ import annotations

_START_CODES = (b"\x00\x00\x00\x01", b"\x00\x00\x01")


def split_annex_b(data: bytes) -> list[bytes]:
    """Split an Annex B byte stream into its NAL units (start codes removed)."""
    nals: list[bytes] = []
    i = 0
    start = -1
    while i < len(data):
        if i + 4 <= len(data) and data[i : i + 4] == b"\x00\x00\x00\x01":
            if start >= 0:
                nals.append(data[start:i])
            start = i + 4
            i += 4
        elif i + 3 <= len(data) and data[i : i + 3] == b"\x00\x00\x01":
            if start >= 0:
                nals.append(data[start:i])
            start = i + 3
            i += 3
        else:
            i += 1
    if start >= 0 and start < len(data):
        nals.append(data[start:])
    return nals


def nal_units(data: bytes) -> list[bytes]:
    """The NAL units of one encoded frame.

    *data* is either a whole access unit in Annex B form (starting with a
    start code) or a single raw NAL unit, which H.264's emulation prevention
    keeps free of start codes.
    """
    if data.startswith(_START_CODES):
        return split_annex_b(data)
    return [data]


def access_unit(nals: list[bytes]) -> bytes:
    """One encoded frame of *nals*, as :func:`nal_units` reads it back: a single
    NAL unit as it is, several in Annex B form."""
    if len(nals) == 1:
        return nals[0]
    return b"".join(_START_CODES[0] + nal for nal in nals)
