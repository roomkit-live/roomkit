"""Curated catalog of NVIDIA PersonaPlex voice prompts.

Hand-maintained, offline list returned by ``PersonaPlexRealtimeProvider.available_voices``.
Sourced from the NVIDIA/personaplex README — the model ships a fixed set of voice
embedding prompts. Ids are the ``.pt`` prompt filenames the provider passes as the
``voice`` parameter (e.g. ``connect(voice="NATF2.pt")``). The hosted commercial
PersonaPlex API may expose a different catalog.
"""

from __future__ import annotations

from roomkit.voice.realtime.provider import VoiceInfo


def _voices() -> list[VoiceInfo]:
    out: list[VoiceInfo] = []
    for prefix, kind in (("NAT", "Natural"), ("VAR", "Variety")):
        # Natural ships 0-3 per gender; Variety ships 0-4.
        count = 4 if prefix == "NAT" else 5
        for letter, gender in (("F", "female"), ("M", "male")):
            for i in range(count):
                vid = f"{prefix}{letter}{i}"
                out.append(VoiceInfo(id=f"{vid}.pt", name=vid, gender=gender, description=kind))
    return out


VOICES: list[VoiceInfo] = _voices()
