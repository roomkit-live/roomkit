"""Curated catalog of OpenAI Realtime API voices.

Hand-maintained, offline list returned by ``OpenAIRealtimeProvider.available_voices``.
Sourced from the OpenAI Realtime conversations guide. The Realtime ``voice``
parameter is a fixed enum — there is no live list endpoint — so refresh this
against the docs when the set changes.
"""

from __future__ import annotations

from roomkit.voice.realtime.provider import VoiceInfo

VOICES: list[VoiceInfo] = [
    VoiceInfo(id="marin", name="Marin", description="Recommended quality; realtime-exclusive"),
    VoiceInfo(id="cedar", name="Cedar", description="Recommended quality; realtime-exclusive"),
    VoiceInfo(id="alloy", name="Alloy"),
    VoiceInfo(id="echo", name="Echo"),
    VoiceInfo(id="shimmer", name="Shimmer"),
    VoiceInfo(id="ash", name="Ash"),
    VoiceInfo(id="ballad", name="Ballad"),
    VoiceInfo(id="coral", name="Coral"),
    VoiceInfo(id="sage", name="Sage"),
    VoiceInfo(id="verse", name="Verse"),
]
