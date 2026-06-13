"""Curated catalog of xAI Grok Voice Agent voices.

Hand-maintained, offline list returned by ``XAIRealtimeProvider.available_voices``.
Sourced from the xAI audio API capability docs (the 5 documented built-in voices).
Voice ids are case-insensitive (``ara`` == ``Ara``). The catalog is *extensible*:
the ``voice`` field also accepts custom voice ids created via the Custom Voices
API, which won't appear here — treat this as the built-in defaults, not a closed
set.
"""

from __future__ import annotations

from roomkit.voice.realtime.provider import VoiceInfo

VOICES: list[VoiceInfo] = [
    VoiceInfo(id="eve", name="Eve", gender="female", description="Energetic, upbeat (default)"),
    VoiceInfo(id="ara", name="Ara", gender="female", description="Warm, friendly, conversational"),
    VoiceInfo(id="rex", name="Rex", gender="male", description="Confident, clear, professional"),
    VoiceInfo(id="sal", name="Sal", gender="neutral", description="Smooth, balanced, versatile"),
    VoiceInfo(id="leo", name="Leo", gender="male", description="Authoritative, decisive"),
]
