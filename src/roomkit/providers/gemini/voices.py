"""Curated catalog of Google Gemini Live (native-audio) voices.

Hand-maintained, offline list returned by ``GeminiLiveProvider.available_voices``.
Sourced from the Gemini API speech-generation docs — native-audio output draws
from the same 30 prebuilt TTS voices. The set is fixed (no live list endpoint);
refresh against the docs when it changes. Gender is not documented by Google, so
only the single-word characterization is captured.
"""

from __future__ import annotations

from roomkit.voice.realtime.provider import VoiceInfo

VOICES: list[VoiceInfo] = [
    VoiceInfo(id="Zephyr", name="Zephyr", description="Bright"),
    VoiceInfo(id="Puck", name="Puck", description="Upbeat"),
    VoiceInfo(id="Charon", name="Charon", description="Informative"),
    VoiceInfo(id="Kore", name="Kore", description="Firm"),
    VoiceInfo(id="Fenrir", name="Fenrir", description="Excitable"),
    VoiceInfo(id="Leda", name="Leda", description="Youthful"),
    VoiceInfo(id="Orus", name="Orus", description="Firm"),
    VoiceInfo(id="Aoede", name="Aoede", description="Breezy"),
    VoiceInfo(id="Callirrhoe", name="Callirrhoe", description="Easy-going"),
    VoiceInfo(id="Autonoe", name="Autonoe", description="Bright"),
    VoiceInfo(id="Enceladus", name="Enceladus", description="Breathy"),
    VoiceInfo(id="Iapetus", name="Iapetus", description="Clear"),
    VoiceInfo(id="Umbriel", name="Umbriel", description="Easy-going"),
    VoiceInfo(id="Algieba", name="Algieba", description="Smooth"),
    VoiceInfo(id="Despina", name="Despina", description="Smooth"),
    VoiceInfo(id="Erinome", name="Erinome", description="Clear"),
    VoiceInfo(id="Algenib", name="Algenib", description="Gravelly"),
    VoiceInfo(id="Rasalgethi", name="Rasalgethi", description="Informative"),
    VoiceInfo(id="Laomedeia", name="Laomedeia", description="Upbeat"),
    VoiceInfo(id="Achernar", name="Achernar", description="Soft"),
    VoiceInfo(id="Alnilam", name="Alnilam", description="Firm"),
    VoiceInfo(id="Schedar", name="Schedar", description="Even"),
    VoiceInfo(id="Gacrux", name="Gacrux", description="Mature"),
    VoiceInfo(id="Pulcherrima", name="Pulcherrima", description="Forward"),
    VoiceInfo(id="Achird", name="Achird", description="Friendly"),
    VoiceInfo(id="Zubenelgenubi", name="Zubenelgenubi", description="Casual"),
    VoiceInfo(id="Vindemiatrix", name="Vindemiatrix", description="Gentle"),
    VoiceInfo(id="Sadachbia", name="Sadachbia", description="Lively"),
    VoiceInfo(id="Sadaltager", name="Sadaltager", description="Knowledgeable"),
    VoiceInfo(id="Sulafat", name="Sulafat", description="Warm"),
]
