"""Curated catalog of ElevenLabs default (premade) voices.

Hand-maintained, offline list returned by ``ElevenLabsRealtimeProvider.available_voices``.
Voice ids are verified ElevenLabs ``voice_id`` strings pulled from the public
``GET /v1/voices`` endpoint. ElevenLabs rotates defaults in and out of "legacy",
so treat ``ElevenLabsRealtimeProvider.list_voices()`` (live ``client.voices``) as
the source of truth; this is an offline fallback.
"""

from __future__ import annotations

from roomkit.voice.realtime.provider import VoiceInfo

VOICES: list[VoiceInfo] = [
    VoiceInfo(
        id="CwhRBWXzGAHq8TQ4Fs17", name="Roger", gender="male", description="laid-back, casual"
    ),
    VoiceInfo(
        id="EXAVITQu4vr4xnSDxMaL", name="Sarah", gender="female", description="confident, warm"
    ),
    VoiceInfo(
        id="FGY2WhTYpPnrIDTdsKH5", name="Laura", gender="female", description="upbeat, quirky"
    ),
    VoiceInfo(
        id="IKne3meq5aSn9XLyUdCD",
        name="Charlie",
        gender="male",
        description="confident, energetic",
    ),
    VoiceInfo(
        id="JBFqnCBsd6RMkjVDRZzb", name="George", gender="male", description="warm narration"
    ),
    VoiceInfo(
        id="N2lVS1w4EtoT3dr4eOWO", name="Callum", gender="male", description="gravelly, edgy"
    ),
    VoiceInfo(
        id="SAz9YHcvj6GT2YYXdXww", name="River", gender="neutral", description="relaxed, neutral"
    ),
    VoiceInfo(id="SOYHLrjzK2X1ezoPC6cr", name="Harry", gender="male", description="animated"),
    VoiceInfo(
        id="TX3LPaxmHKxFdv7VOQHJ", name="Liam", gender="male", description="young, energetic"
    ),
    VoiceInfo(
        id="Xb7hH8MSUJpSbSDYk0k2", name="Alice", gender="female", description="clear, friendly"
    ),
    VoiceInfo(
        id="XrExE9yKIg1WjnnlVkGX", name="Matilda", gender="female", description="professional alto"
    ),
    VoiceInfo(id="bIHbv24MWmeRgasZH58o", name="Will", gender="male", description="conversational"),
    VoiceInfo(
        id="cgSgspJ2msm6clMCkdW9", name="Jessica", gender="female", description="young, playful"
    ),
    VoiceInfo(id="cjVigY5qzO86Huf0OWal", name="Eric", gender="male", description="smooth tenor"),
    VoiceInfo(
        id="hpp4J3VqNfWAUOO0d1Us", name="Bella", gender="female", description="warm, bright"
    ),
    VoiceInfo(
        id="iP95p4xoKVk53GoZ742B",
        name="Chris",
        gender="male",
        description="natural, down-to-earth",
    ),
    VoiceInfo(
        id="nPczCjzI2devNBz1zQrb", name="Brian", gender="male", description="resonant, comforting"
    ),
    VoiceInfo(
        id="onwK4e9ZLuTAKqWW03F9", name="Daniel", gender="male", description="broadcast/news"
    ),
    VoiceInfo(
        id="pFZP5JQG7iQjIQuC4Bku", name="Lily", gender="female", description="velvety narration"
    ),
    VoiceInfo(
        id="pNInz6obpgDQGcFmaJgB", name="Adam", gender="male", description="bright, deep tenor"
    ),
    VoiceInfo(
        id="pqHfZKP75CvOlQylNhV4", name="Bill", gender="male", description="friendly, comforting"
    ),
]
