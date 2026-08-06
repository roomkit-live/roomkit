"""Curated catalog of Deepgram Aura voices.

Hand-maintained, offline list returned by ``DeepgramAgentProvider.available_voices``.
Sourced from Deepgram's TTS model documentation. Ids are the values passed as the
``voice`` parameter (e.g. ``connect(voice="aura-2-thalia-en")``), which the provider
puts in ``agent.speak.provider.model``.

Aura-2 is the current generation and is listed first. The twelve Aura-1 voices are
kept, flagged ``deprecated=True``: they still resolve, and Deepgram's own API
reference still shows ``aura-asteria-en`` in its default payload, so a caller who
inherits that id should find it here rather than conclude it is unsupported.
"""

from __future__ import annotations

from roomkit.voice.realtime.provider import VoiceInfo

# (name, locale, gender, description) — the model id is derived as
# "aura-2-{name}-{language}", where language is the locale's first segment.
_AURA2: tuple[tuple[str, str, str, str], ...] = (
    # English
    ("thalia", "en-us", "female", "Clear, confident, energetic, enthusiastic"),
    ("andromeda", "en-us", "female", "Casual, expressive, comfortable"),
    ("helena", "en-us", "female", "Caring, natural, positive, friendly, raspy"),
    ("apollo", "en-us", "male", "Confident, comfortable, casual"),
    ("arcas", "en-us", "male", "Natural, smooth, clear, comfortable"),
    ("aries", "en-us", "male", "Warm, energetic, caring"),
    ("amalthea", "en-ph", "female", "Engaging, natural, cheerful"),
    ("asteria", "en-us", "female", "Clear, confident, knowledgeable, energetic"),
    ("athena", "en-us", "female", "Calm, smooth, professional"),
    ("atlas", "en-us", "male", "Enthusiastic, confident, approachable, friendly"),
    ("aurora", "en-us", "female", "Cheerful, expressive, energetic"),
    ("callista", "en-us", "female", "Clear, energetic, professional, smooth"),
    ("cora", "en-us", "female", "Smooth, melodic, caring"),
    ("cordelia", "en-us", "female", "Approachable, warm, polite"),
    ("delia", "en-us", "female", "Casual, friendly, cheerful, breathy"),
    ("draco", "en-gb", "male", "Warm, approachable, trustworthy, baritone"),
    ("electra", "en-us", "female", "Professional, engaging, knowledgeable"),
    ("harmonia", "en-us", "female", "Empathetic, clear, calm, confident"),
    ("hera", "en-us", "female", "Smooth, warm, professional"),
    ("hermes", "en-us", "male", "Expressive, engaging, professional"),
    ("hyperion", "en-au", "male", "Caring, warm, empathetic"),
    ("iris", "en-us", "female", "Cheerful, positive, approachable"),
    ("janus", "en-us", "female", "Southern, smooth, trustworthy"),
    ("juno", "en-us", "female", "Natural, engaging, melodic, breathy"),
    ("jupiter", "en-us", "male", "Expressive, knowledgeable, baritone"),
    ("luna", "en-us", "female", "Friendly, natural, engaging"),
    ("mars", "en-us", "male", "Smooth, patient, trustworthy, baritone"),
    ("minerva", "en-us", "female", "Positive, friendly, natural"),
    ("neptune", "en-us", "male", "Professional, patient, polite"),
    ("odysseus", "en-us", "male", "Calm, smooth, comfortable, professional"),
    ("ophelia", "en-us", "female", "Expressive, enthusiastic, cheerful"),
    ("orion", "en-us", "male", "Approachable, comfortable, calm, polite"),
    ("orpheus", "en-us", "male", "Professional, clear, confident, trustworthy"),
    ("pandora", "en-gb", "female", "Smooth, calm, melodic, breathy"),
    ("phoebe", "en-us", "female", "Energetic, warm, casual"),
    ("pluto", "en-us", "male", "Smooth, calm, empathetic, baritone"),
    ("saturn", "en-us", "male", "Knowledgeable, confident, baritone"),
    ("selene", "en-us", "female", "Expressive, engaging, energetic"),
    ("theia", "en-au", "female", "Expressive, polite, sincere"),
    ("vesta", "en-us", "female", "Natural, expressive, patient, empathetic"),
    ("zeus", "en-us", "male", "Deep, trustworthy, smooth"),
    # Spanish
    ("celeste", "es-co", "female", "Clear, energetic, positive, friendly"),
    ("estrella", "es-mx", "female", "Approachable, natural, calm, comfortable"),
    ("nestor", "es-es", "male", "Calm, professional, approachable, clear"),
    ("sirio", "es-mx", "male", "Calm, professional, comfortable, baritone"),
    ("carina", "es-es", "female", "Professional, raspy, energetic, breathy"),
    ("alvaro", "es-es", "male", "Calm, professional, clear, knowledgeable"),
    ("diana", "es-es", "female", "Professional, confident, expressive, polite"),
    ("aquila", "es-419", "male", "Expressive, enthusiastic, confident, casual"),
    ("selena", "es-419", "female", "Approachable, casual, friendly, calm"),
    ("javier", "es-mx", "male", "Approachable, professional, friendly, calm"),
    ("agustina", "es-es", "female", "Calm, clear, expressive, knowledgeable"),
    ("antonia", "es-ar", "female", "Approachable, enthusiastic, friendly, natural"),
    ("gloria", "es-co", "female", "Casual, clear, expressive, natural, smooth"),
    ("luciano", "es-mx", "male", "Charismatic, cheerful, energetic, expressive"),
    ("olivia", "es-mx", "female", "Breathy, calm, casual, expressive, warm"),
    ("silvia", "es-es", "female", "Charismatic, clear, expressive, natural, warm"),
    ("valerio", "es-mx", "male", "Deep, knowledgeable, natural, polite"),
    # Dutch
    ("rhea", "nl-nl", "female", "Caring, knowledgeable, positive, smooth, warm"),
    ("sander", "nl-nl", "male", "Calm, clear, deep, professional, smooth"),
    ("beatrix", "nl-nl", "female", "Cheerful, enthusiastic, friendly, trustworthy"),
    ("daphne", "nl-nl", "female", "Calm, clear, confident, professional, smooth"),
    ("cornelia", "nl-nl", "female", "Approachable, friendly, polite, positive, warm"),
    ("hestia", "nl-nl", "female", "Approachable, caring, expressive, friendly"),
    ("lars", "nl-nl", "male", "Breathy, casual, comfortable, sincere, trustworthy"),
    ("roman", "nl-nl", "male", "Calm, casual, deep, natural, patient"),
    ("leda", "nl-nl", "female", "Caring, comfortable, empathetic, friendly, sincere"),
    # French
    ("agathe", "fr-fr", "female", "Charismatic, cheerful, enthusiastic, friendly"),
    ("hector", "fr-fr", "male", "Confident, empathetic, expressive, friendly, patient"),
    # German
    ("julius", "de-de", "male", "Casual, cheerful, engaging, expressive, friendly"),
    ("viktoria", "de-de", "female", "Charismatic, cheerful, enthusiastic, friendly, warm"),
    ("elara", "de-de", "female", "Calm, clear, natural, patient, trustworthy"),
    ("aurelia", "de-de", "female", "Approachable, casual, comfortable, natural, sincere"),
    ("lara", "de-de", "female", "Caring, cheerful, empathetic, expressive, warm"),
    ("fabian", "de-de", "male", "Confident, knowledgeable, natural, polite, professional"),
    ("kara", "de-de", "female", "Caring, empathetic, expressive, professional, warm"),
    # Italian
    ("livia", "it-it", "female", "Approachable, cheerful, clear, engaging, expressive"),
    ("dionisio", "it-it", "male", "Confident, engaging, friendly, melodic, positive"),
    ("melia", "it-it", "female", "Clear, comfortable, engaging, friendly, natural"),
    ("elio", "it-it", "male", "Breathy, calm, professional, smooth, trustworthy"),
    ("flavio", "it-it", "male", "Confident, deep, empathetic, professional, trustworthy"),
    ("maia", "it-it", "female", "Caring, energetic, expressive, professional, warm"),
    ("cinzia", "it-it", "female", "Approachable, friendly, smooth, trustworthy, warm"),
    ("cesare", "it-it", "male", "Clear, empathetic, knowledgeable, natural, smooth"),
    ("perseo", "it-it", "male", "Casual, clear, natural, polite, smooth"),
    ("demetra", "it-it", "female", "Calm, comfortable, patient"),
    # Japanese
    ("fujin", "ja-jp", "male", "Calm, confident, knowledgeable, professional, smooth"),
    ("izanami", "ja-jp", "female", "Approachable, clear, knowledgeable, polite"),
    ("uzume", "ja-jp", "female", "Approachable, clear, polite, professional, trustworthy"),
    ("ebisu", "ja-jp", "male", "Calm, deep, natural, patient, sincere"),
    ("ama", "ja-jp", "female", "Casual, comfortable, confident, knowledgeable, natural"),
)

# Aura-1 — superseded by Aura-2 but still resolvable. Ids are "aura-{name}-en".
_AURA1: tuple[tuple[str, str, str, str], ...] = (
    ("asteria", "en-us", "female", "Clear, confident, knowledgeable, energetic"),
    ("luna", "en-us", "female", "Friendly, natural, engaging"),
    ("stella", "en-us", "female", "Clear, professional, engaging"),
    ("athena", "en-gb", "female", "Calm, smooth, professional"),
    ("hera", "en-us", "female", "Smooth, warm, professional"),
    ("orion", "en-us", "male", "Approachable, comfortable, calm, polite"),
    ("arcas", "en-us", "male", "Natural, smooth, clear, comfortable"),
    ("perseus", "en-us", "male", "Confident, professional, clear"),
    ("angus", "en-ie", "male", "Warm, friendly, natural"),
    ("orpheus", "en-us", "male", "Professional, clear, confident, trustworthy"),
    ("helios", "en-gb", "male", "Professional, clear, confident"),
    ("zeus", "en-us", "male", "Deep, trustworthy, smooth"),
)


def _voices() -> list[VoiceInfo]:
    out: list[VoiceInfo] = []
    for name, locale, gender, description in _AURA2:
        language = locale.split("-")[0]
        out.append(
            VoiceInfo(
                id=f"aura-2-{name}-{language}",
                name=name,
                language=locale,
                gender=gender,
                description=description,
            )
        )
    for name, locale, gender, description in _AURA1:
        out.append(
            VoiceInfo(
                id=f"aura-{name}-en",
                name=name,
                language=locale,
                gender=gender,
                description=description,
                deprecated=True,
            )
        )
    return out


VOICES: list[VoiceInfo] = _voices()
