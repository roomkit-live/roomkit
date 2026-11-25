"""Voices: the catalog entry, the filters every catalog honours, and dialogue turns.

Shared by the TTS providers and the realtime (speech-to-speech) providers, which
name their voices the same way (RFC §12.2, §12.4): the ``id`` a catalog returns
is the string the provider accepts as ``voice``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel, Field


class VoiceInfo(BaseModel):
    """Metadata describing a single voice a provider offers.

    Both the curated catalog (``available_voices``) and the live query
    (``list_voices``) return these. Only ``id`` is guaranteed; the other
    fields are what the vendor reports, and are ``None`` when it reports
    nothing — never inferred from another voice (RFC §12.2).

    Attributes:
        id: Exact voice identifier passed as ``voice`` (e.g. ``"alloy"``,
            ``"Puck"``, ``"fr-ca-advisor-1"``, an ElevenLabs ``voice_id``).
        name: Human-friendly display name.
        language: BCP-47 tag if voice-specific (e.g. ``"en-US"``), or
            ``"multilingual"``, else ``None``.
        gender: ``"male"``/``"female"``/``"neutral"`` if known.
        accent: The accent the vendor names (e.g. ``"Montreal French"``).
        description: Short characterization (e.g. ``"Upbeat"``) if known.
        deprecated: Whether the provider marks the voice deprecated.
        attributes: What the vendor reports beyond the common fields, under
            its own names (persona, age, use case, category).
    """

    id: str
    name: str | None = None
    language: str | None = None
    gender: str | None = None
    accent: str | None = None
    description: str | None = None
    deprecated: bool = False
    attributes: dict[str, str] = Field(default_factory=dict)


class DialogueTurn(BaseModel):
    """One line of a scripted exchange for ``synthesize_dialogue`` (RFC §12.2).

    Attributes:
        speaker: Key into the call's ``voices`` map.
        text: What the speaker says.
        style: Delivery direction for this turn only (e.g. ``"whispering"``).
    """

    speaker: str
    text: str
    style: str | None = None


def filter_voices(
    voices: Iterable[VoiceInfo],
    *,
    language: str | None = None,
    gender: str | None = None,
    query: str | None = None,
) -> list[VoiceInfo]:
    """Keep the voices every given filter matches, the way RFC §12.2 defines them.

    ``language`` matches a tag equal to it or starting with it and a hyphen
    (``"fr"`` matches ``fr-CA``; ``fr-CA`` does not match ``fr-FR``);
    ``gender`` matches its whole value, without regard to case; ``query`` is a
    case-insensitive substring of the name or the description. A voice with no
    language or gender does not match a filter on it.
    """
    wanted_language = language.lower() if language else None
    wanted_gender = gender.lower() if gender else None
    needle = query.lower() if query else None
    kept: list[VoiceInfo] = []
    for voice in voices:
        if wanted_language and not _language_matches(voice.language, wanted_language):
            continue
        if wanted_gender and (voice.gender or "").lower() != wanted_gender:
            continue
        if needle and needle not in f"{voice.name or ''}\n{voice.description or ''}".lower():
            continue
        kept.append(voice)
    return kept


def _language_matches(tag: str | None, wanted: str) -> bool:
    if not tag:
        return False
    tag = tag.lower()
    return tag == wanted or tag.startswith(f"{wanted}-")


def check_dialogue(
    turns: Sequence[DialogueTurn],
    voices: Mapping[str, str],
    *,
    max_speakers: int,
    provider: str,
) -> list[str]:
    """Refuse, before any call, a dialogue the provider cannot voice (RFC §12.2).

    Returns the speakers in order of first appearance.

    Raises:
        NotImplementedError: The provider voices no dialogue (``max_speakers`` 0).
        ValueError: No turn, a turn naming a speaker ``voices`` does not map,
            or more distinct speakers than ``max_speakers``.
    """
    if max_speakers <= 0:
        raise NotImplementedError(f"{provider} does not synthesize dialogue")
    if not turns:
        raise ValueError("a dialogue needs at least one turn")
    speakers: list[str] = []
    for turn in turns:
        if turn.speaker not in voices:
            raise ValueError(f"speaker {turn.speaker!r} has no voice in the voices map")
        if turn.speaker not in speakers:
            speakers.append(turn.speaker)
    if len(speakers) > max_speakers:
        raise ValueError(
            f"{provider} voices at most {max_speakers} speakers per dialogue, "
            f"these turns name {len(speakers)}"
        )
    return speakers


def dialogue_transcript(turns: Sequence[DialogueTurn]) -> str:
    """The ``transcript`` of a dialogue clip: ``"<speaker>: <text>"``, one per line."""
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)
