"""PhraseBackchannelDetector: an utterance made only of acknowledgements is a backchannel."""

from __future__ import annotations

import pytest

from roomkit.voice.pipeline.backchannel import (
    ENGLISH_BACKCHANNELS,
    BackchannelContext,
    PhraseBackchannelDetector,
)


def _is_backchannel(text: str | None, detector: PhraseBackchannelDetector | None = None) -> bool:
    detector = detector or PhraseBackchannelDetector()
    context = BackchannelContext(transcript=text, speech_duration_ms=800)
    return detector.classify(context).is_backchannel


@pytest.mark.parametrize(
    "text",
    [
        "Okay",
        "Okay.",
        "Yeah, right",
        "Okay, interesting",
        "Mm-hmm",
        "mhm",
        "Hmmm",
        "Uh huh",
        "I see",
        "oh okay",
        "Oui",
        "Ouiii",
        "D'accord",
        "d’accord",
        "C'est ça",
        "Ah bon",
        "Tout à fait",
        "Merci.",
        "Merci beaucoup",
        "Ok, merci",
        "Thank you",
        "Thanks a lot",
        "Beurk",
    ],
)
def test_acknowledgements_are_backchannels(text: str) -> None:
    assert _is_backchannel(text)


@pytest.mark.parametrize(
    "text",
    [
        "No",
        "Wait",
        "Stop",
        "Okay, and do you remember the number",
        "Attends, stop",
        "Yeah but what about Calgary",
        "Is it dangerous",
        "Merci, mais attends",
        "Non merci",
        "Thanks, but stop",
    ],
)
def test_anything_else_is_an_interruption(text: str) -> None:
    assert not _is_backchannel(text)


def test_an_acknowledgement_growing_into_a_question_stops_being_one() -> None:
    """The channel classifies each partial: the later words decide."""
    assert _is_backchannel("Okay,")
    assert not _is_backchannel("Okay, and do you")


def test_too_many_words_is_someone_talking() -> None:
    assert _is_backchannel("yeah yeah okay sure")
    assert not _is_backchannel("yeah yeah yeah okay sure")
    assert _is_backchannel("yeah yeah yeah okay sure", PhraseBackchannelDetector(max_words=6))


@pytest.mark.parametrize("text", [None, "", "  ", "..."])
def test_without_words_the_strategy_judges_on_duration(text: str | None) -> None:
    """No words, no verdict of backchannel: SEMANTIC falls back on duration."""
    decision = PhraseBackchannelDetector().classify(
        BackchannelContext(transcript=text, speech_duration_ms=300)
    )
    assert not decision.is_backchannel
    assert decision.label == "no words"


def test_phrases_can_be_replaced() -> None:
    english_only = PhraseBackchannelDetector(ENGLISH_BACKCHANNELS)
    assert _is_backchannel("okay", english_only)
    assert not _is_backchannel("ouais", english_only)
    assert _is_backchannel("tiens donc", PhraseBackchannelDetector(["tiens donc"]))
