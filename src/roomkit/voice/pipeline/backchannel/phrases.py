"""A backchannel detector that knows the acknowledgements people say."""

from __future__ import annotations

import re
from collections.abc import Iterable

from roomkit.voice.pipeline.backchannel.base import (
    BackchannelContext,
    BackchannelDecision,
    BackchannelDetector,
)

ENGLISH_BACKCHANNELS: tuple[str, ...] = (
    "ok",
    "okay",
    "yeah",
    "yes",
    "yep",
    "yup",
    "sure",
    "right",
    "alright",
    "all right",
    "mm",
    "mhm",
    "mm-hmm",
    "hmm",
    "uh-huh",
    "aha",
    "ah",
    "oh",
    "i see",
    "got it",
    "cool",
    "nice",
    "great",
    "exactly",
    "interesting",
    "true",
    "indeed",
    "of course",
    "wow",
    "thanks",
    "thank you",
    "thanks a lot",
    "thank you so much",
    "yuck",
)
"""Acknowledgements, listener fillers and reactions in English: thanks included, said
to what the bot just said or did, not to stop it."""

FRENCH_BACKCHANNELS: tuple[str, ...] = (
    "oui",
    "ouais",
    "ok",
    "okay",
    "d'accord",
    "ah",
    "oh",
    "ah bon",
    "hum",
    "hm",
    "mm",
    "mmh",
    "bon",
    "bien",
    "très bien",
    "super",
    "cool",
    "parfait",
    "intéressant",
    "exact",
    "exactement",
    "voilà",
    "effectivement",
    "en effet",
    "tout à fait",
    "je vois",
    "c'est ça",
    "merci",
    "merci beaucoup",
    "merci bien",
    "berk",
    "beurk",
)
"""Acknowledgements, listener fillers and reactions in French: thanks included (live, a
« Merci » to « je m'occupe de ça » cut the result that came next, RMK-555)."""

_WORD = re.compile(r"[^\W_]+")  # letters and digits: apostrophes and hyphens split words
_REPEATED = re.compile(r"(\w)\1+")


def _words(text: str) -> tuple[str, ...]:
    """Lowercase words, each repeated letter written once.

    Applied to the phrases and the transcript alike, so a held sound matches
    however the STT spells it: "hmm", "hmmm" and "hm" are one word, as are
    "oui" and "ouiii".
    """
    return tuple(_WORD.findall(_REPEATED.sub(r"\1", text.lower())))


class PhraseBackchannelDetector(BackchannelDetector):
    """A backchannel is an utterance made only of known acknowledgements.

    "Okay", "mm-hmm", "yeah, right" or "d'accord" let the bot keep talking;
    "okay, and what about Calgary?" does not, since "and what about Calgary"
    is not an acknowledgement. The channel classifies each partial transcript
    of the speech as it grows, so words added after an acknowledgement still
    cut in.

    The detector reads words. Without any (the transcript is empty or not
    there yet), it judges no utterance a backchannel and the strategy falls
    back on speech duration, as ``CONFIRMED`` does (RFC §12.3.13).

    Args:
        phrases: The acknowledgements, matched case-insensitively and
            regardless of punctuation. Defaults to English and French.
        max_words: Longer utterances are never backchannels, whatever their
            words ("yeah yeah yeah, okay, sure, right" is someone talking).
    """

    def __init__(
        self,
        phrases: Iterable[str] = ENGLISH_BACKCHANNELS + FRENCH_BACKCHANNELS,
        *,
        max_words: int = 4,
    ) -> None:
        self._phrases = frozenset(words for phrase in phrases if (words := _words(phrase)))
        self._longest = max((len(p) for p in self._phrases), default=0)
        self._max_words = max_words

    @property
    def name(self) -> str:
        return "PhraseBackchannelDetector"

    def classify(self, context: BackchannelContext) -> BackchannelDecision:
        words = _words(context.transcript or "")
        if not words:
            return BackchannelDecision(is_backchannel=False, confidence=0.5, label="no words")
        if len(words) <= self._max_words and self._made_of_phrases(words):
            return BackchannelDecision(is_backchannel=True, label="acknowledgement")
        return BackchannelDecision(is_backchannel=False, label="interruption")

    def _made_of_phrases(self, words: tuple[str, ...]) -> bool:
        """Whether *words* split entirely into known phrases."""
        reachable = [True] + [False] * len(words)
        for start in range(len(words)):
            if not reachable[start]:
                continue
            for size in range(1, min(self._longest, len(words) - start) + 1):
                if words[start : start + size] in self._phrases:
                    reachable[start + size] = True
        return reachable[-1]
