"""STTLanguageLock: detect, lock to what the speaker uses, release when it stops fitting."""

from __future__ import annotations

import pytest

from roomkit import STTLanguageLock
from roomkit.voice.base import TranscriptionResult


def _final(text: str = "bonjour", language: str | None = None, confidence: float | None = 0.9):
    return TranscriptionResult(text=text, is_final=True, language=language, confidence=confidence)


class TestConstruction:
    def test_defaults(self) -> None:
        lock = STTLanguageLock()
        assert lock.detect_language == "multi"
        assert lock.prefer == {}
        assert lock.lock_after == 1
        assert lock.release_after == 2
        assert lock.min_confidence == 0.5

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"detect_language": " "},
            {"lock_after": 0},
            {"release_after": 0},
            {"min_confidence": 1.5},
            {"min_confidence": -0.1},
        ],
    )
    def test_rejects_bad_values(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            STTLanguageLock(**kwargs)

    def test_unknown_session_is_detecting(self) -> None:
        assert STTLanguageLock().language_for("s1") == "multi"


class TestLocking:
    def test_locks_on_the_first_agreeing_final(self) -> None:
        lock = STTLanguageLock(prefer={"fr": "fr-CA"})
        assert lock.observe("s1", _final(language="fr")) == "fr-CA"
        assert lock.language_for("s1") == "fr-CA"

    def test_lock_after_counts_consecutive_agreement(self) -> None:
        lock = STTLanguageLock(lock_after=2)
        assert lock.observe("s1", _final(language="fr")) == "multi"
        assert lock.observe("s1", _final(language="en")) == "multi"  # streak reset
        assert lock.observe("s1", _final(language="en")) == "en"

    def test_reported_language_without_prefer_is_used_as_is(self) -> None:
        lock = STTLanguageLock()
        assert lock.observe("s1", _final(language="es")) == "es"

    def test_nothing_reported_or_no_text_keeps_detecting(self) -> None:
        lock = STTLanguageLock()
        assert lock.observe("s1", _final(language=None)) == "multi"
        assert lock.observe("s1", _final(text="  ", language="fr")) == "multi"

    def test_sessions_are_independent(self) -> None:
        lock = STTLanguageLock()
        lock.observe("s1", _final(language="fr"))
        assert lock.language_for("s2") == "multi"


class TestReleasing:
    def _locked(self) -> STTLanguageLock:
        lock = STTLanguageLock(prefer={"fr": "fr-CA"})
        lock.observe("s1", _final(language="fr"))
        assert lock.language_for("s1") == "fr-CA"
        return lock

    def test_consecutive_empty_finals_release(self) -> None:
        lock = self._locked()
        assert lock.observe("s1", _final(text="")) == "fr-CA"
        assert lock.observe("s1", _final(text="")) == "multi"
        assert lock.language_for("s1") == "multi"

    def test_a_fitting_final_resets_the_misses(self) -> None:
        lock = self._locked()
        lock.observe("s1", _final(text=""))
        lock.observe("s1", _final(text="encore"))  # pinned stream reports nothing
        assert lock.observe("s1", _final(text="")) == "fr-CA"

    def test_low_confidence_is_a_miss(self) -> None:
        lock = self._locked()
        lock.observe("s1", _final(text="krzt", confidence=0.1))
        assert lock.observe("s1", _final(text="krzt", confidence=0.1)) == "multi"

    def test_unknown_confidence_fits(self) -> None:
        lock = self._locked()
        lock.observe("s1", _final(text="oui", confidence=None))
        lock.observe("s1", _final(text="oui", confidence=None))
        assert lock.language_for("s1") == "fr-CA"

    def test_another_reported_language_is_a_miss(self) -> None:
        lock = self._locked()
        lock.observe("s1", _final(text="hello", language="en"))
        assert lock.observe("s1", _final(text="hello", language="en")) == "multi"

    def test_the_locked_language_reported_through_prefer_fits(self) -> None:
        lock = self._locked()
        lock.observe("s1", _final(text=""))
        lock.observe("s1", _final(text="oui", language="fr"))  # -> fr-CA, fits
        assert lock.observe("s1", _final(text="")) == "fr-CA"

    def test_release_after_one(self) -> None:
        lock = STTLanguageLock(release_after=1)
        lock.observe("s1", _final(language="fr"))
        assert lock.observe("s1", _final(text="")) == "multi"

    def test_relocks_after_release(self) -> None:
        lock = self._locked()
        lock.observe("s1", _final(text=""))
        lock.observe("s1", _final(text=""))
        assert lock.observe("s1", _final(text="hello", language="en")) == "en"

    def test_forget_drops_the_session(self) -> None:
        lock = self._locked()
        lock.forget("s1")
        assert lock.language_for("s1") == "multi"
        lock.forget("never-seen")
