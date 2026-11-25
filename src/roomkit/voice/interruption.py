"""Interruption (barge-in) strategy configuration and handler."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.pipeline.backchannel.base import BackchannelDetector

logger = logging.getLogger("roomkit.voice.interruption")

_PENDING_LOOK_AGAIN_MS = 100
"""How soon a second look at the words follows a pending verdict, when the
caller looks again on a timer rather than on each partial transcript."""


@unique
class InterruptionStrategy(StrEnum):
    """Strategy for handling user interruptions during TTS playback."""

    IMMEDIATE = "immediate"
    """Interrupt as soon as speech is detected."""

    CONFIRMED = "confirmed"
    """Wait for min_speech_ms before confirming the interruption."""

    SEMANTIC = "semantic"
    """Use backchannel detection to distinguish interruptions from
    acknowledgements (e.g. 'uh-huh')."""

    DISABLED = "disabled"
    """Ignore user speech during playback."""


@dataclass
class InterruptionConfig:
    """Configuration for interruption (barge-in) behaviour."""

    strategy: InterruptionStrategy = InterruptionStrategy.CONFIRMED
    """Which strategy to use."""

    min_speech_ms: int = 300
    """Minimum speech duration (ms) before triggering (used by CONFIRMED)."""

    allow_during_first_ms: int = 0
    """If > 0, only allow interruptions once this many ms of audio were played
    (synthesis latency does not count)."""

    flush_partial_tts: bool = True
    """Whether to flush any buffered TTS audio on interruption."""

    keep_partial_transcript: bool = True
    """Whether to keep the partial transcript when interrupted."""

    backchannel_detector: BackchannelDetector | None = None
    """Optional backchannel detector for SEMANTIC strategy."""

    transcript_wait_ms: int = 1000
    """SEMANTIC: how long speech a streaming STT is transcribing may run
    without words before it is judged on duration alone (RFC §12.3.13). The
    wait is ``max(min_speech_ms, transcript_wait_ms)``."""

    echo_guard_ms: int = 6000
    """While a session has played less than this much response audio in all
    (ms), speech whose words are only words of the response being played,
    consecutive and in order, is that response's echo: it does not interrupt,
    and its final transcript is no turn (RFC §12.3.13). An echo canceller
    learns the room from the responses it hears; until it has, their opening
    words come back through the microphone. Past it, someone repeating the
    response's words ("the sword" to "the sword or the shield?") cuts in as
    anyone does. 0 turns the guard off."""


@dataclass
class InterruptionDecision:
    """Result of evaluating whether an interruption should proceed."""

    should_interrupt: bool
    """Whether to interrupt playback."""

    is_backchannel: bool = False
    """True if the utterance was classified as a backchannel."""

    is_echo: bool = False
    """True if the utterance was taken for the response's own echo
    (``InterruptionConfig.echo_guard_ms``): it is no turn either."""

    reason: str = ""
    """Human-readable reason for the decision."""

    pending_confirmation: bool = False
    """True when the speech is not yet long enough to interrupt, but would be
    if it sustains (RFC §12.6 CONFIRMED). The caller must re-evaluate after
    ``confirm_after_ms``; without that second look CONFIRMED can never fire,
    because the first evaluation happens at speech onset with no duration."""

    confirm_after_ms: int = 0
    """How much longer the speech must sustain before re-evaluating. Only
    meaningful when ``pending_confirmation`` is True."""

    awaiting_transcript: bool = False
    """True when SEMANTIC had nothing to classify yet: no words and too short
    a speech for a duration-only judgement, or a partial transcript the
    detector gave no verdict on (``BackchannelDecision.pending``: "Ou" of
    "Ouais"). The caller transcribes the held speech and re-evaluates with
    each partial transcript; the ``pending_confirmation`` second look
    classifies on duration alone when no words arrive in time, and looks
    at the words again after a pending verdict."""

    queue_speech: bool = False
    """True when the speech must be queued and delivered once playback ends
    rather than discarded (RFC §12.6 DISABLED: "user speech is queued until
    the bot finishes")."""


class InterruptionHandler:
    """Evaluates whether a user's speech during TTS should trigger an interruption.

    Delegates to the configured InterruptionStrategy:
    - IMMEDIATE: always interrupt
    - CONFIRMED: interrupt only if speech_duration_ms >= min_speech_ms
    - SEMANTIC: use BackchannelDetector — backchannel → no interrupt
    - DISABLED: never interrupt
    """

    def __init__(
        self,
        config: InterruptionConfig,
        *,
        backchannel_detector: BackchannelDetector | None = None,
    ) -> None:
        self._config = config
        self._backchannel_detector = backchannel_detector or config.backchannel_detector

    @property
    def config(self) -> InterruptionConfig:
        return self._config

    def evaluate(
        self,
        *,
        playback_position_ms: int,
        speech_duration_ms: int = 0,
        speech_text: str = "",
        transcript_expected: bool = False,
        response_text: str = "",
        session_played_ms: int | None = None,
        speech_final: bool = True,
    ) -> InterruptionDecision:
        """Evaluate whether an interruption should proceed.

        Args:
            playback_position_ms: How far into TTS playback (ms).
            speech_duration_ms: How long the user has been speaking (ms).
            speech_text: Transcribed text of the speech (for SEMANTIC).
            transcript_expected: A streaming STT is transcribing this speech,
                so SEMANTIC waits up to ``transcript_wait_ms`` for its words.
            response_text: The text of the response being played, as far as
                it was handed to the TTS, for the echo guard.
            session_played_ms: Response audio the session played in all, this
                playback included; ``playback_position_ms`` when not given.
            speech_final: Whether *speech_text* is the whole utterance;
                ``False`` for a partial transcript, whose last word may be
                a piece of the word being said (SEMANTIC).

        Returns:
            An InterruptionDecision.
        """
        played = playback_position_ms if session_played_ms is None else session_played_ms
        if (
            played < self._config.echo_guard_ms
            and speech_text
            and echoes(speech_text, response_text, cut_short=not speech_final)
        ):
            return InterruptionDecision(
                should_interrupt=False,
                is_echo=True,
                reason="echo of the response",
            )

        # Check allow_during_first_ms constraint
        if (
            self._config.allow_during_first_ms > 0
            and playback_position_ms < self._config.allow_during_first_ms
        ):
            return InterruptionDecision(
                should_interrupt=False,
                reason="playback too early",
            )

        strategy = self._config.strategy

        if strategy == InterruptionStrategy.DISABLED:
            return InterruptionDecision(
                should_interrupt=False,
                queue_speech=True,
                reason="interruptions disabled",
            )

        if strategy == InterruptionStrategy.IMMEDIATE:
            return InterruptionDecision(
                should_interrupt=True,
                reason="immediate strategy",
            )

        if strategy == InterruptionStrategy.CONFIRMED:
            if speech_duration_ms >= self._config.min_speech_ms:
                return InterruptionDecision(
                    should_interrupt=True,
                    reason="speech confirmed",
                )
            return InterruptionDecision(
                should_interrupt=False,
                pending_confirmation=True,
                confirm_after_ms=self._config.min_speech_ms - speech_duration_ms,
                reason="speech too short",
            )

        if strategy == InterruptionStrategy.SEMANTIC:
            if self._backchannel_detector is None:
                # No detector configured — fall back to confirmed behaviour
                logger.warning(
                    "SEMANTIC strategy but no backchannel_detector; falling back to CONFIRMED"
                )
                if speech_duration_ms >= self._config.min_speech_ms:
                    return InterruptionDecision(
                        should_interrupt=True,
                        reason="semantic fallback (no detector)",
                    )
                return InterruptionDecision(
                    should_interrupt=False,
                    pending_confirmation=True,
                    confirm_after_ms=self._config.min_speech_ms - speech_duration_ms,
                    reason="speech too short (semantic fallback)",
                )

            wait_ms = self._config.min_speech_ms
            if transcript_expected:
                wait_ms = max(wait_ms, self._config.transcript_wait_ms)
            if not speech_text and (speech_duration_ms <= 0 or speech_duration_ms < wait_ms):
                # No words yet. Any detector answering here would judge an
                # empty utterance (RFC §12.3.13 timing constraint), so wait for
                # a transcript, or for long enough to judge on duration alone.
                return InterruptionDecision(
                    should_interrupt=False,
                    pending_confirmation=True,
                    confirm_after_ms=max(wait_ms - speech_duration_ms, 1),
                    awaiting_transcript=True,
                    reason="nothing to classify yet",
                )

            from roomkit.voice.pipeline.backchannel.base import BackchannelContext

            ctx = BackchannelContext(
                transcript=speech_text,
                speech_duration_ms=float(speech_duration_ms),
                is_final=speech_final,
            )
            bc_result = self._backchannel_detector.classify(ctx)
            if bc_result.pending and not speech_final:
                # The words may still become an acknowledgement: the next
                # ones decide, never the speech's duration. A final has no
                # next words: it is never pending, and is judged as it is.
                return InterruptionDecision(
                    should_interrupt=False,
                    pending_confirmation=True,
                    confirm_after_ms=_PENDING_LOOK_AGAIN_MS,
                    awaiting_transcript=True,
                    reason="words still being said",
                )
            if bc_result.is_backchannel:
                return InterruptionDecision(
                    should_interrupt=False,
                    is_backchannel=True,
                    reason="backchannel detected",
                )
            return InterruptionDecision(
                should_interrupt=True,
                reason="not a backchannel",
            )

        return InterruptionDecision(
            should_interrupt=False,
            reason=f"unknown strategy: {strategy}",
        )


_WORD = re.compile(r"[^\W_]+")


def _words(text: str) -> list[str]:
    """Lowercase words without accents: the STT spells the response its own way."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    return _WORD.findall("".join(c for c in folded if not unicodedata.combining(c)))


def echoes_nearly(speech_text: str, response_text: str) -> bool:
    """Whether *speech_text* is words of *response_text* in their order, one of
    them misheard at most, and three at least heard right.

    How an echo the canceller let through comes back from the STT: "c'est bien
    la lampe Salon" heard as "Eh bien, la lampe salon".
    """
    heard, said = _words(speech_text), _words(response_text)
    span = len(heard)
    if span < 3:
        return False
    for start in range(len(said) - span + 1):
        slips = sum(h != s for h, s in zip(heard, said[start : start + span], strict=True))
        if slips <= 1 and span - slips >= 3:
            return True
    return False


def echoes(speech_text: str, response_text: str, *, cut_short: bool = True) -> bool:
    """Whether *speech_text* is only words of *response_text*, consecutive and in order.

    With *cut_short*, its last word may be the start of the response's, as a
    partial transcript catches a word being said ("Salut ! Ça v" echoes "Salut !
    Ça va très bien"), but only after a whole word: a lone "No" is no echo of
    "not", nor "Stop" of "stopped". A final transcript is read without it
    (``cut_short=False``). One word the response does not say there makes it
    speech.
    """
    heard, said = _words(speech_text), _words(response_text)
    if not heard:
        return False
    *whole, last = heard
    span = len(heard)
    partial_last = cut_short and bool(whole)
    return any(
        said[start : start + span - 1] == whole
        and (
            said[start + span - 1].startswith(last)
            if partial_last
            else said[start + span - 1] == last
        )
        for start in range(len(said) - span + 1)
    )
