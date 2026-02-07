"""Interruption (barge-in) strategy configuration and handler."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.pipeline.backchannel_detector import BackchannelDetector

logger = logging.getLogger("roomkit.voice.interruption")


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

    strategy: InterruptionStrategy = InterruptionStrategy.IMMEDIATE
    """Which strategy to use."""

    min_speech_ms: int = 200
    """Minimum speech duration (ms) before triggering (used by CONFIRMED)."""

    allow_during_first_ms: int = 0
    """If > 0, only allow interruptions after this many ms of playback."""


@dataclass
class InterruptionDecision:
    """Result of evaluating whether an interruption should proceed."""

    should_interrupt: bool
    """Whether to interrupt playback."""

    is_backchannel: bool = False
    """True if the utterance was classified as a backchannel."""

    reason: str = ""
    """Human-readable reason for the decision."""


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
        self._backchannel_detector = backchannel_detector

    @property
    def config(self) -> InterruptionConfig:
        return self._config

    def evaluate(
        self,
        *,
        playback_position_ms: int,
        speech_duration_ms: int = 0,
        speech_text: str = "",
    ) -> InterruptionDecision:
        """Evaluate whether an interruption should proceed.

        Args:
            playback_position_ms: How far into TTS playback (ms).
            speech_duration_ms: How long the user has been speaking (ms).
            speech_text: Transcribed text of the speech (for SEMANTIC).

        Returns:
            An InterruptionDecision.
        """
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
                reason="speech too short",
            )

        if strategy == InterruptionStrategy.SEMANTIC:
            if self._backchannel_detector is None:
                # No detector configured — fall back to confirmed behaviour
                logger.warning(
                    "SEMANTIC strategy but no backchannel_detector; "
                    "falling back to CONFIRMED"
                )
                if speech_duration_ms >= self._config.min_speech_ms:
                    return InterruptionDecision(
                        should_interrupt=True,
                        reason="semantic fallback (no detector)",
                    )
                return InterruptionDecision(
                    should_interrupt=False,
                    reason="speech too short (semantic fallback)",
                )

            from roomkit.voice.pipeline.backchannel_detector import BackchannelContext

            ctx = BackchannelContext(
                text=speech_text,
                duration_ms=float(speech_duration_ms),
            )
            bc_result = self._backchannel_detector.evaluate(ctx)
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
