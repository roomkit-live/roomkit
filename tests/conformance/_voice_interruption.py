"""Scripted interruption for the voice scenarios (RFC §12.3.13, §12.6).

``voice.open``'s ``interruption`` sets the channel's InterruptionConfig, and
``backchannels`` or ``backchannel_decisions`` script its backchannel detector,
whose every context ``voice.read`` returns as ``backchannel_contexts``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from roomkit.voice import (
    BackchannelContext,
    BackchannelDecision,
    InterruptionConfig,
    InterruptionStrategy,
    MockBackchannelDetector,
)

_MEMBERS = (
    "min_speech_ms",
    "allow_during_first_ms",
    "transcript_wait_ms",
    "echo_guard_ms",
    "flush_partial_tts",
    "keep_partial_transcript",
)


def _decision(spec: Mapping[str, Any]) -> BackchannelDecision:
    return BackchannelDecision(
        is_backchannel=spec["is_backchannel"], pending=spec.get("pending", False)
    )


class _WordsDetector(MockBackchannelDetector):
    """The mock detector, answering the words a decision names however often
    they are judged; every other context takes the next decision in order."""

    def __init__(self, decisions: list[Mapping[str, Any]]) -> None:
        super().__init__([_decision(d) for d in decisions if "transcript" not in d])
        self._by_words = {d["transcript"]: _decision(d) for d in decisions if "transcript" in d}

    def classify(self, context: BackchannelContext) -> BackchannelDecision:
        decision = self._by_words.get(context.transcript or "")
        if decision is None:
            return super().classify(context)
        self.evaluations.append(context)
        return decision


def _detector(spec: Mapping[str, Any]) -> MockBackchannelDetector | None:
    if "backchannel_decisions" in spec:
        return _WordsDetector(spec["backchannel_decisions"])
    if "backchannels" in spec:
        decisions = [BackchannelDecision(is_backchannel=v) for v in spec["backchannels"]]
        return MockBackchannelDetector(decisions)
    return None


def interruption(
    spec: Mapping[str, Any],
) -> tuple[InterruptionConfig, MockBackchannelDetector | None]:
    """The channel's InterruptionConfig and its scripted backchannel detector.

    Without ``interruption.strategy`` the strategy is IMMEDIATE, or SEMANTIC
    with a scripted detector; every other member absent keeps
    InterruptionConfig's default, ``allow_during_first_ms`` 0 included.
    """
    detector = _detector(spec)
    given = spec.get("interruption", {})
    default = InterruptionStrategy.SEMANTIC if detector else InterruptionStrategy.IMMEDIATE
    config = InterruptionConfig(
        strategy=InterruptionStrategy(given.get("strategy", default.value)),
        backchannel_detector=detector,
        **{name: given[name] for name in _MEMBERS if name in given},
    )
    return config, detector


def read_backchannels(detector: MockBackchannelDetector | None) -> list[dict[str, Any]]:
    """Each context the scripted detector was given, in order; the same words
    judged again in a row (a pending verdict's second look) read once."""
    read: list[dict[str, Any]] = []
    for context in detector.evaluations if detector is not None else []:
        record = {
            # No words, whether given as None or as "", read as none.
            "transcript": context.transcript or None,
            "is_final": context.is_final,
            "speech_duration_ms": context.speech_duration_ms,
        }
        if read and (read[-1]["transcript"], read[-1]["is_final"]) == (
            record["transcript"],
            record["is_final"],
        ):
            continue
        read.append(record)
    return read


INTERRUPTION_OPEN_ARGUMENTS = {"interruption", "backchannel_decisions", "max_sentences"}
