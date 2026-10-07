"""The policy that always speaks: an AI channel's behaviour without one."""

from __future__ import annotations

from roomkit.speaking.base import SpeakDecision, SpeakPolicy, SpeakTurn


class AlwaysSpeak(SpeakPolicy):
    """Answers every event. Unlike a channel without a policy, it reports each
    decision to ``ON_SPEAK_DECISION``: a baseline to measure another policy against."""

    async def decide(self, turn: SpeakTurn) -> SpeakDecision:
        return SpeakDecision("speak", reason="always")
