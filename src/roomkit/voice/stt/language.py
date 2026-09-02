"""Lock a session's STT language to the one its speaker uses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from roomkit.voice.base import TranscriptionResult

logger = logging.getLogger("roomkit.voice.stt")


@dataclass
class _LockState:
    """What the lock knows about one session."""

    locked: str | None = None
    candidate: str | None = None
    agreements: int = 0
    misses: int = 0


@dataclass
class STTLanguageLock:
    """Start detecting, pin the language the speaker uses, let go when it stops fitting.

    A streaming STT does better with its language set than in a detecting
    mode (Deepgram Nova-3 ``multi``), but nobody knows the language before
    the caller speaks. This policy watches the final results a session
    produces and tells :class:`~roomkit.channels.VoiceChannel` which
    language its next stream should use:

    - Every session starts in ``detect_language``.
    - Once ``lock_after`` consecutive finals report the same language, the
      session is locked to it — mapped through ``prefer`` first, so a
      reported ``"fr"`` can become ``"fr-CA"``.
    - A locked session counts a miss for every final with no text, a
      confidence below ``min_confidence``, or a reported language other
      than the lock. ``release_after`` consecutive misses send it back to
      ``detect_language``; a fitting final resets the count.

    The object is shared by the channel and keeps one small state per
    session id; the channel calls :meth:`forget` when a session goes away.
    Everything it does is reachable from a hook with ``event.language`` and
    ``VoiceChannel.set_stt_language`` — it is the packaged version of that
    loop, not the only way to run it.
    """

    detect_language: str = "multi"
    prefer: dict[str, str] = field(default_factory=dict)
    lock_after: int = 1
    release_after: int = 2
    min_confidence: float = 0.5
    _sessions: dict[str, _LockState] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.detect_language.strip():
            raise ValueError("detect_language must not be blank")
        if self.lock_after < 1:
            raise ValueError("lock_after must be at least 1")
        if self.release_after < 1:
            raise ValueError("release_after must be at least 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

    def language_for(self, session_id: str) -> str:
        """The language the session should be using right now."""
        state = self._sessions.get(session_id)
        locked = state.locked if state is not None else None
        return locked or self.detect_language

    def observe(self, session_id: str, result: TranscriptionResult) -> str:
        """Account for a final result and return the language for the next stream."""
        state = self._sessions.setdefault(session_id, _LockState())
        if state.locked is None:
            self._observe_detecting(session_id, state, result)
        else:
            self._observe_locked(session_id, state, state.locked, result)
        return state.locked or self.detect_language

    def forget(self, session_id: str) -> None:
        """Drop the state kept for a session."""
        self._sessions.pop(session_id, None)

    def _observe_detecting(
        self, session_id: str, state: _LockState, result: TranscriptionResult
    ) -> None:
        reported = result.language
        if not reported or not result.text.strip():
            return
        target = self.prefer.get(reported, reported)
        if state.candidate == target:
            state.agreements += 1
        else:
            state.candidate, state.agreements = target, 1
        if state.agreements < self.lock_after:
            return
        state.locked = target
        state.candidate, state.agreements, state.misses = None, 0, 0
        logger.info(
            "STT language locked to %s for session %s (reported %s)",
            target,
            session_id,
            reported,
        )

    def _observe_locked(
        self, session_id: str, state: _LockState, locked: str, result: TranscriptionResult
    ) -> None:
        reported = result.language
        fits = bool(result.text.strip())
        if fits and result.confidence is not None and result.confidence < self.min_confidence:
            fits = False
        if fits and reported and self.prefer.get(reported, reported) != locked:
            fits = False
        if fits:
            state.misses = 0
            return
        state.misses += 1
        if state.misses < self.release_after:
            return
        logger.info(
            "STT language %s released for session %s after %d misses",
            locked,
            session_id,
            state.misses,
        )
        state.locked = None
        state.misses = 0
