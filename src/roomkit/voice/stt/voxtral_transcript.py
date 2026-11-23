"""The text of a Voxtral realtime session, cut into RoomKit results.

A session's events become partials, speech starts and finals here, kept apart
from :mod:`roomkit.voice.stt.voxtral_realtime`, which reads them off the
socket and closes a turn when :meth:`deadline` says. Two shapes:
:class:`Transcript` for a plain session (with or without turn detection), and
:class:`SpeakerTurns` for a diarized one, which has no turn events.
"""

from __future__ import annotations

from typing import Any

from roomkit.voice.base import SpeakerSegment, TranscriptionResult, speaker_label
from roomkit.voice.stt.voxtral_protocol import UNATTRIBUTED_LABELS, event_error

_DELTA = "transcription.text.delta"
_SEGMENT = "transcription.segment"
_DONE = "transcription.done"

_SPEECH_START = TranscriptionResult(text="", is_final=False, is_speech_start=True)


class Transcript:
    """The text of one realtime session, as RoomKit results.

    Each ``transcription.text.delta`` adds text after the text already sent,
    final as it arrives and carrying its own spaces, so a partial is the text
    so far. Without turn detection the session is one utterance, and
    ``transcription.done`` gives its text of record.

    With turn detection the text is cut into turns. ``turn.start`` opens one
    (a speech start) and ``turn.end`` sets :attr:`closing`; the caller then
    calls :meth:`close` once the turn's last words have had time to arrive.
    The turn events come at once while the text runs a streaming delay
    behind the audio, so a turn's last words can follow its ``turn.end`` by
    nearly that delay, and even the next ``turn.start`` (measured 2026-10-21:
    "Italy?" arrived after the next turn had started). The text that comes
    before :meth:`close` is the ending turn's; a ``turn.start`` meanwhile is
    a speech start, its turn opening at the close, and the close comes a
    little sooner, ``next_turn_wait_s`` after the end: that turn's own words
    come at least a delay after the end. Text that arrives after its turn's
    final belongs to a turn already given: it is counted in
    :attr:`late_chars`, never sent as a turn of its own.

    Args:
        turns: Whether the session detects turns.
        final_wait_s: How long a turn waits after ``turn.end`` for its last words.
        next_turn_wait_s: How long it waits after ``turn.end`` once the next
            turn has started.
    """

    def __init__(
        self, *, turns: bool, final_wait_s: float = 0.0, next_turn_wait_s: float = 0.0
    ) -> None:
        self._turns = turns
        self._final_wait_s = final_wait_s
        self._next_turn_wait_s = next_turn_wait_s
        self._text = ""
        self._given = False
        self._next_open = False
        self._next_ended = False
        self._ended_at = 0.0
        self._late_unreported = 0
        self.closing = False
        self.late_chars = 0

    def feed(self, event: dict[str, Any]) -> list[TranscriptionResult]:
        """The results one server event gives, in order.

        Raises:
            VoxtralSTTError: The event reports an error.
        """
        kind = event.get("type")
        if kind == "error":
            raise event_error(event)
        if kind == _DELTA:
            return self._delta(str(event.get("text") or ""))
        if not self._turns:
            if kind == _DONE:
                return _final(str(event.get("text") or self._text))
            return []
        if kind == _DONE:
            return self.close()
        return self._turn_event(kind)

    @property
    def next_turn_started(self) -> bool:
        """Whether a turn has started while the last one is closing."""
        return self._next_open

    def deadline(self, now: float, current: float | None) -> float | None:
        """When the ending turn closes, given the deadline set so far; ``None`` if none ends."""
        if not self.closing:
            return None
        if current is None:
            # The first call of a closing turn is the one at its turn.end.
            self._ended_at = now
            current = now + self._final_wait_s
        if self._next_open:
            return min(current, self._ended_at + self._next_turn_wait_s)
        return current

    def take_late(self) -> int:
        """Characters that came after their turn's final since the last call."""
        late, self._late_unreported = self._late_unreported, 0
        return late

    def close(self) -> list[TranscriptionResult]:
        """Give the ending turn as a final, ``[]`` when it holds no words.

        A turn that started while this one was closing is open from here, and
        :attr:`closing` again if it has ended too.
        """
        text, self._text = self._text, ""
        self._given = not self._next_open
        self.closing = self._next_open and self._next_ended
        self._next_open = self._next_ended = False
        return _final(text)

    def _turn_event(self, kind: Any) -> list[TranscriptionResult]:
        if kind == "turn.start":
            if self.closing:
                # The latest turn to start is the one open after the close.
                self._next_open, self._next_ended = True, False
            else:
                self._given = False
            return [_SPEECH_START]
        if kind == "turn.end":
            if self._next_open:
                self._next_ended = True
            else:
                self.closing = True
        return []

    def _delta(self, text: str) -> list[TranscriptionResult]:
        if self._turns and self._given:
            self.late_chars += len(text.strip())
            self._late_unreported += len(text.strip())
            return []
        self._text += text
        so_far = self._text.strip()
        return [TranscriptionResult(text=so_far, is_final=False)] if so_far else []


class SpeakerTurns:
    """The text of a diarized session, one turn per speaker's run of words.

    With ``diarize`` the session sends ``transcription.segment`` events, a few
    words each with their speaker, and no turn events: diarization and turn
    detection cannot share a session. A turn therefore ends on a change of
    speaker, which closes the last speaker's turn at once, or on a pause:
    :meth:`deadline` lets the caller close it once ``pause_s`` passes with no
    new words. Within one person's speech the words came up to 2.03 s apart
    and at a change of speaker 1.5 s apart (measured 2026-10-21), so a pause
    can end a person's turn in two. Every final names its speaker as its one
    segment (RFC §12.2.3).

    Args:
        pause_s: How long without new words ends the turn under way.
    """

    late_chars = 0

    def __init__(self, *, pause_s: float) -> None:
        self._pause_s = pause_s
        self._speaker: str | None = None
        self._text = ""
        self._words_came = False

    def feed(self, event: dict[str, Any]) -> list[TranscriptionResult]:
        """The results one server event gives, in order.

        Raises:
            VoxtralSTTError: The event reports an error.
        """
        kind = event.get("type")
        self._words_came = False
        if kind == "error":
            raise event_error(event)
        if kind == _DONE:
            return self.close()
        if kind != _SEGMENT:
            return []
        text = str(event.get("text") or "")
        if not text.strip():
            return []
        speaker = speaker_label(event.get("speaker_id"), unattributed=UNATTRIBUTED_LABELS)
        results = self.close() if self._text and speaker != self._speaker else []
        if not self._text:
            self._speaker = speaker
            results.append(_SPEECH_START)
        self._text += text
        self._words_came = True
        results.append(TranscriptionResult(text=self._text.strip(), is_final=False))
        return results

    def deadline(self, now: float, current: float | None) -> float | None:
        """A pause from the last words ends the turn; ``None`` when none is under way.

        Only words push the pause back: an event without any keeps it.
        """
        if not self._text:
            return None
        return now + self._pause_s if self._words_came or current is None else current

    def take_late(self) -> int:
        """Nothing comes late here: a turn ends on the words that follow it."""
        return 0

    def close(self) -> list[TranscriptionResult]:
        """Give the turn under way as a final naming its speaker, ``[]`` when there is none."""
        text, self._text = self._text.strip(), ""
        if not text:
            return []
        return [TranscriptionResult(text=text, segments=[SpeakerSegment(self._speaker, text)])]


def _final(text: str) -> list[TranscriptionResult]:
    text = text.strip()
    return [TranscriptionResult(text=text, is_final=True)] if text else []
