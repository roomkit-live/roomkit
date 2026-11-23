"""Transcript models for the Gemini STT provider.

What :meth:`~roomkit.voice.stt.gemini.GeminiSTTProvider.transcribe_recording`
hands back: a whole recording as speaker turns, with the timestamps as the
model read them. Kept beside the provider so ``gemini.py`` holds the
transcription alone; ``roomkit.voice.stt`` exports them as ``GeminiTranscript``,
``GeminiTranscriptWord`` and ``GeminiTranscriptSegment``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from roomkit.voice.base import SpeakerSegment, speaker_label

# How this vendor's wire may spell a speaker it could not attribute: the
# provider names them, RoomKit's label rule knows no vendor's (RFC §12.2.3).
_UNATTRIBUTED = ("unknown", "UU", "PENDING")


@dataclass(frozen=True)
class TranscriptSegment:
    """One speaker turn.

    Timestamps are ``MM:SS`` strings, as the model returns them. They are the
    model's reading of the recording, not a forced alignment: treat them as
    navigation, not as sync marks. They are empty when the model returned no
    timing at all (the dedicated recogniser with ``word_timestamps=False``).
    """

    speaker: str
    start: str
    end: str
    text: str


@dataclass(frozen=True)
class TranscriptWord:
    """One recognised word with its timing, from the dedicated recogniser.

    Offsets are seconds from the start of the recording, at the 100 ms
    resolution the service reports.
    """

    text: str
    start: float
    end: float
    speaker: str | None = None
    """Label of the turn the word belongs to, or ``None`` without diarization."""


@dataclass(frozen=True)
class Transcript:
    """A whole recording, as speaker turns."""

    language: str
    """BCP-47 code of the language spoken. Empty when the model reported none:
    the dedicated recogniser only knows the language it was told."""
    segments: list[TranscriptSegment]
    words: list[TranscriptWord] = field(default_factory=list)
    """Word-level timing, when the model returned it (the dedicated recogniser
    with ``word_timestamps=True``); empty otherwise."""

    @property
    def text(self) -> str:
        """The turns joined into a readable transcript, one line per speaker."""
        return "\n".join(f"{s.speaker}: {s.text}" for s in self.segments)

    @property
    def plain_text(self) -> str:
        """The spoken words alone, without speaker labels."""
        return " ".join(s.text for s in self.segments)

    def speaker_segments(self) -> list[SpeakerSegment]:
        """The turns as RoomKit's shared :class:`SpeakerSegment` (RFC §12.2.3).

        Labels lose their "Speaker" word (``"Speaker 1"`` → ``"1"``). With
        word timing (the recogniser), a turn is a run of one speaker's words
        timed to 100 ms; otherwise the turns keep the model's ``MM:SS``
        reading, to the second, or no offsets when it gave none.
        """
        if self.words:
            return _segments_from_words(self.words)
        return [
            SpeakerSegment(
                speaker_label(s.speaker, unattributed=_UNATTRIBUTED),
                s.text,
                _ms(s.start),
                _ms(s.end),
            )
            for s in self.segments
            if s.text
        ]


def _segments_from_words(words: list[TranscriptWord]) -> list[SpeakerSegment]:
    runs: list[list[TranscriptWord]] = []
    for word in words:
        if runs and runs[-1][-1].speaker == word.speaker:
            runs[-1].append(word)
        else:
            runs.append([word])
    return [
        SpeakerSegment(
            speaker_label(run[0].speaker, unattributed=_UNATTRIBUTED),
            " ".join(w.text for w in run),
            round(run[0].start * 1000),
            round(run[-1].end * 1000),
        )
        for run in runs
    ]


def _ms(mmss: str) -> int | None:
    """``"01:05"`` (or ``"1:02:05"``) as milliseconds, ``None`` when unreadable."""
    if not mmss.strip():
        return None
    try:
        seconds = 0.0
        for part in mmss.strip().split(":"):
            seconds = seconds * 60 + float(part)
    except ValueError:
        return None
    return round(seconds * 1000)
