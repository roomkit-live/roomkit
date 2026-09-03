"""Transcript models for the Gemini STT provider.

What :meth:`~roomkit.voice.stt.gemini.GeminiSTTProvider.transcribe_recording`
hands back: a whole recording as speaker turns, with the timestamps as the
model read them. Kept beside the provider so ``gemini.py`` holds the
transcription alone; the public import path stays ``roomkit.voice.stt.gemini``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptSegment:
    """One speaker turn.

    Timestamps are ``MM:SS`` strings, as the model returns them. They are the
    model's reading of the recording, not a forced alignment: treat them as
    navigation, not as sync marks.
    """

    speaker: str
    start: str
    end: str
    text: str


@dataclass(frozen=True)
class Transcript:
    """A whole recording, as speaker turns."""

    language: str
    segments: list[TranscriptSegment]

    @property
    def text(self) -> str:
        """The turns joined into a readable transcript, one line per speaker."""
        return "\n".join(f"{s.speaker}: {s.text}" for s in self.segments)

    @property
    def plain_text(self) -> str:
        """The spoken words alone, without speaker labels."""
        return " ".join(s.text for s in self.segments)
