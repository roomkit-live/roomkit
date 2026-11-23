"""Speaker attribution from a diarizing STT, as a Voice Channel carries it (RFC §12.2.3).

A diarizing STT labels who said each segment of a final result. The labels
hold within one stream only, so the channel keeps one stream per session and
numbers each new one (the *epoch*); this module turns a label and its epoch
into what the room message carries, and decides when a label is a speaker
change. When the STT labels nothing, the pipeline's diarization stage can
name the utterance's speaker instead (:class:`PipelineSpeakerTally`).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from roomkit.voice.base import speaker_label

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger("roomkit.voice")

_CLAIM_WAIT_S = 1.0

_NOBODY = ""
"""The tally's key for a voice the stage heard but matched to no speaker (a
null or empty ``speaker_id``): it competes like a label and, if heard the
longest, the transcript is :data:`UNKNOWN_SPEAKER`'s."""

UNKNOWN_SPEAKER = "Unknown speaker"
"""The name of words the STT attributed to nobody. Without a name the AI
channel would fall back to the stream owner's, for words nobody said as them."""


def default_sender_name(label: str | None, epoch: int) -> str:
    """``"Speaker A"`` in the first stream, ``"Speaker A#1"`` after a reconnect.

    The epoch is in the name from the second stream on, so the same letter
    from two streams never reads as one person to the model.
    """
    if label is None:
        return UNKNOWN_SPEAKER
    return f"Speaker {label}" if epoch == 0 else f"Speaker {label}#{epoch}"


@dataclass(frozen=True)
class SpeakerAttribution:
    """Who said a routed transcript: the STT's label, its epoch, the room name.

    ``sender_name`` starts as :func:`default_sender_name` and is what an
    ``ON_TRANSCRIPTION`` hook left, a label matched to a known voice (``named``
    when the hook gave it); ``participant_id`` is the room participant the
    hook matched it to, if any.
    """

    label: str | None
    epoch: int
    sender_name: str
    source: Literal["stt", "pipeline"] = "stt"
    participant_id: str | None = None
    named: bool = False

    @classmethod
    def of(
        cls, label: str | None, epoch: int, source: Literal["stt", "pipeline"] = "stt"
    ) -> SpeakerAttribution:
        return cls(label, epoch, default_sender_name(label, epoch), source)

    def same_voice(self, other: SpeakerAttribution) -> bool:
        """Same label in the same stream matched to the same participant,
        whatever name a hook gave either: a turn has one speaker (RFC §12.2.3)."""
        mine = (self.label, self.epoch, self.participant_id)
        return mine == (other.label, other.epoch, other.participant_id)

    def metadata(self) -> dict[str, Any]:
        """The keys the inbound message carries; no label for unattributed
        words, and no name for words matched to a participant without one:
        the participant's record names them, not the stream's default name."""
        meta: dict[str, Any] = {"speaker_epoch": self.epoch}
        if self.named or self.participant_id is None:
            meta["sender_name"] = self.sender_name
        if self.label is not None:
            meta["speaker_label"] = self.label
        return meta


def routed_sender(stream_owner: str, speaker: SpeakerAttribution | None) -> str:
    """Who a routed transcript is from: the room participant an
    ``ON_TRANSCRIPTION`` hook matched its speaker to, else *stream_owner*,
    the session's participant (RFC §12.2.3)."""
    if speaker is not None and speaker.participant_id:
        return speaker.participant_id
    return stream_owner


class SpeakerTracker:
    """The speaker-change rule for one session, fed each routed segment in order.

    A segment with no label neither fires nor resets. A label fires when it
    differs from the last one routed in the same epoch, and on the first
    labelled segment of each epoch.
    """

    def __init__(self) -> None:
        self._epoch: int | None = None
        self._last: str | None = None
        self._seen: set[str] = set()

    def observe(self, label: str | None, epoch: int) -> bool | None:
        """``None`` when no change fires; otherwise whether the speaker is new."""
        if label is None:
            return None
        if epoch != self._epoch:
            self._epoch, self._last, self._seen = epoch, None, set()
        is_new = label not in self._seen
        self._seen.add(label)
        if label == self._last:
            return None
        self._last = label
        return is_new


class PipelineSpeakerTally:
    """Who the pipeline's diarization stage heard, by seconds of audio, per session.

    The stage marks each frame it identified
    (``frame.metadata["diarization"]["speaker_id"]``); every processed frame is
    added here, after the stage ran. A transcript's speaker is the one heard
    the longest since the count last closed (RFC §12.2.3, "From the pipeline
    stage"); a voice the stage matched to nobody counts too, as
    :data:`UNKNOWN_SPEAKER`. The stage's labels hold for the whole session, so
    the epoch is always 0.

    A result counts for the audio since the stage's previous one: a stage that
    labels every frame weighs each frame, and one that judges a buffer —
    sherpa-onnx, every 2 s of speech and on the rest at SPEECH_END — weighs
    each verdict by the speech it heard, so a short tail does not tie with a
    two-second stretch.

    Behind a VAD the count closes on the SPEECH_END frame itself — the frame a
    stage such as sherpa-onnx's identifies on — and the pipeline fires its
    SPEECH_END callbacks *before* the stage sees that frame. So a speech end
    claims its speaker (:meth:`claim`), answered once the closing frame is
    counted, and awaits the answer (:func:`claimed_speaker`). Without a VAD a
    transcript takes the count as its final lands (:meth:`take`).

    A pipeline without an event loop of its own calls back on whatever thread
    runs its frames, hence the lock and the thread-safe futures.
    """

    def __init__(self) -> None:
        self._seconds: dict[str, dict[str, float]] = {}
        self._unjudged: dict[str, float] = {}  # audio since the stage's last result
        self._claims: dict[str, Future[SpeakerAttribution | None]] = {}
        self._lock = threading.Lock()

    def add(self, session_id: str, frame: AudioFrame) -> None:
        verdict = _verdict(frame)
        with self._lock:
            unjudged = self._unjudged.pop(session_id, 0.0) + _seconds_of(frame)
            if verdict is None:
                self._unjudged[session_id] = unjudged
            else:
                heard = self._seconds.setdefault(session_id, {})
                heard[verdict] = heard.get(verdict, 0.0) + unjudged
            if not frame.metadata.get("vad_speech_end"):
                return
            speaker = self._close(session_id)
            claim = self._claims.pop(session_id, None)
        _answer(claim, speaker)

    def add_frame(self, session: Any, frame: AudioFrame) -> None:
        """The pipeline's processed-frame callback: ``(session, frame)``."""
        self.add(session.id, frame)

    def claim(self, session_id: str) -> Future[SpeakerAttribution | None]:
        """The speaker of the utterance ending now, answered by its closing frame."""
        claim: Future[SpeakerAttribution | None] = Future()
        with self._lock:
            stale = self._claims.pop(session_id, None)
            self._claims[session_id] = claim
        _answer(stale, None)
        return claim

    def take(self, session_id: str) -> SpeakerAttribution | None:
        """The speaker heard the longest since the last take, then a clean slate."""
        with self._lock:
            return self._close(session_id)

    def reset(self, session_id: str) -> None:
        """Forget the count; a claim still open is answered with nobody."""
        with self._lock:
            self._close(session_id)
            claim = self._claims.pop(session_id, None)
        _answer(claim, None)

    def _close(self, session_id: str) -> SpeakerAttribution | None:
        """The count's speaker, then a clean slate; the caller holds the lock."""
        self._unjudged.pop(session_id, None)
        return _heard_longest(self._seconds.pop(session_id, None))


async def claimed_speaker(
    claim: Future[SpeakerAttribution | None] | None,
) -> SpeakerAttribution | None:
    """The claimed speaker, once the frame closing its utterance is counted.

    That frame is in the very pipeline pass that fired SPEECH_END, so the wait
    is the rest of that pass; the bound only matters if the pass failed midway.
    """
    if claim is None:
        return None
    answer = asyncio.wrap_future(claim)
    done, _ = await asyncio.wait({answer}, timeout=_CLAIM_WAIT_S)
    if not done:
        logger.warning("No pipeline speaker: the utterance's closing frame was never counted")
        return None
    return answer.result()


def _verdict(frame: AudioFrame) -> str | None:
    """The stage's verdict on a frame: a label, :data:`_NOBODY`, or ``None`` for none."""
    identified = frame.metadata.get("diarization")
    if not isinstance(identified, dict):
        return None
    label = speaker_label(identified.get("speaker_id"))
    return _NOBODY if label is None else label


def _seconds_of(frame: AudioFrame) -> float:
    width = max(frame.sample_width * frame.channels, 1)
    return len(frame.data) / (width * frame.sample_rate)


def _heard_longest(heard: dict[str, float] | None) -> SpeakerAttribution | None:
    if not heard:
        return None
    label = max(heard, key=lambda name: heard[name])
    return SpeakerAttribution.of(label or None, 0, source="pipeline")


def _answer(
    claim: Future[SpeakerAttribution | None] | None, speaker: SpeakerAttribution | None
) -> None:
    if claim is None:
        return
    # A waiter gone (its task cancelled) may have cancelled the future.
    with contextlib.suppress(InvalidStateError):
        claim.set_result(speaker)
