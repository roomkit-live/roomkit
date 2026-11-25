"""SEMANTIC in VAD mode classifies words, not an empty speech onset (RFC §12.3.13).

The pipeline VAD's SPEECH_START carries no transcript and no duration, and a
keyword detector cannot find "uh-huh" in ``""``. SEMANTIC therefore holds the
speech: with a streaming STT the held segment is transcribed and each partial
is classified; without one, the second look at ``min_speech_ms`` judges on
duration alone (the CONFIRMED fallback).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult, VoiceSession
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.backchannel import PhraseBackchannelDetector
from roomkit.voice.pipeline.backchannel.base import (
    BackchannelContext,
    BackchannelDecision,
    BackchannelDetector,
)
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.pipeline.vad.mock import MockVADProvider
from roomkit.voice.stt.base import STTProvider
from tests.channel_tasks import settled

_ONSET = VADEvent(type=VADEventType.SPEECH_START, audio_bytes=b"\x01\x00" * 320)


class _KeywordDetector(BackchannelDetector):
    """The guide's detector: a short acknowledgement is a backchannel."""

    _WORDS = {"uh-huh", "mm-hmm", "ok", "yeah"}

    def __init__(self) -> None:
        self.seen: list[str | None] = []

    @property
    def name(self) -> str:
        return "keywords"

    def classify(self, context: BackchannelContext) -> BackchannelDecision:
        self.seen.append(context.transcript)
        text = (context.transcript or "").strip().lower()
        return BackchannelDecision(is_backchannel=text in self._WORDS)


class _AlwaysPending(BackchannelDetector):
    """A detector that never decides, whatever the words, a final's included."""

    @property
    def name(self) -> str:
        return "always-pending"

    def classify(self, context: BackchannelContext) -> BackchannelDecision:
        return BackchannelDecision(is_backchannel=False, pending=True)


class _PartialsSTT(STTProvider):
    """Streaming STT that speaks its partials once audio arrives."""

    def __init__(self, partials: list[str]) -> None:
        self._partials = partials
        self.said = asyncio.Event()

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text=" ".join(self._partials), is_final=True)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            break
        try:
            for partial in self._partials:
                yield TranscriptionResult(text=partial, is_final=False)
        finally:
            self.said.set()  # every partial read by the channel
        async for _ in audio_stream:
            pass
        yield TranscriptionResult(text=self._partials[-1], is_final=True)


class _FinalOnlySTT(STTProvider):
    """Streaming STT that releases its words only once the audio ends, as a
    transducer often does for a short word ("okay", "stop"); with *is_final*
    false, as a partial and no final after it."""

    def __init__(self, final: str, *, is_final: bool = True) -> None:
        self._final = final
        self._is_final = is_final

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text=self._final, is_final=True)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            pass
        yield TranscriptionResult(text=self._final, is_final=self._is_final)


class _BatchSTT(STTProvider):
    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="hello", is_final=True)


async def _room(
    stt: STTProvider,
    *,
    min_speech_ms: int = 300,
    detector: BackchannelDetector | None = None,
    transcript_wait_ms: int = 1000,
) -> tuple[RoomKit, VoiceChannel, VoiceSession, Any, dict[str, list[Any]]]:
    detector = detector or _KeywordDetector()
    backend = MockVoiceBackend()
    channel = VoiceChannel(
        "voice-1",
        stt=stt,
        backend=backend,
        pipeline=AudioPipelineConfig(vad=MockVADProvider()),
        interruption=InterruptionConfig(
            strategy=InterruptionStrategy.SEMANTIC,
            backchannel_detector=detector,
            min_speech_ms=min_speech_ms,
            transcript_wait_ms=transcript_wait_ms,
        ),
    )
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice-1")
    session = await kit.connect_voice("r1", "user-1", "voice-1")
    channel.bind_session(
        session,
        "r1",
        ChannelBinding(room_id="r1", channel_id="voice-1", channel_type=ChannelType.VOICE),
    )
    assert not channel._continuous_stt  # noqa: SLF001

    seen: dict[str, list[Any]] = {"barge_in": [], "backchannel": [], "transcription": []}

    @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
    async def on_barge_in(event: Any, context: Any) -> None:
        seen["barge_in"].append(event)

    @kit.hook(HookTrigger.ON_BACKCHANNEL, HookExecution.ASYNC)
    async def on_backchannel(event: Any, context: Any) -> None:
        seen["backchannel"].append(event)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, HookExecution.ASYNC)
    async def on_transcription(event: Any, context: Any) -> None:
        seen["transcription"].append(event)

    channel._playing_sessions[session.id] = TTSPlaybackState(  # noqa: SLF001
        session_id=session.id,
        text="Your appointment is confirmed for Tuesday at ten.",
        started_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    return kit, channel, session, detector, seen


async def _until(predicate: Any, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


async def _settle(channel: VoiceChannel, *, ended: bool = False) -> None:
    """Until the work the channel scheduled on what it heard is done. The STT
    stream of a held segment runs until the speech *ended*, then finishes; the
    held segment's confirm timer outlives its verdict up to transcript_wait_ms
    and is left out (a test needing its second look waits for what it does)."""
    await settled(
        channel,
        lasting=("_confirm_barge_in",) if ended else ("_confirm_barge_in", "consume"),
        timeout=3,
    )


# A min_speech_ms on the test's scale, and a pause past it: the second look a
# held segment gets there would have cut by then.
_MIN_SPEECH_MS = 50
_PAST_MIN_SPEECH = 0.1


async def test_speech_onset_alone_never_consults_the_detector() -> None:
    kit, channel, session, detector, seen = await _room(_BatchSTT())

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.sleep(0.05)

    assert detector.seen == []
    assert seen["barge_in"] == []
    assert session.id in channel._playing_sessions  # noqa: SLF001
    await kit.close()


async def test_a_backchannel_lets_the_bot_talk_and_its_segment_is_discarded() -> None:
    stt = _PartialsSTT(["uh-huh"])
    kit, channel, session, detector, seen = await _room(stt, min_speech_ms=_MIN_SPEECH_MS)

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.wait_for(stt.said.wait(), 2)
    await _settle(channel)
    channel._on_pipeline_speech_end(session, b"\x11\x22" * 160)  # noqa: SLF001
    await _settle(channel, ended=True)
    await asyncio.sleep(_PAST_MIN_SPEECH)  # past min_speech_ms: the timer must not cut it

    assert session.id in channel._playing_sessions  # noqa: SLF001
    await kit.close()
    assert detector.seen == ["uh-huh"]
    assert len(seen["backchannel"]) == 1
    assert seen["barge_in"] == []
    assert seen["transcription"] == []


async def test_an_acknowledged_utterance_running_long_is_not_cut() -> None:
    """Once "uh-huh" is recognized, speech still going at min_speech_ms is not
    judged again on duration alone."""
    stt = _PartialsSTT(["uh-huh"])
    kit, channel, session, detector, seen = await _room(stt, min_speech_ms=_MIN_SPEECH_MS)

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.wait_for(stt.said.wait(), 2)
    await _settle(channel)
    await asyncio.sleep(_PAST_MIN_SPEECH)  # still speaking at min_speech_ms
    await kit.close()

    assert seen["barge_in"] == []
    assert len(seen["backchannel"]) == 1


async def test_a_real_interruption_cuts_in_and_becomes_the_turn() -> None:
    stt = _PartialsSTT(["attends", "attends, stop"])
    kit, channel, session, detector, seen = await _room(stt)

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.wait_for(stt.said.wait(), 2)
    await _settle(channel)

    assert session.id not in channel._playing_sessions  # noqa: SLF001
    assert detector.seen == ["attends"]  # the first real words decide

    channel._on_pipeline_speech_end(session, b"\x11\x22" * 160)  # noqa: SLF001
    await _settle(channel, ended=True)
    await kit.close()

    assert len(seen["barge_in"]) == 1
    # The held stream kept the utterance from its first word.
    assert [e.text for e in seen["transcription"]] == ["attends, stop"]


async def test_without_streaming_stt_semantic_falls_back_to_confirmed() -> None:
    kit, channel, session, detector, seen = await _room(_BatchSTT(), min_speech_ms=50)

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.sleep(0.02)
    assert seen["barge_in"] == []

    await _until(lambda: seen["barge_in"])  # sustained past min_speech_ms: the second look
    await kit.close()

    assert detector.seen == [""]  # judged on duration alone, once
    assert len(seen["barge_in"]) == 1


async def test_without_streaming_stt_a_short_blip_does_not_cut() -> None:
    kit, channel, session, detector, seen = await _room(_BatchSTT(), min_speech_ms=100)

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.sleep(0.02)
    channel._on_pipeline_speech_end(session, b"\x11\x22" * 160)  # noqa: SLF001
    await _settle(channel, ended=True)
    await asyncio.sleep(_PAST_MIN_SPEECH)  # past the blip's min_speech_ms: no second look cuts
    await kit.close()

    assert detector.seen == []
    assert seen["barge_in"] == []
    assert seen["transcription"] == []


async def test_a_held_segment_reaches_no_partial_hook() -> None:
    stt = _PartialsSTT(["uh-huh"])
    kit, channel, session, detector, seen = await _room(stt)
    partials: list[Any] = []

    @kit.hook(HookTrigger.ON_PARTIAL_TRANSCRIPTION, HookExecution.ASYNC)
    async def on_partial(event: Any, context: Any) -> None:
        partials.append(event)

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.wait_for(stt.said.wait(), 2)
    await _settle(channel)
    await kit.close()

    assert detector.seen == ["uh-huh"]
    assert partials == []


async def test_a_speech_end_right_after_the_cut_in_keeps_the_turn() -> None:
    """The cut-in decision lets the segment through at once: a speech end
    landing before the scheduled barge-in runs is the user's turn, not echo."""
    kit, channel, session, detector, seen = await _room(_PartialsSTT(["attends, stop"]))
    processed: list[bytes] = []

    async def capture(sess: Any, audio: bytes, room_id: str, stream_state: Any, **kw: Any) -> None:
        processed.append(audio)

    channel._on_pipeline_vad_event(session, VADEvent(type=VADEventType.SPEECH_START))  # noqa: SLF001
    channel._process_speech_end = capture  # type: ignore[method-assign]  # noqa: SLF001
    # The partial decides, and the speech ends before any task has run.
    channel._on_held_transcript(session, "r1", "attends, stop")  # noqa: SLF001
    channel._on_pipeline_speech_end(session, b"\x11\x22" * 160)  # noqa: SLF001
    await _until(lambda: processed)
    await _settle(channel)
    await kit.close()

    assert processed == [b"\x11\x22" * 160]
    assert len(seen["barge_in"]) == 1


async def test_phrase_detector_lets_the_bot_talk_through_an_okay() -> None:
    stt = _PartialsSTT(["Okay."])
    kit, channel, session, _, seen = await _room(
        stt, detector=PhraseBackchannelDetector(), min_speech_ms=_MIN_SPEECH_MS
    )

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.wait_for(stt.said.wait(), 2)
    await _settle(channel)
    channel._on_pipeline_speech_end(session, b"\x11\x22" * 160)  # noqa: SLF001
    await _settle(channel, ended=True)
    await asyncio.sleep(_PAST_MIN_SPEECH)

    assert session.id in channel._playing_sessions  # noqa: SLF001
    await kit.close()
    assert len(seen["backchannel"]) == 1
    assert seen["barge_in"] == []


async def test_phrase_detector_cuts_in_when_the_okay_becomes_a_question() -> None:
    stt = _PartialsSTT(["Okay,", "Okay, and do you", "Okay, and do you remember"])
    kit, channel, session, _, seen = await _room(stt, detector=PhraseBackchannelDetector())

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.wait_for(stt.said.wait(), 2)
    await _settle(channel)
    channel._on_pipeline_speech_end(session, b"\x11\x22" * 160)  # noqa: SLF001
    await _settle(channel, ended=True)

    assert session.id not in channel._playing_sessions  # noqa: SLF001
    await kit.close()
    assert len(seen["barge_in"]) == 1
    assert [e.text for e in seen["transcription"]] == ["Okay, and do you remember"]


async def test_phrase_detector_talks_through_an_acknowledgement_said_word_by_word() -> None:
    """ "Je" alone may be "Je vois": the held segment waits for its next words (RMK-889)."""
    stt = _PartialsSTT(["Je", "Je vois"])
    kit, channel, session, _, seen = await _room(
        stt, detector=PhraseBackchannelDetector(), min_speech_ms=_MIN_SPEECH_MS
    )

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.wait_for(stt.said.wait(), 2)
    await _settle(channel)
    channel._on_pipeline_speech_end(session, b"\x11\x22" * 160)  # noqa: SLF001
    await _settle(channel, ended=True)
    await asyncio.sleep(_PAST_MIN_SPEECH)

    assert session.id in channel._playing_sessions  # noqa: SLF001
    await kit.close()
    assert seen["barge_in"] == []
    assert len(seen["backchannel"]) == 1


async def test_phrase_detector_cuts_in_on_a_final_left_pending() -> None:
    """ "No" might have been "noted": its final, "No", is judged whole and cuts."""
    stt = _PartialsSTT(["No"])
    kit, channel, session, _, seen = await _room(stt, detector=PhraseBackchannelDetector())

    channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
    await asyncio.wait_for(stt.said.wait(), 2)
    await _settle(channel)
    channel._on_pipeline_speech_end(session, b"\x11\x22" * 160)  # noqa: SLF001
    await _settle(channel, ended=True)
    await _until(lambda: seen["transcription"])
    await kit.close()

    assert len(seen["barge_in"]) == 1
    assert [e.text for e in seen["transcription"]] == ["No"]


class TestHeldSpeechEndingBeforeItsWords:
    """A held segment that ends before its first word is judged on its final
    transcript, not discarded unheard (RFC §12.3.13)."""

    async def _speak_and_stop(
        self,
        final: str,
        *,
        bot_still_talking: bool = True,
        is_final: bool = True,
        detector: BackchannelDetector | None = None,
    ) -> tuple:
        kit, channel, session, _, seen = await _room(
            _FinalOnlySTT(final, is_final=is_final),
            detector=detector or PhraseBackchannelDetector(),
            transcript_wait_ms=2000,
        )
        channel._on_pipeline_vad_event(session, _ONSET)  # noqa: SLF001
        await _until(lambda: session.id in channel._stt_streams)  # noqa: SLF001 — held, transcribed
        if not bot_still_talking:
            channel._playing_sessions.pop(session.id)  # noqa: SLF001
        channel._on_pipeline_speech_end(session, b"\x11\x22" * 160)  # noqa: SLF001
        await _settle(channel, ended=True)
        # Whatever the verdict, no turn waits on the segment's transcript any more.
        assert channel._turn_transcripts_due == {}  # noqa: SLF001
        return kit, channel, session, seen

    async def test_an_okay_lets_the_bot_talk_on(self) -> None:
        kit, channel, session, seen = await self._speak_and_stop("Okay")

        assert len(seen["backchannel"]) == 1
        assert seen["barge_in"] == []
        assert seen["transcription"] == []
        assert session.id in channel._playing_sessions  # noqa: SLF001
        await kit.close()

    @pytest.mark.parametrize(
        "detector",
        [PhraseBackchannelDetector, _AlwaysPending],
        ids=["phrase-detector", "detector-answering-pending"],
    )
    async def test_a_stop_cuts_the_bot_off_and_is_the_turn(
        self, detector: type[BackchannelDetector]
    ) -> None:
        """A final is never pending (RFC §12.3.13): a detector that answers
        pending for it holds nothing, and its words cut in and are the turn."""
        kit, channel, session, seen = await self._speak_and_stop("Stop", detector=detector())

        assert len(seen["barge_in"]) == 1
        assert [e.text for e in seen["transcription"]] == ["Stop"]
        assert session.id not in channel._playing_sessions  # noqa: SLF001
        await kit.close()

    async def test_a_last_partial_cut_short_on_the_response_stays_its_echo(self) -> None:
        """Closed on a partial and no final, the segment is read as a partial:
        its last word may be the start of the response's (RFC §12.3.13)."""
        kit, channel, session, seen = await self._speak_and_stop("Your appoint", is_final=False)

        assert seen["barge_in"] == []
        assert seen["transcription"] == []
        assert session.id in channel._playing_sessions  # noqa: SLF001
        await kit.close()

    async def test_no_words_is_discarded(self) -> None:
        kit, channel, session, seen = await self._speak_and_stop("")

        assert seen["barge_in"] == []
        assert seen["transcription"] == []
        assert seen["backchannel"] == []
        assert session.id in channel._playing_sessions  # noqa: SLF001
        await kit.close()

    async def test_words_after_the_bot_finished_are_a_plain_turn(self) -> None:
        kit, channel, session, seen = await self._speak_and_stop("Stop", bot_still_talking=False)

        assert seen["barge_in"] == []
        assert [e.text for e in seen["transcription"]] == ["Stop"]
        await kit.close()
