"""A response's own echo does not cut it while the echo canceller learns the room
(RMK-697, RFC §12.3.13).

Each session starts with a fresh canceller: the response's opening words come
back through the microphone, an STT transcribes them ("Salut"), and they cut the
agent off; the final of that echo, seconds later, came in as the person's turn.
"""

from __future__ import annotations

import asyncio
import io
import struct
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult, VoiceCapability
from roomkit.voice.interruption import (
    InterruptionConfig,
    InterruptionHandler,
    InterruptionStrategy,
    echoes,
)
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.backchannel import PhraseBackchannelDetector
from roomkit.voice.stt.base import STTProvider
from tests.channel_tasks import settled

RESPONSE = "Salut ! Ça va très bien, merci. Et toi, comment tu vas ?"
_SILENT_CHUNK = AudioFrame(data=b"\x00\x00" * 1600)
_LOUD_CHUNK = AudioFrame(data=struct.pack("<320h", *([3000, -3000] * 160)))  # 20 ms


@pytest.mark.parametrize(
    ("heard", "echo"),
    [
        ("Salut", True),
        ("Salut.", True),
        ("salut ! ca va", True),  # spelled by the STT, accents and case its own
        ("Salut ! Ça v", True),  # a partial catches the last word being said
        ("Et toi, comment", True),
        ("Salut, attends", False),  # a word the response does not say there
        ("ça va salut", False),  # its words, not in its order
        ("Bonjour", False),
        ("", False),
    ],
)
def test_speech_echoes_a_response_by_its_words_in_order(heard: str, echo: bool) -> None:
    assert echoes(heard, RESPONSE) is echo


@pytest.mark.parametrize(
    ("heard", "response"),
    [
        ("No", "I'm not sure I follow."),
        ("Stop", "It stopped raining."),
        ("Wait", "I'm waiting for the result."),
        ("Yes", "Yesterday it rained."),
        ("Non", "Nonante euros, c'est le prix."),
    ],
)
def test_a_lone_word_is_no_echo_of_a_longer_one(heard: str, response: str) -> None:
    """Only after a whole word may the last one be cut short."""
    assert not echoes(heard, response)


def test_a_final_is_read_in_whole_words() -> None:
    assert not echoes("Salut ! Ça v", RESPONSE, cut_short=False)
    assert echoes("Salut ! Ça va", RESPONSE, cut_short=False)


@pytest.mark.parametrize(("speech_final", "echo"), [(True, False), (False, True)])
def test_the_policy_reads_a_final_in_whole_words(speech_final: bool, echo: bool) -> None:
    """A final ending on the start of the response's word is the person; a
    partial ending so is still the echo (RFC §12.3.13)."""
    handler = InterruptionHandler(InterruptionConfig(strategy=InterruptionStrategy.IMMEDIATE))

    decision = handler.evaluate(
        playback_position_ms=900,
        speech_text="Salut ! Ça v",
        response_text=RESPONSE,
        speech_final=speech_final,
    )

    assert decision.is_echo is echo


@pytest.mark.parametrize(
    "strategy", [InterruptionStrategy.IMMEDIATE, InterruptionStrategy.DISABLED]
)
def test_an_echo_early_in_the_session_neither_cuts_nor_waits(strategy) -> None:
    handler = InterruptionHandler(InterruptionConfig(strategy=strategy))

    decision = handler.evaluate(
        playback_position_ms=900, speech_text="Salut", response_text=RESPONSE
    )

    assert (decision.is_echo, decision.should_interrupt, decision.queue_speech) == (
        True,
        False,
        False,
    )


@pytest.mark.parametrize(
    ("config", "session_played_ms"),
    [
        (InterruptionConfig(strategy=InterruptionStrategy.IMMEDIATE), 6000),
        (InterruptionConfig(strategy=InterruptionStrategy.IMMEDIATE, echo_guard_ms=0), 0),
    ],
)
def test_once_the_canceller_has_learned_the_room_its_words_cut_in(
    config: InterruptionConfig, session_played_ms: int
) -> None:
    """ "The sword" to "the sword or the shield?" is an answer, not an echo."""
    decision = InterruptionHandler(config).evaluate(
        playback_position_ms=900,
        speech_text="Salut",
        response_text=RESPONSE,
        session_played_ms=session_played_ms,
    )

    assert decision.should_interrupt and not decision.is_echo


class _StagedSTT(STTProvider):
    """Continuous STT: the partials at once, the final once ``release`` is set."""

    def __init__(self, partials: list[str], final: str) -> None:
        self._partials, self._final = partials, final
        self.release = asyncio.Event()
        self.partials_said = asyncio.Event()
        self.final_taken = asyncio.Event()

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="", is_final=True)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            break
        for text in self._partials:
            yield TranscriptionResult(text=text, is_final=False)
        self.partials_said.set()
        await self.release.wait()
        try:
            yield TranscriptionResult(text=self._final, is_final=True)
        finally:
            # Read: the channel asks for the next result, or closes the stream.
            self.final_taken.set()
        await asyncio.Event().wait()  # the stream stays open, as a live one does


async def _settle(channel: VoiceChannel) -> None:
    """Until the work the channel started from what it heard is done: the tasks
    it schedules for a barge-in or a turn, which its close cancels. Its
    continuous STT loop runs as long as its session and is left out."""
    await settled(channel, lasting=("run_continuous",), timeout=2)


async def _speak_over(partials: list[str], final: str, *, played_before_ms: int = 0):
    """A response playing, *partials* over it, then *final* once it ended."""
    stt = _StagedSTT(partials, final)
    backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
    channel = VoiceChannel(
        "voice-1",
        stt=stt,
        backend=backend,
        pipeline=AudioPipelineConfig(),  # no VAD + streaming STT = continuous
        interruption=InterruptionConfig(strategy=InterruptionStrategy.IMMEDIATE),
    )
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice-1")
    session = await kit.connect_voice("r1", "user-1", "voice-1")
    turns: list[str] = []
    cuts: list[object] = []
    partials: list[str] = []

    @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
    async def on_barge_in(event: object, context: object) -> None:
        cuts.append(event)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event: Any, context: object) -> HookResult:
        turns.append(event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_PARTIAL_TRANSCRIPTION, HookExecution.ASYNC)
    async def on_partial(event: Any, context: object) -> None:
        partials.append(event.text)

    if played_before_ms:
        channel._session_played_ms[session.id] = played_before_ms  # noqa: SLF001
    channel._playing_sessions[session.id] = TTSPlaybackState(  # noqa: SLF001
        session_id=session.id, text=RESPONSE
    )
    await backend.simulate_audio_received(session, _SILENT_CHUNK)
    await asyncio.wait_for(stt.partials_said.wait(), 2)
    await _settle(channel)
    channel._playing_sessions.pop(session.id, None)  # noqa: SLF001 — the response ended
    channel._last_tts_ended_at[session.id] = time.monotonic()  # noqa: SLF001
    stt.release.set()
    await asyncio.wait_for(stt.final_taken.wait(), 2)
    await _settle(channel)
    await channel.close()
    await kit.close()
    return turns, cuts, partials


async def test_the_echo_of_a_response_opening_neither_cuts_it_nor_becomes_a_turn() -> None:
    # The final comes after the response ended, as Meta's did 5.5 s later.
    turns, cuts, partials = await _speak_over(["Salut", "Salut ! Ça va"], "Salut")

    assert cuts == []
    assert turns == []
    assert partials == []  # nobody's words: a chat shows none of them


async def test_words_the_response_does_not_say_cut_it_and_are_the_turn() -> None:
    turns, cuts, partials = await _speak_over(
        ["Salut", "Salut, attends"], "Salut, attends une seconde"
    )

    assert len(cuts) == 1
    assert turns == ["Salut, attends une seconde"]
    assert partials == ["Salut, attends"]


async def test_past_the_guard_the_response_words_cut_in_and_are_the_turn() -> None:
    turns, cuts, _ = await _speak_over(["Salut"], "Salut", played_before_ms=10_000)

    assert len(cuts) == 1
    assert turns == ["Salut"]


class _ScriptedSTT(STTProvider):
    """Continuous STT: each step a result, or ``("wait", name)`` until ``go(name)``."""

    def __init__(self, script: list[Any]) -> None:
        self._script = script
        self._go: dict[str, asyncio.Event] = {}
        self._at: dict[str, asyncio.Event] = {}

    def go(self, name: str) -> None:
        self._go.setdefault(name, asyncio.Event()).set()

    async def reached(self, name: str) -> None:
        await asyncio.wait_for(self._at.setdefault(name, asyncio.Event()).wait(), 2)

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="", is_final=True)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            break
        try:
            while self._script:
                step = self._script.pop(0)
                if isinstance(step, tuple):
                    self._at.setdefault(step[1], asyncio.Event()).set()
                    await self._go.setdefault(step[1], asyncio.Event()).wait()
                    continue
                yield step
        finally:
            # Every step read, the stream then kept or closed by the channel.
            if not self._script:
                self._at.setdefault("end", asyncio.Event()).set()
        await asyncio.Event().wait()


def _partial(text: str) -> TranscriptionResult:
    return TranscriptionResult(text=text, is_final=False)


def _final(text: str) -> TranscriptionResult:
    return TranscriptionResult(text=text, is_final=True)


class _Call:
    """A continuous-STT session whose response is set by hand, with what came of it."""

    def __init__(self, script: list[Any], strategy: InterruptionStrategy, **config: Any) -> None:
        self.stt = _ScriptedSTT(script)
        self.backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
        detector = {}
        if strategy == InterruptionStrategy.SEMANTIC:
            detector = {"backchannel_detector": PhraseBackchannelDetector()}
        self.channel = VoiceChannel(
            "voice-1",
            stt=self.stt,
            backend=self.backend,
            pipeline=AudioPipelineConfig(),
            interruption=InterruptionConfig(strategy=strategy, **detector, **config),
        )
        self.kit = RoomKit(stt=self.stt, voice=self.backend)
        self.kit.register_channel(self.channel)
        self.turns: list[str] = []
        self.cuts: list[object] = []

    async def __aenter__(self) -> _Call:
        await self.kit.create_room(room_id="r1")
        await self.kit.attach_channel("r1", "voice-1")
        self.session = await self.kit.connect_voice("r1", "user-1", "voice-1")

        @self.kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
        async def on_barge_in(event: object, context: object) -> None:
            self.cuts.append(event)

        @self.kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def on_transcription(event: Any, context: object) -> HookResult:
            self.turns.append(event.text)
            return HookResult.allow()

        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.channel.close()
        await self.kit.close()

    def play(self, text: str) -> None:
        self.channel._playing_sessions[self.session.id] = TTSPlaybackState(  # noqa: SLF001
            session_id=self.session.id, text=text
        )

    def end(self, *, seconds_ago: float = 0.0) -> None:
        """The response ended *seconds_ago*."""
        self.channel._playing_sessions.pop(self.session.id, None)  # noqa: SLF001
        ended = time.monotonic() - seconds_ago
        self.channel._last_tts_ended_at[self.session.id] = ended  # noqa: SLF001

    async def hear(self, frame: AudioFrame = _SILENT_CHUNK) -> None:
        await self.backend.simulate_audio_received(self.session, frame)

    async def heard(self, step: str) -> None:
        """Until the STT reached *step* and the channel's work on what came
        before it is done."""
        await self.stt.reached(step)
        await _settle(self.channel)


async def test_a_one_word_answer_starting_a_word_of_the_response_cuts_in() -> None:
    """ "No" over "I'm not sure": a lone word is matched whole (review of RMK-697)."""
    script = [_partial("No"), ("wait", "final"), _final("No.")]
    async with _Call(script, InterruptionStrategy.IMMEDIATE) as call:
        call.play("I'm not sure I follow. Could you say that again?")
        await call.hear()
        await call.heard("final")
        call.end()
        call.stt.go("final")
        await call.heard("end")

    assert len(call.cuts) == 1
    assert call.turns == ["No."]


@pytest.mark.parametrize(
    "strategy",
    [
        InterruptionStrategy.IMMEDIATE,
        InterruptionStrategy.CONFIRMED,
        InterruptionStrategy.SEMANTIC,
    ],
)
async def test_a_final_saying_more_than_the_echo_cuts_in_while_it_plays(
    strategy: InterruptionStrategy,
) -> None:
    """Its only partial echoed the response; the final, during playback, is the
    person's: before, it was thrown away as echo (review of RMK-697)."""
    script = [_partial("Salut"), ("wait", "final"), _final("Salut, attends une seconde")]
    async with _Call(script, strategy) as call:
        call.play(RESPONSE)
        await call.hear()
        await call.heard("final")
        call.stt.go("final")
        await call.heard("end")

    assert len(call.cuts) == 1
    assert call.turns == ["Salut, attends une seconde"]


async def test_an_echo_verdict_ends_with_its_response() -> None:
    """An echo whose utterance closed without words leaves no verdict on the
    person's answer, once the response is long over (review of RMK-697)."""
    script = [
        _partial("Salut"),
        _final(""),
        ("wait", "later"),
        _partial("Très bien"),
        _final("Très bien, merci"),
    ]
    async with _Call(script, InterruptionStrategy.IMMEDIATE) as call:
        call.play(RESPONSE)
        await call.hear()
        await call.heard("later")
        call.end(seconds_ago=60)
        call.stt.go("later")
        await call.heard("end")

    assert call.turns == ["Très bien, merci"]


async def test_speech_starting_once_nothing_plays_is_no_echo() -> None:
    script = [
        _partial("Salut"),
        ("wait", "later"),
        TranscriptionResult(text="", is_final=False, is_speech_start=True),
        _final("Très bien, merci"),
    ]
    async with _Call(script, InterruptionStrategy.IMMEDIATE) as call:
        call.play(RESPONSE)
        await call.hear()
        await call.heard("later")
        call.end()
        call.stt.go("later")
        await call.heard("end")

    assert call.turns == ["Très bien, merci"]


@pytest.mark.parametrize(
    "strategy", [InterruptionStrategy.IMMEDIATE, InterruptionStrategy.CONFIRMED]
)
async def test_loud_echo_before_its_words_waits_for_them(strategy: InterruptionStrategy) -> None:
    """The energy run reaches the channel before the STT's first partial: while
    the guard is open it waits for words, and they are the echo's (review of
    RMK-697)."""
    script = [("wait", "words"), _partial("Salut"), ("wait", "final"), _final("Salut")]
    # Past min_speech_ms, a run with no guard would cut under either strategy.
    async with _Call(script, strategy, min_speech_ms=60) as call:
        call.play(RESPONSE)
        loop = asyncio.get_running_loop()
        started = loop.time()
        while loop.time() - started < 0.15:  # loud echo, words not in yet
            await call.hear(_LOUD_CHUNK)
            await asyncio.sleep(0.02)
        call.stt.go("words")
        await call.heard("final")
        call.end()
        call.stt.go("final")
        await call.heard("end")

    assert call.cuts == []
    assert call.turns == []


async def test_a_wav_played_whole_counts_toward_the_guard(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x01\x00" * 16000 * 8)  # 8 s
    greeting = tmp_path / "greeting.wav"
    greeting.write_bytes(buffer.getvalue())

    async with _Call([("wait", "never")], InterruptionStrategy.IMMEDIATE) as call:
        await call.channel.play(call.session, str(greeting), text="Hello and welcome.")
        played = call.channel._session_played_ms.get(call.session.id)  # noqa: SLF001

    assert played == 8000


class _UnfinishedPartialSTT(STTProvider):
    """Continuous STT: one partial at the first audio, and never its final."""

    def __init__(self, partial: str) -> None:
        self._partial = partial
        self.said = asyncio.Event()

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="", is_final=True)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            break
        try:
            yield TranscriptionResult(text=self._partial, is_final=False)
        finally:
            self.said.set()  # read by the channel
        async for _ in audio_stream:
            pass


async def _echo_then_next_response(*, grace_passed: bool) -> list[object]:
    """A response whose opening echo ("Je") the STT never finalized, cut short
    by the next one (a task's result relayed), the room still ringing with it:
    the barge-ins that next response gets."""
    import time
    from datetime import UTC, datetime

    from roomkit.models.channel import ChannelBinding
    from roomkit.models.enums import ChannelType
    from roomkit.voice.pipeline.backchannel import PhraseBackchannelDetector

    backend = MockVoiceBackend()
    stt = _UnfinishedPartialSTT("Je")
    channel = VoiceChannel(
        "voice-1",
        stt=stt,
        backend=backend,
        pipeline=AudioPipelineConfig(),
        interruption=InterruptionConfig(
            strategy=InterruptionStrategy.SEMANTIC,
            backchannel_detector=PhraseBackchannelDetector(),
            min_speech_ms=50,
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
    cuts: list[object] = []

    @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
    async def on_barge_in(event: object, context: object) -> None:
        cuts.append(event)

    playing = channel._playing_sessions  # noqa: SLF001
    playing[session.id] = TTSPlaybackState(
        session_id=session.id, text="Je lance ça en arrière-plan.", started_at=datetime.now(UTC)
    )
    for _ in range(3):
        await backend.simulate_audio_received(session, _SILENT_CHUNK)
    await asyncio.wait_for(stt.said.wait(), 2)
    await _settle(channel)
    assert channel._echo_bursts.get(session.id)  # noqa: SLF001 — "Je" taken for the echo

    await channel.interrupt(session, reason="new_tts")
    if grace_passed:
        channel._last_tts_ended_at[session.id] = time.monotonic() - 60  # noqa: SLF001
    playing[session.id] = TTSPlaybackState(
        session_id=session.id,
        text="Voici les campagnes : Le Trésor du Dragon.",
        started_at=datetime.now(UTC),
    )
    loud = AudioFrame(data=(2000).to_bytes(2, "little", signed=True) * 320)
    for _ in range(8):  # 160 ms of the cut response's tail: past 5 frames and 50 ms
        await backend.simulate_audio_received(session, loud)
        await asyncio.sleep(0.02)
    await _settle(channel)
    await channel.close()
    await kit.close()
    return cuts


async def test_words_taken_for_an_echo_cut_no_later_response() -> None:
    assert await _echo_then_next_response(grace_passed=False) == []


async def test_an_echo_verdict_past_its_grace_takes_its_words_with_it() -> None:
    # Stale, "Je" would be the words of whatever is heard next, and cut.
    assert await _echo_then_next_response(grace_passed=True) == []


# -- the agent's own voice, past the guard (diarizing STT) ---------------------------

_LAMP = "Oui, c'est bien la lampe Salon (Hue, id 525) : elle est allumée, à 92 %."


class _ScriptedDiarizingSTT(STTProvider):
    """A diarizing streaming STT that says what the test hands it, when it does."""

    def __init__(self) -> None:
        # ``None`` closes the stream, as a provider does; the channel opens the next.
        self.results: asyncio.Queue[TranscriptionResult | None] = asyncio.Queue()

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_diarization(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="", is_final=True)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            break
        while (result := await self.results.get()) is not None:
            try:
                yield result
            finally:
                # Read: the channel asks for the next result, or closes the stream.
                self.results.task_done()
        self.results.task_done()

    async def read(self, channel: VoiceChannel) -> None:
        """Until the channel read every result put and its work on them is done."""
        await asyncio.wait_for(self.results.join(), 2)
        await _settle(channel)


async def test_a_later_partial_of_an_echo_cuts_no_later_response() -> None:
    """The STT's next partial of an utterance taken for a response's echo,
    still that response's words, is judged against nothing: the next response,
    which took the session over, does not say them (RMK-703, partial path)."""
    from datetime import UTC, datetime

    from roomkit.models.channel import ChannelBinding
    from roomkit.models.enums import ChannelType

    stt = _ScriptedDiarizingSTT()
    backend = MockVoiceBackend()
    channel = VoiceChannel(
        "voice-1",
        stt=stt,
        backend=backend,
        pipeline=AudioPipelineConfig(),
        interruption=InterruptionConfig(
            strategy=InterruptionStrategy.SEMANTIC,
            backchannel_detector=PhraseBackchannelDetector(),
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
    cuts: list[object] = []

    @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
    async def on_barge_in(event: object, context: object) -> None:
        cuts.append(event)

    await backend.simulate_audio_received(session, _SILENT_CHUNK)
    playing = channel._playing_sessions  # noqa: SLF001
    playing[session.id] = TTSPlaybackState(
        session_id=session.id,
        text="I'm starting that in the background.",
        started_at=datetime.now(UTC),
    )
    await stt.results.put(TranscriptionResult(text="", is_speech_start=True))
    await stt.results.put(TranscriptionResult(text="I'm", is_final=False))
    await stt.read(channel)
    assert channel._echo_bursts.get(session.id)  # noqa: SLF001 — taken for the echo

    # Sent whole, the next response takes the session over.
    channel._last_tts_ended_at[session.id] = time.monotonic()  # noqa: SLF001
    playing[session.id] = TTSPlaybackState(
        session_id=session.id,
        text="Here are the campaigns: The Dragon's Treasure.",
        started_at=datetime.now(UTC),
    )
    await stt.results.put(TranscriptionResult(text="I'm starting that", is_final=False))
    await stt.read(channel)
    await channel.close()
    await kit.close()

    assert cuts == []


async def _voices_over_a_response(
    utterances: list[tuple[str, str, bool] | None],
) -> tuple[list[tuple[str, str]], list[object]]:
    """Each (voice, words, over the response) said in turn, the session past its
    echo guard; ``None`` for the provider closing its stream. The turns
    (voice, words) and barge-ins that came of them."""
    from datetime import UTC, datetime

    from roomkit.models.channel import ChannelBinding
    from roomkit.models.enums import ChannelType
    from roomkit.voice.base import SpeakerSegment
    from roomkit.voice.pipeline.backchannel import PhraseBackchannelDetector

    stt = _ScriptedDiarizingSTT()
    backend = MockVoiceBackend()
    channel = VoiceChannel(
        "voice-1",
        stt=stt,
        backend=backend,
        pipeline=AudioPipelineConfig(),
        interruption=InterruptionConfig(
            strategy=InterruptionStrategy.SEMANTIC,
            backchannel_detector=PhraseBackchannelDetector(),
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
    channel._session_played_ms[session.id] = 60_000  # noqa: SLF001 — past the guard
    turns: list[tuple[str, str]] = []
    cuts: list[object] = []

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event: Any, context: object) -> HookResult:
        turns.append((str(event.speaker or ""), event.text))
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
    async def on_barge_in(event: object, context: object) -> None:
        cuts.append(event)

    await backend.simulate_audio_received(session, _SILENT_CHUNK)
    playing = channel._playing_sessions  # noqa: SLF001
    for utterance in utterances:
        if utterance is None:
            await stt.results.put(None)
            await stt.read(channel)  # the stream closed
            await backend.simulate_audio_received(session, _SILENT_CHUNK)  # the next one opens
            continue
        voice, words, over = utterance
        if over:
            playing[session.id] = TTSPlaybackState(
                session_id=session.id, text=_LAMP, started_at=datetime.now(UTC)
            )
        await stt.results.put(TranscriptionResult(text="", is_speech_start=True))
        # Word by word, as a streaming STT writes them: a first word that may
        # open an acknowledgement ("Eh" of "eh boy") decides nothing alone.
        for count in (1, 2):
            partial = " ".join(words.split()[:count])
            await stt.results.put(TranscriptionResult(text=partial, is_final=False))
        await stt.read(channel)
        playing.pop(session.id, None)  # cut, or over
        await stt.results.put(
            TranscriptionResult(text=words, segments=[SpeakerSegment(voice, words)])
        )
        await stt.read(channel)
    await channel.close()
    await kit.close()
    return turns, cuts


async def test_the_agents_voice_misheard_past_the_guard_is_no_turn() -> None:
    turns, _cuts = await _voices_over_a_response(
        [
            ("Speaker A", "Et la lampe du salon ?", False),
            # "c'est bien la lampe Salon" back through the canceller, another
            # voice to the STT, one word misheard
            ("Speaker B", "Eh bien, la lampe salon.", True),
        ]
    )

    assert [words for _voice, words in turns] == ["Et la lampe du salon ?"]


async def test_a_person_heard_alone_repeating_the_response_is_heard() -> None:
    turns, cuts = await _voices_over_a_response(
        [
            ("Speaker A", "Et la lampe du salon ?", False),
            ("Speaker A", "Eh bien, la lampe salon.", True),
        ]
    )

    assert [words for _voice, words in turns][-1] == "Eh bien, la lampe salon."
    assert len(cuts) == 1


async def test_a_voice_heard_alone_on_a_closed_stream_is_nobody_on_the_next() -> None:
    """Each stream numbers its voices afresh: "Speaker A" heard alone on a
    stream the provider closed names nobody on the next, where the same label
    over the response, in its words, is the response's echo (RFC §12.3.13)."""
    turns, _cuts = await _voices_over_a_response(
        [
            ("Speaker A", "Et la lampe du salon ?", False),
            None,
            ("Speaker A", "Eh bien, la lampe salon.", True),
        ]
    )

    assert [words for _voice, words in turns] == ["Et la lampe du salon ?"]


async def test_a_new_voice_saying_its_own_words_over_a_response_is_heard() -> None:
    turns, _cuts = await _voices_over_a_response(
        [
            ("Speaker A", "Et la lampe du salon ?", False),
            ("Speaker B", "Attends, je veux la cuisine.", True),
        ]
    )

    assert [words for _voice, words in turns][-1] == "Attends, je veux la cuisine."


@pytest.mark.parametrize(
    ("heard", "near"),
    [
        ("Eh bien, la lampe salon.", True),  # one word misheard
        ("c'est bien la lampe salon", True),
        ("la lampe salon", True),
        ("la lampe", False),  # too few words to tell
        ("Eh oui, la lampe cuisine.", False),  # two words differ
    ],
)
def test_nearly_the_responses_words(heard: str, near: bool) -> None:
    from roomkit.voice.interruption import echoes_nearly

    assert echoes_nearly(heard, _LAMP) is near
