"""Tests for turn detection integration in VoiceChannel."""

from __future__ import annotations

import asyncio
import threading

from roomkit.channels.voice import VoiceChannel
from roomkit.models.enums import HookTrigger
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.config import AudioPipelineConfig
from roomkit.voice.pipeline.turn.base import TurnDecision
from roomkit.voice.pipeline.turn.mock import MockTurnDetector
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.pipeline.vad.mock import MockVADProvider


class _MockSTT:
    """Minimal mock STT that returns preconfigured transcripts."""

    name = "mock_stt"
    supports_streaming = False

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = transcripts
        self._index = 0

    async def transcribe(self, frame: AudioFrame):
        from roomkit.voice.base import TranscriptionResult

        if self._index < len(self._transcripts):
            text = self._transcripts[self._index]
            self._index += 1
            return TranscriptionResult(text=text)
        return TranscriptionResult(text="")

    async def close(self) -> None:
        pass


class _MockBackend:
    """Minimal mock backend for wiring."""

    name = "mock_backend"

    from roomkit.voice.base import VoiceCapability

    _caps = VoiceCapability.NONE
    _sessions: dict = {}
    _barge_in_cbs: list = []
    _audio_cbs: list = []

    @property
    def capabilities(self):
        return self._caps

    @property
    def feeds_aec_reference(self):
        return False

    @property
    def supports_playback_callback(self):
        return False

    def on_audio_received(self, cb):
        self._audio_cbs.append(cb)

    def on_barge_in(self, cb):
        self._barge_in_cbs.append(cb)

    def on_session_ready(self, cb):
        pass

    def on_client_disconnected(self, cb):
        pass

    async def send_transcription(self, session, text, role):
        pass

    def get_session(self, sid):
        return None

    def list_sessions(self, room_id):
        return []

    async def cancel_audio(self, session):
        pass

    async def close(self):
        pass

    async def simulate_audio(self, session, frame):
        for cb in self._audio_cbs:
            cb(session, frame)


class TestTurnDetection:
    async def test_complete_turn_routes_immediately(self):
        """When turn detector says complete, text is routed and ON_TURN_COMPLETE fires."""
        from unittest.mock import AsyncMock, MagicMock

        from roomkit.voice.base import VoiceSession

        # Turn detector returns complete
        detector = MockTurnDetector(
            decisions=[TurnDecision(is_complete=True, confidence=0.9, reason="done")]
        )
        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio\x00"),
            ]
        )
        config = AudioPipelineConfig(vad=vad, turn_detector=detector)
        stt = _MockSTT(transcripts=["Hello world"])
        backend = _MockBackend()

        channel = VoiceChannel("ch1", stt=stt, backend=backend, pipeline=config)

        # Mock framework
        mock_fw = AsyncMock()
        mock_fw._build_context = AsyncMock(return_value=AsyncMock())
        mock_fw._hold_channel_work = MagicMock()
        mock_fw.hook_engine.has_hooks = MagicMock(return_value=True)
        mock_fw.hook_engine.run_async_hooks = AsyncMock()
        mock_fw.hook_engine.run_sync_hooks = AsyncMock(
            return_value=AsyncMock(allowed=True, event="Hello world")
        )
        mock_fw.process_inbound = AsyncMock()
        channel.set_framework(mock_fw)

        session = VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="ch1")
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType

        binding = ChannelBinding(room_id="r1", channel_id="ch1", channel_type=ChannelType.VOICE)
        channel.bind_session(session, "r1", binding)

        # Simulate audio -> speech end -> transcribe -> turn detect
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await _until(lambda: mock_fw.process_inbound.called)

        # Check ON_TURN_COMPLETE was fired
        turn_complete_calls = [
            c
            for c in mock_fw.hook_engine.run_async_hooks.call_args_list
            if c.args[1] == HookTrigger.ON_TURN_COMPLETE
        ]
        assert len(turn_complete_calls) == 1

        # Check text was routed
        assert mock_fw.process_inbound.called

    async def test_incomplete_turn_accumulates(self):
        """When turn detector says incomplete, text accumulates."""
        from unittest.mock import AsyncMock, MagicMock

        from roomkit.voice.base import VoiceSession

        # First call: incomplete. Second call: complete.
        detector = MockTurnDetector(
            decisions=[
                TurnDecision(is_complete=False, confidence=0.6, reason="trailing"),
                TurnDecision(is_complete=True, confidence=0.95, reason="done"),
            ]
        )
        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1"),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio2"),
            ]
        )
        config = AudioPipelineConfig(vad=vad, turn_detector=detector)
        stt = _MockSTT(transcripts=["Hello", "world"])
        backend = _MockBackend()

        channel = VoiceChannel("ch1", stt=stt, backend=backend, pipeline=config)

        mock_fw = AsyncMock()
        mock_fw._build_context = AsyncMock(return_value=AsyncMock())
        mock_fw._hold_channel_work = MagicMock()
        mock_fw.hook_engine.has_hooks = MagicMock(return_value=True)
        mock_fw.hook_engine.run_async_hooks = AsyncMock()
        mock_fw.hook_engine.run_sync_hooks = AsyncMock(
            side_effect=[
                AsyncMock(allowed=True, event="Hello"),
                AsyncMock(allowed=True, event="world"),
            ]
        )
        mock_fw.process_inbound = AsyncMock()
        channel.set_framework(mock_fw)

        session = VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="ch1")
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType

        binding = ChannelBinding(room_id="r1", channel_id="ch1", channel_type=ChannelType.VOICE)
        channel.bind_session(session, "r1", binding)

        # First utterance — incomplete, should not route
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await _judged_incomplete(mock_fw)

        incomplete_calls = [
            c
            for c in mock_fw.hook_engine.run_async_hooks.call_args_list
            if c.args[1] == HookTrigger.ON_TURN_INCOMPLETE
        ]
        assert len(incomplete_calls) == 1
        assert not mock_fw.process_inbound.called

        # Second utterance — complete, should route combined text
        await backend.simulate_audio(session, AudioFrame(data=b"\x01\x00"))
        await _until(lambda: mock_fw.process_inbound.called)

        complete_calls = [
            c
            for c in mock_fw.hook_engine.run_async_hooks.call_args_list
            if c.args[1] == HookTrigger.ON_TURN_COMPLETE
        ]
        assert len(complete_calls) == 1
        assert mock_fw.process_inbound.called

        # The routed text should be the combined "Hello world"
        inbound_call = mock_fw.process_inbound.call_args
        assert "Hello world" in str(inbound_call)

        # The detector reads the latest utterance as transcript, and the turn
        # under evaluation as its history (RFC §12.3.12)
        handed = [
            (c.transcript, [(e.role, e.text) for e in c.conversation_history])
            for c in detector.evaluations
        ]
        assert handed == [
            ("Hello", [("user", "Hello")]),
            ("world", [("user", "Hello"), ("user", "world")]),
        ]

    async def test_no_turn_detector_routes_immediately(self):
        """Without a turn detector, text is routed immediately (existing behaviour)."""
        from unittest.mock import AsyncMock, MagicMock

        from roomkit.voice.base import VoiceSession

        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio\x00"),
            ]
        )
        config = AudioPipelineConfig(vad=vad)  # No turn_detector
        stt = _MockSTT(transcripts=["Hello"])
        backend = _MockBackend()

        channel = VoiceChannel("ch1", stt=stt, backend=backend, pipeline=config)

        mock_fw = AsyncMock()
        mock_fw._build_context = AsyncMock(return_value=AsyncMock())
        mock_fw._hold_channel_work = MagicMock()
        mock_fw.hook_engine.has_hooks = MagicMock(return_value=True)
        mock_fw.hook_engine.run_async_hooks = AsyncMock()
        mock_fw.hook_engine.run_sync_hooks = AsyncMock(
            return_value=AsyncMock(allowed=True, event="Hello")
        )
        mock_fw.process_inbound = AsyncMock()
        channel.set_framework(mock_fw)

        session = VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="ch1")
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType

        binding = ChannelBinding(room_id="r1", channel_id="ch1", channel_type=ChannelType.VOICE)
        channel.bind_session(session, "r1", binding)

        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await _until(lambda: mock_fw.process_inbound.called)

        # Should route directly without turn hooks
        assert mock_fw.process_inbound.called
        turn_calls = [
            c
            for c in mock_fw.hook_engine.run_async_hooks.call_args_list
            if c.args[1] in (HookTrigger.ON_TURN_COMPLETE, HookTrigger.ON_TURN_INCOMPLETE)
        ]
        assert len(turn_calls) == 0


class _LateSTT(_MockSTT):
    """Transcripts after the first come once ``release`` is set, as a streaming
    STT's final comes after the speech."""

    def __init__(self, transcripts: list[str]) -> None:
        super().__init__(transcripts)
        self.release = asyncio.Event()

    async def transcribe(self, frame: AudioFrame):
        from roomkit.voice.base import TranscriptionResult

        index = self._index
        self._index += 1
        if index:
            await self.release.wait()
        return TranscriptionResult(text=self._transcripts[index])


class _SilentSTT(_MockSTT):
    """An STT that answers the first segment and never the next ones."""

    async def transcribe(self, frame: AudioFrame):
        if self._index == 0:
            return await super().transcribe(frame)
        await asyncio.Event().wait()


def _wired_channel(detector, vad_events, transcripts, *, stt=None, **config_kwargs):
    """A VoiceChannel on mocks, bound to one session, with a recording framework."""
    from unittest.mock import AsyncMock, MagicMock

    from roomkit.models.channel import ChannelBinding
    from roomkit.models.enums import ChannelType
    from roomkit.voice.base import VoiceSession

    vad = MockVADProvider(events=vad_events)
    config = AudioPipelineConfig(vad=vad, turn_detector=detector, **config_kwargs)
    backend = _MockBackend()
    backend._audio_cbs = []
    stt = stt if stt is not None else _MockSTT(transcripts)
    channel = VoiceChannel("ch1", stt=stt, backend=backend, pipeline=config)
    fw = AsyncMock()
    fw._build_context = AsyncMock(return_value=AsyncMock())
    fw._hold_channel_work = MagicMock()
    fw.hook_engine.has_hooks = MagicMock(return_value=True)
    fw.hook_engine.run_async_hooks = AsyncMock()
    fw.hook_engine.run_sync_hooks = AsyncMock(
        side_effect=[AsyncMock(allowed=True, event=t) for t in transcripts]
    )
    fw.process_inbound = AsyncMock()
    channel.set_framework(fw)
    session = VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="ch1")
    binding = ChannelBinding(room_id="r1", channel_id="ch1", channel_type=ChannelType.VOICE)
    channel.bind_session(session, "r1", binding)
    return channel, backend, session, fw


def _routed_texts(fw) -> list[str]:
    return [c.args[0].content.body for c in fw.process_inbound.call_args_list]


def _fired(fw, trigger: HookTrigger) -> int:
    return sum(1 for c in fw.hook_engine.run_async_hooks.call_args_list if c.args[1] == trigger)


async def _until(predicate, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


async def _routed(fw, texts: list[str]) -> None:
    """Until the turns routed are *texts*."""
    await _until(lambda: _routed_texts(fw) == texts)


async def _judged_incomplete(fw) -> None:
    """Until the detector's verdict made the turn wait (ON_TURN_INCOMPLETE fired)."""
    await _until(lambda: _fired(fw, HookTrigger.ON_TURN_INCOMPLETE) >= 1)


# The incomplete turn's wait under test, and a pause past it: what the wait
# would have routed by then, it routed.
_WAIT_MS = 50
_PAST_THE_WAIT = 2 * _WAIT_MS / 1000


_INCOMPLETE = TurnDecision(is_complete=False, confidence=0.3, reason="trailing")


class _SlowTurnDetector(MockTurnDetector):
    """A detector that takes as long to decide as Smart Turn on a CPU: its first
    verdict comes once ``release`` is set, ``deciding`` set meanwhile and
    ``decided`` once it is given."""

    def __init__(self, decisions: list[TurnDecision]) -> None:
        super().__init__(decisions=decisions)
        self.deciding, self.release = threading.Event(), threading.Event()
        self.decided = threading.Event()

    def evaluate(self, context):
        first = not self.evaluations
        if first:
            self.deciding.set()
            self.release.wait(2)
        decision = super().evaluate(context)
        if first:
            self.decided.set()
        return decision

    async def decides(self) -> None:
        """Until its first verdict is being decided (in a worker thread)."""
        assert await asyncio.to_thread(self.deciding.wait, 2)


class TestIncompleteTurnWait:
    """RFC §12: an incomplete turn is routed once its wait ends in silence (RMK-218)."""

    async def test_silence_after_an_incomplete_turn_routes_it(self):
        channel, backend, session, fw = _wired_channel(
            MockTurnDetector(decisions=[_INCOMPLETE]),
            [VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1")],
            ["Combien de boards je vois,"],
            turn_incomplete_wait_ms=_WAIT_MS,
        )
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await _judged_incomplete(fw)
        assert _routed_texts(fw) == []  # still waiting

        await _routed(fw, ["Combien de boards je vois,"])
        complete = [
            c
            for c in fw.hook_engine.run_async_hooks.call_args_list
            if c.args[1] == HookTrigger.ON_TURN_COMPLETE
        ]
        assert len(complete) == 1

    async def test_speech_during_the_wait_cancels_it_and_joins_the_turn(self):
        channel, backend, session, fw = _wired_channel(
            MockTurnDetector(
                decisions=[_INCOMPLETE, TurnDecision(is_complete=True, confidence=0.9)]
            ),
            [
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1"),
                VADEvent(type=VADEventType.SPEECH_START),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio2"),
            ],
            ["Combien de boards je vois,", "et lequel a le plus de cartes ?"],
            turn_incomplete_wait_ms=_WAIT_MS,
        )
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await _judged_incomplete(fw)
        await backend.simulate_audio(session, AudioFrame(data=b"\x01\x00"))  # speech starts
        await asyncio.sleep(_PAST_THE_WAIT)  # the wait would have elapsed by now
        assert _routed_texts(fw) == []

        await backend.simulate_audio(session, AudioFrame(data=b"\x02\x00"))  # speech ends
        await _routed(fw, ["Combien de boards je vois, et lequel a le plus de cartes ?"])

    async def test_the_detectors_suggested_wait_wins_over_the_default(self):
        channel, backend, session, fw = _wired_channel(
            MockTurnDetector(
                decisions=[TurnDecision(is_complete=False, confidence=0.3, suggested_wait_ms=50)]
            ),
            [VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1")],
            ["Et donc"],
            turn_incomplete_wait_ms=5000,
        )
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await _routed(fw, ["Et donc"])  # long before the default's 5 s

    async def test_a_session_that_ends_during_the_wait_routes_nothing(self):
        channel, backend, session, fw = _wired_channel(
            MockTurnDetector(decisions=[_INCOMPLETE]),
            [VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1")],
            ["Au revoir"],
            turn_incomplete_wait_ms=_WAIT_MS,
        )
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await _judged_incomplete(fw)
        channel.unbind_session(session)
        await asyncio.sleep(_PAST_THE_WAIT)
        assert _routed_texts(fw) == []

    async def test_a_segment_that_yields_no_text_does_not_hold_the_turn(self):
        # A cough during the wait: speech starts and ends, the STT hears nothing.
        channel, backend, session, fw = _wired_channel(
            MockTurnDetector(decisions=[_INCOMPLETE]),
            [
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1"),
                VADEvent(type=VADEventType.SPEECH_START),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"cough"),
            ],
            ["Combien de boards je vois,", ""],
            turn_incomplete_wait_ms=_WAIT_MS,
        )
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await _judged_incomplete(fw)
        await backend.simulate_audio(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio(session, AudioFrame(data=b"\x02\x00"))
        await _routed(fw, ["Combien de boards je vois,"])

    async def test_speech_that_starts_while_the_turn_is_judged_keeps_it_open(self):
        detector = _SlowTurnDetector(
            decisions=[_INCOMPLETE, TurnDecision(is_complete=True, confidence=0.9)]
        )
        channel, backend, session, fw = _wired_channel(
            detector,
            [
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1"),
                VADEvent(type=VADEventType.SPEECH_START),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio2"),
            ],
            ["Combien de boards je vois,", "et lequel a le plus de cartes ?"],
            turn_incomplete_wait_ms=_WAIT_MS,
        )
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await detector.decides()
        await backend.simulate_audio(session, AudioFrame(data=b"\x01\x00"))  # speech starts
        detector.release.set()
        await _judged_incomplete(fw)
        await asyncio.sleep(_PAST_THE_WAIT)  # well past the wait: the user is still talking
        assert _routed_texts(fw) == []

        await backend.simulate_audio(session, AudioFrame(data=b"\x02\x00"))  # speech ends
        await _routed(fw, ["Combien de boards je vois, et lequel a le plus de cartes ?"])

    async def test_closing_the_channel_during_the_wait_routes_nothing(self):
        channel, backend, session, fw = _wired_channel(
            MockTurnDetector(decisions=[_INCOMPLETE]),
            [VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1")],
            ["Au revoir"],
            turn_incomplete_wait_ms=_WAIT_MS,
        )
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await _judged_incomplete(fw)
        await channel.close()
        await asyncio.sleep(_PAST_THE_WAIT)
        assert _routed_texts(fw) == []

    async def test_a_turn_judged_complete_while_the_user_speaks_again_is_held(self):
        # Smart Turn said 0.502 "complete", but the user had resumed 130 ms after
        # the VAD's end of speech: the two fragments are one question.
        detector = _SlowTurnDetector(
            decisions=[
                TurnDecision(is_complete=True, confidence=0.5),
                TurnDecision(is_complete=True, confidence=0.9),
            ]
        )
        channel, backend, session, fw = _wired_channel(
            detector,
            [
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1"),
                VADEvent(type=VADEventType.SPEECH_START),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio2"),
            ],
            ["J'aimerais bien voir le board", "et les cartes aussi"],
            turn_incomplete_wait_ms=_WAIT_MS,
        )
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await detector.decides()
        await backend.simulate_audio(session, AudioFrame(data=b"\x01\x00"))  # speech resumes
        detector.release.set()
        assert await asyncio.to_thread(detector.decided.wait, 2)
        await asyncio.sleep(_PAST_THE_WAIT)
        assert _routed_texts(fw) == []

        await backend.simulate_audio(session, AudioFrame(data=b"\x02\x00"))  # speech ends
        await _routed(fw, ["J'aimerais bien voir le board et les cartes aussi"])

    async def test_a_resumption_transcribed_after_the_wait_joins_the_turn(self):
        # The log of RMK-391: "Mm so um" was held, the user went on, and the
        # streaming STT's final for the rest came 0.4 s after the 1.5 s wait.
        # The wait routed "Mm so um" alone and the rest got a second answer.
        transcripts = ["Mm so um", "can you give me a poem about Quebec"]
        stt = _LateSTT(transcripts)
        channel, backend, session, fw = _wired_channel(
            MockTurnDetector(
                decisions=[_INCOMPLETE, TurnDecision(is_complete=True, confidence=0.9)]
            ),
            [
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1"),
                VADEvent(type=VADEventType.SPEECH_START),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio2"),
            ],
            transcripts,
            stt=stt,
            turn_incomplete_wait_ms=_WAIT_MS,
        )
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await _judged_incomplete(fw)
        await backend.simulate_audio(session, AudioFrame(data=b"\x01\x00"))  # speech resumes
        await backend.simulate_audio(session, AudioFrame(data=b"\x02\x00"))  # and ends
        await asyncio.sleep(_PAST_THE_WAIT)  # the wait is over, the transcript still on its way
        assert _routed_texts(fw) == []

        stt.release.set()
        await _routed(fw, ["Mm so um can you give me a poem about Quebec"])

    async def test_a_resumption_that_turns_out_empty_after_the_wait_releases_the_turn(self):
        transcripts = ["Combien de boards je vois,", ""]
        stt = _LateSTT(transcripts)
        channel, backend, session, fw = _wired_channel(
            MockTurnDetector(decisions=[_INCOMPLETE]),
            [
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1"),
                VADEvent(type=VADEventType.SPEECH_START),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio2"),
            ],
            transcripts,
            stt=stt,
            turn_incomplete_wait_ms=_WAIT_MS,
        )
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await _judged_incomplete(fw)
        await backend.simulate_audio(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio(session, AudioFrame(data=b"\x02\x00"))
        await asyncio.sleep(_PAST_THE_WAIT)
        assert _routed_texts(fw) == []

        stt.release.set()
        await _routed(fw, ["Combien de boards je vois,"])

    async def test_a_transcript_that_never_comes_does_not_hold_the_turn_forever(self, monkeypatch):
        # Lost, the transcript holds the turn this long past its wait.
        monkeypatch.setattr("roomkit.channels._voice_turn._TRANSCRIPT_WAIT_MAX_S", 0.15)
        channel, backend, session, fw = _wired_channel(
            MockTurnDetector(decisions=[_INCOMPLETE]),
            [
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1"),
                VADEvent(type=VADEventType.SPEECH_START),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio2"),
            ],
            ["Combien de boards je vois,"],
            stt=_SilentSTT(["Combien de boards je vois,"]),
            turn_incomplete_wait_ms=_WAIT_MS,
        )
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await _judged_incomplete(fw)
        await backend.simulate_audio(session, AudioFrame(data=b"\x01\x00"))
        await backend.simulate_audio(session, AudioFrame(data=b"\x02\x00"))
        await asyncio.sleep(_PAST_THE_WAIT)  # past the wait, within the transcript's hold
        assert _routed_texts(fw) == []

        await _routed(fw, ["Combien de boards je vois,"])
        await channel.close()

    async def test_a_turn_judged_complete_in_silence_is_routed_at_once(self):
        channel, backend, session, fw = _wired_channel(
            MockTurnDetector(decisions=[TurnDecision(is_complete=True, confidence=0.9)]),
            [VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1")],
            ["Combien de boards je vois ?"],
            turn_incomplete_wait_ms=5000,
        )
        await backend.simulate_audio(session, AudioFrame(data=b"\x00\x00"))
        await _routed(fw, ["Combien de boards je vois ?"])  # long before the wait's 5 s
