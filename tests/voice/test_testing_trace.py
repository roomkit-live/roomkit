"""VoiceTrace: the hook timeline that replaces sleeps in voice tests."""

from __future__ import annotations

import pytest

from roomkit import HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.events import TranscriptionEvent
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.testing import VOICE_TRIGGERS, ScenarioVoiceBackend, VoiceTrace, tone
from roomkit.voice.tts.mock import MockTTSProvider


def _turn(audio: bytes = b"speech\x00\x00") -> list[VADEvent | None]:
    return [
        VADEvent(type=VADEventType.SPEECH_START),
        VADEvent(type=VADEventType.SPEECH_END, audio_bytes=audio),
    ]


async def _kit(
    vad_events: list[VADEvent | None],
    *,
    transcripts: tuple[str, ...] = ("hello there",),
    triggers: tuple[HookTrigger, ...] = VOICE_TRIGGERS,
    with_agent: bool = True,
):
    stt = MockSTTProvider(transcripts=list(transcripts))
    tts = MockTTSProvider()
    backend = ScenarioVoiceBackend()
    kit = RoomKit(stt=stt, tts=tts, voice=backend)
    # Before the first session: ON_SESSION_STARTED fires at connect time.
    trace = VoiceTrace(kit, triggers=triggers)
    kit.register_channel(
        VoiceChannel(
            "voice-1",
            stt=stt,
            tts=tts,
            backend=backend,
            pipeline=AudioPipelineConfig(vad=MockVADProvider(events=vad_events)),
        )
    )
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    if with_agent:
        kit.register_channel(AIChannel("ai-1", provider=MockAIProvider(responses=["hi back"])))
        await kit.attach_channel(room.id, "ai-1")
    session = await kit.connect_voice(room.id, "user-1", "voice-1")
    return kit, backend, trace, session


class TestWaitFor:
    async def test_returns_the_entry_with_its_payload(self) -> None:
        kit, backend, trace, session = await _kit(_turn())
        await backend.play(session, tone(40), realtime=False)

        entry = await trace.wait_for(HookTrigger.ON_TRANSCRIPTION)

        assert isinstance(entry.payload, TranscriptionEvent)
        assert entry.payload.text == "hello there"
        assert entry.session_id == session.id
        assert entry.room_id == session.room_id
        await kit.close()

    async def test_a_turn_reads_in_order_off_the_timeline(self) -> None:
        kit, backend, trace, session = await _kit(_turn())
        await backend.play(session, tone(40), realtime=False)

        await trace.wait_for(HookTrigger.AFTER_TTS)

        seq = trace.sequence()
        assert seq[0] is HookTrigger.ON_SESSION_STARTED
        assert (
            seq.index(HookTrigger.ON_SPEECH_START)
            < seq.index(HookTrigger.ON_SPEECH_END)
            < seq.index(HookTrigger.ON_TRANSCRIPTION)
            < seq.index(HookTrigger.BEFORE_TTS)
            < seq.index(HookTrigger.AFTER_TTS)
        )
        started = trace.first(HookTrigger.ON_SESSION_STARTED)
        assert started is not None and started.session_id == session.id
        speech_end = trace.first(HookTrigger.ON_SPEECH_END)
        transcription = trace.first(HookTrigger.ON_TRANSCRIPTION)
        assert speech_end is not None and transcription is not None
        assert speech_end.payload is session
        assert trace.elapsed_ms(speech_end, transcription) >= 0
        after_tts = trace.last(HookTrigger.AFTER_TTS)
        assert after_tts is not None and after_tts.payload == "hi back"
        await kit.close()

    async def test_timeout_names_what_did_fire(self) -> None:
        kit, backend, trace, session = await _kit([VADEvent(type=VADEventType.SPEECH_START)])
        await backend.play(session, tone(20), realtime=False)
        await trace.wait_for(HookTrigger.ON_SPEECH_START)

        with pytest.raises(TimeoutError, match="on_transcription not seen.*on_speech_start"):
            await trace.wait_for(HookTrigger.ON_TRANSCRIPTION, timeout=0.05)
        await kit.close()

    async def test_after_skips_an_earlier_firing(self) -> None:
        kit, backend, trace, session = await _kit(
            _turn() + _turn(), transcripts=("one", "two"), with_agent=False
        )
        await backend.play(session, tone(40), realtime=False)
        first = await trace.wait_for(HookTrigger.ON_TRANSCRIPTION)
        await backend.play(session, tone(40), realtime=False)

        second = await trace.wait_for(HookTrigger.ON_TRANSCRIPTION, after=first)

        assert first.payload.text == "one"
        assert second.payload.text == "two"
        assert len(trace.entries(HookTrigger.ON_TRANSCRIPTION)) == 2
        assert trace.entries(HookTrigger.ON_TRANSCRIPTION, after=second) == []
        await kit.close()

    async def test_entries_filter_by_session(self) -> None:
        kit, backend, trace, session = await _kit(_turn(), with_agent=False)
        other = await kit.connect_voice(session.room_id, "user-2", "voice-1")
        await backend.play(other, tone(40), realtime=False)

        entry = await trace.wait_for(HookTrigger.ON_TRANSCRIPTION, session_id=other.id)

        assert entry.session_id == other.id
        assert trace.entries(HookTrigger.ON_SPEECH_START, session_id=session.id) == []
        assert len(trace.entries(session_id=other.id)) >= 3
        await kit.close()

    async def test_clear_forgets_the_timeline(self) -> None:
        kit, backend, trace, session = await _kit(_turn(), with_agent=False)
        await backend.play(session, tone(40), realtime=False)
        await trace.wait_for(HookTrigger.ON_TRANSCRIPTION)

        trace.clear()

        assert trace.sequence() == []
        assert trace.first(HookTrigger.ON_TRANSCRIPTION) is None
        await kit.close()


class TestTriggers:
    async def test_a_custom_trigger_set_observes_only_those(self) -> None:
        kit, backend, trace, session = await _kit(
            _turn(), triggers=(HookTrigger.ON_SPEECH_START,), with_agent=False
        )
        await backend.play(session, tone(40), realtime=False)

        await trace.wait_for(HookTrigger.ON_SPEECH_START)

        assert trace.triggers == (HookTrigger.ON_SPEECH_START,)
        with pytest.raises(TimeoutError):
            await trace.wait_for(HookTrigger.ON_TRANSCRIPTION, timeout=0.05)
        await kit.close()

    async def test_any_hook_trigger_can_be_traced(self) -> None:
        """A non-voice trigger joins the timeline: the transcription's
        broadcast is what ends the turn when no agent answers."""
        kit, backend, trace, session = await _kit(
            _turn(), triggers=(*VOICE_TRIGGERS, HookTrigger.AFTER_BROADCAST), with_agent=False
        )
        await backend.play(session, tone(40), realtime=False)

        broadcast = await trace.wait_for(HookTrigger.AFTER_BROADCAST)

        assert broadcast.payload.content.body == "hello there"
        seq = trace.sequence()
        assert seq.index(HookTrigger.ON_TRANSCRIPTION) < seq.index(HookTrigger.AFTER_BROADCAST)
        await kit.close()
