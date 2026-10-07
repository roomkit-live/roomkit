"""An answer cut off by a barge-in (RMK-563, RMK-533, RFC §6.4 and §12.3.13): the
voice channel's record names the answer, the AI context marks it as cut, and the
speak policy reads it and judges whether the agent goes on."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from roomkit import ClassifierSpeakPolicy, CutReply, MockClassifier, RoomKit, VoiceChannel
from roomkit.channels._ai_cuts import CUT_MARK, cut_answer_ids, cut_records, cut_reply
from roomkit.channels._ai_speaking import _speak_turn
from roomkit.channels.ai import AIChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType, Visibility
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.speaking.base import SpeakTurn
from roomkit.speaking.classifier import RESUME_NOTE
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceCapability
from roomkit.voice.interruption import InterruptionConfig
from roomkit.voice.tts.mock import MockTTSProvider
from tests.tool_loop_modes import respond

T0 = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
FORECAST = "Tomorrow in Quebec City, 8 degrees and cloudy, with rain in the evening."

_AI = ChannelBinding(
    channel_id="ai1",
    room_id="r1",
    channel_type=ChannelType.AI,
    category=ChannelCategory.INTELLIGENCE,
)
_SMS = ChannelBinding(
    channel_id="sms1",
    room_id="r1",
    channel_type=ChannelType.SMS,
    category=ChannelCategory.TRANSPORT,
)
_VOICE = ChannelBinding(
    channel_id="voice1",
    room_id="r1",
    channel_type=ChannelType.VOICE,
    category=ChannelCategory.TRANSPORT,
)


def _event(body: str, *, channel: str = "sms1", at: int = 0, **kwargs: Any) -> RoomEvent:
    kind = ChannelType.AI if channel == "ai1" else ChannelType.SMS
    return RoomEvent(
        room_id="r1",
        source=EventSource(
            channel_id=channel,
            channel_type=kind,
            participant_id=None if channel == "ai1" else "p1",
        ),
        content=TextContent(body=body),
        created_at=T0 + timedelta(seconds=at),
        **kwargs,
    )


def _cut(
    answer_responds_to: object,
    *,
    at: int,
    played_ms: object = 1200,
    source: EventSource | None = None,
    visibility: str = Visibility.INTERNAL,
) -> RoomEvent:
    """The record a voice channel writes when the agent is cut (RFC §12.3.13)."""
    return RoomEvent(
        room_id="r1",
        source=source or EventSource(channel_id="voice1", channel_type=ChannelType.VOICE),
        content=TextContent(body=FORECAST),
        visibility=visibility,
        metadata={
            "interrupted": True,
            "played_ms": played_ms,
            "answer_channel_id": "ai1",
            "answer_responds_to": answer_responds_to,
        },
        created_at=T0 + timedelta(seconds=at),
    )


def _context(*events: RoomEvent) -> RoomContext:
    return RoomContext(
        room=Room(id="r1"), bindings=[_AI, _SMS, _VOICE], recent_events=list(events)
    )


# -- the voice channel's record ----------------------------------------------------


async def _speaking(playback: TTSPlaybackState) -> tuple[RoomKit, VoiceChannel, Any]:
    backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
    channel = VoiceChannel("voice-1", backend=backend, interruption=InterruptionConfig())
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice-1")
    session = await kit.connect_voice("r1", "user-1", "voice-1")
    playback.session_id = session.id
    channel._playing_sessions[session.id] = playback  # noqa: SLF001
    return kit, channel, session


async def test_the_cut_record_names_the_answer_and_not_the_listener() -> None:
    kit, channel, session = await _speaking(
        TTSPlaybackState(
            session_id="", text=FORECAST, answer_channel_id="ai1", answer_responds_to="q1"
        )
    )
    await channel.interrupt(session, reason="barge_in")

    [record] = [e for e in await kit.get_timeline("r1") if e.metadata.get("interrupted")]
    assert record.source.participant_id is None
    assert record.metadata["answer_channel_id"] == "ai1"
    assert record.metadata["answer_responds_to"] == "q1"
    context = await kit._build_context("r1", recent_limit=10)  # noqa: SLF001
    assert cut_records(context, "ai1") == {"q1": record}


async def test_a_playback_that_answers_nothing_names_no_answer() -> None:
    kit, channel, session = await _speaking(TTSPlaybackState(session_id="", text="Welcome!"))
    await channel.interrupt(session, reason="barge_in")

    [record] = [e for e in await kit.get_timeline("r1") if e.metadata.get("interrupted")]
    assert "answer_channel_id" not in record.metadata


async def test_a_delivered_answer_plays_under_its_name() -> None:
    """The answer's channel and responds_to reach the playback the cut is taken from."""
    backend = MockVoiceBackend()
    channel = VoiceChannel("voice-1", tts=MockTTSProvider(), backend=backend)
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice-1")
    session = await kit.connect_voice("r1", "user-1", "voice-1")
    seen: list[TTSPlaybackState] = []
    original = channel._play_tts  # noqa: SLF001

    async def watch(*args: Any, **kwargs: Any) -> None:
        task = asyncio.create_task(original(*args, **kwargs))
        await asyncio.sleep(0)
        playback = channel._playing_sessions.get(session.id)  # noqa: SLF001
        if playback is not None:
            seen.append(playback)
        await task

    channel._play_tts = watch  # type: ignore[method-assign]  # noqa: SLF001
    answer = RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="ai1", channel_type=ChannelType.AI),
        content=TextContent(body=FORECAST),
        responds_to="q1",
    )
    binding = ChannelBinding(
        channel_id="voice-1",
        room_id="r1",
        channel_type=ChannelType.VOICE,
        category=ChannelCategory.TRANSPORT,
    )
    await channel._deliver_voice(answer, binding, await kit._build_context("r1"))  # noqa: SLF001

    assert [(p.answer_channel_id, p.answer_responds_to) for p in seen] == [("ai1", "q1")]


# -- finding the cut ---------------------------------------------------------------


def test_a_cut_is_the_latest_answer_its_record_names() -> None:
    question = _event("What's the weather tomorrow?", at=0)
    answer = _event(FORECAST, channel="ai1", at=1, responds_to=question.id)
    record = _cut(question.id, at=3)
    events = [question, answer, record]

    records = cut_records(_context(*events), "ai1")
    assert records == {question.id: record}
    assert cut_records(_context(*events), "ai2") == {}
    assert cut_answer_ids(events, records, "ai1") == {answer.id}
    assert cut_reply([question, answer], records, "ai1") == CutReply(
        FORECAST, 1200, record.created_at
    )


def test_an_answer_given_since_leaves_no_cut() -> None:
    question = _event("What's the weather tomorrow?", at=0)
    answer = _event(FORECAST, channel="ai1", at=1, responds_to=question.id)
    later = _event("Anything else?", at=5)
    again = _event("No, that's all.", channel="ai1", at=6, responds_to=later.id)
    context = _context(question, answer, _cut(question.id, at=3), later, again)
    assert cut_reply([question, answer, later, again], cut_records(context, "ai1"), "ai1") is None


@pytest.mark.parametrize(
    "record",
    [
        _cut("q1", at=3, source=EventSource(channel_id="sms1", channel_type=ChannelType.SMS)),
        _cut("q1", at=3, visibility=Visibility.ALL),
        _cut(
            "q1",
            at=3,
            source=EventSource(
                channel_id="voice1", channel_type=ChannelType.VOICE, participant_id="p1"
            ),
        ),
        _cut("q1", at=3, source=EventSource(channel_id="voice9", channel_type=ChannelType.VOICE)),
        _cut({"not": "hashable"}, at=3),
    ],
    ids=["from-a-sender", "not-internal", "with-a-participant", "unbound-voice", "bad-answer-id"],
)
def test_only_a_voice_channels_record_counts(record: RoomEvent) -> None:
    """Metadata is anyone's to write: a record a voice channel of the room did not
    write, or a malformed one, names no cut."""
    assert cut_records(_context(record), "ai1") == {}


@pytest.mark.parametrize("played_ms", ["1200", -5, True, None])
def test_a_malformed_played_time_reads_as_none(played_ms: object) -> None:
    question = _event("What's the weather tomorrow?", at=0)
    answer = _event(FORECAST, channel="ai1", at=1, responds_to=question.id)
    records = cut_records(_context(_cut(question.id, at=3, played_ms=played_ms)), "ai1")
    cut = cut_reply([question, answer], records, "ai1")
    assert cut is not None and cut.played_ms == 0


# -- on the AI channel -------------------------------------------------------------


async def test_the_context_marks_the_cut_answer() -> None:
    provider = MockAIProvider(["So, as I was saying: rain in the evening."])
    channel = AIChannel("ai1", provider=provider)
    question = _event("What's the weather tomorrow?", at=0)
    answer = _event(FORECAST, channel="ai1", at=1, responds_to=question.id)
    thanks = _event("Ok great, thanks.", at=4)

    await respond(
        channel, thanks, _AI, _context(question, answer, _cut(question.id, at=3), thanks)
    )

    assistant = [m for m in provider.calls[0].messages if m.role == "assistant"]
    assert assistant[-1].content == f"{FORECAST}\n{CUT_MARK}"


async def test_an_answer_heard_whole_is_not_marked() -> None:
    provider = MockAIProvider(["Sure."])
    channel = AIChannel("ai1", provider=provider)
    question = _event("What's the weather tomorrow?", at=0)
    answer = _event(FORECAST, channel="ai1", at=1, responds_to=question.id)
    thanks = _event("Ok great, thanks.", at=4)

    await respond(channel, thanks, _AI, _context(question, answer, thanks))

    assert all(CUT_MARK not in str(m.content) for m in provider.calls[0].messages)


def test_the_speak_turn_carries_the_cut() -> None:
    question = _event("What's the weather tomorrow?", at=0)
    answer = _event(FORECAST, channel="ai1", at=1, responds_to=question.id)
    record = _cut(question.id, at=3)
    thanks = _event("Ok great, thanks.", at=4)

    turn = _speak_turn(thanks, _context(question, answer, record, thanks), "ai1")

    assert turn.cut == CutReply(FORECAST, 1200, record.created_at)


# -- the classifier policy -----------------------------------------------------------


def _cut_turn(text: str = "Ok great, thanks.") -> SpeakTurn:
    question = _event("What's the weather tomorrow?", at=0)
    answer = _event(FORECAST, channel="ai1", at=1, responds_to=question.id)
    event = _event(text, at=4)
    return SpeakTurn(
        event=event,
        recent=(question, answer),
        people=("Sylvain", "Paul"),
        channel_id="ai1",
        speakers={question.id: "Sylvain", event.id: "Sylvain"},
        cut=CutReply(FORECAST, 1200, T0 + timedelta(seconds=3)),
    )


async def test_free_to_go_on_the_agent_resumes() -> None:
    classifier = MockClassifier({"resume": 0.9})
    decision = await ClassifierSpeakPolicy(classifier, agent_name="Nova").decide(_cut_turn())

    assert (decision.mode, decision.reason) == ("speak", "resume after cut")
    assert decision.notes[0] == RESUME_NOTE.format(seconds=1.2)
    state, questions = classifier.calls[0]
    assert "resume" in questions
    assert state["assistant_cut_off"] == {"was_saying": FORECAST, "heard_seconds": 1.2}  # type: ignore[index]


async def test_wanting_the_turn_the_turn_decides() -> None:
    classifier = MockClassifier({"resume": 0.2})
    decision = await ClassifierSpeakPolicy(classifier, agent_name="Nova").decide(
        _cut_turn("Wait, and for Montreal?")
    )
    assert (decision.mode, decision.reason) == ("silent", "not addressed")


async def test_asked_for_quiet_it_does_not_resume() -> None:
    classifier = MockClassifier({"resume": 0.9, "hush": 0.9})
    decision = await ClassifierSpeakPolicy(classifier, agent_name="Nova").decide(_cut_turn())
    assert (decision.mode, decision.reason) == ("silent", "asked to keep quiet")


async def test_without_a_cut_no_resume_is_asked() -> None:
    classifier = MockClassifier({"directness": 3.0})
    turn = SpeakTurn(event=_event("Nova?"), people=("Sylvain",), channel_id="ai1")
    await ClassifierSpeakPolicy(classifier, agent_name="Nova").decide(turn)
    _, questions = classifier.calls[0]
    assert "resume" not in questions
