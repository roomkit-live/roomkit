"""StripTechnicalText: a tool call, a note to self or a separator the model wrote
into a spoken reply is never read aloud (RMK-624)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable

import pytest

from roomkit import RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext, AIProvider, AIResponse
from roomkit.voice import StripTechnicalText, TTSFilterChain
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.filters import StripEmoji
from tests.test_voice_streaming_ai_tts import _speech_events
from tests.voice.test_voice_tts_filter import _StreamingMockTTS

TOOL_CALL = '{"name": "delegate", "arguments": {"task": "Weather. Today"}}'


def _whole(text: str) -> str:
    return " ".join(StripTechnicalText()(text).split())


def _chunks(size: int) -> Callable[[str], str]:
    def run(text: str) -> str:
        f = StripTechnicalText()
        out = "".join(f.feed(text[i : i + size]) for i in range(0, len(text), size))
        return " ".join((out + f.flush()).split())

    return run


MODES = pytest.mark.parametrize(
    "spoken", [_whole, _chunks(1), _chunks(3), _chunks(10_000)], ids=["call", "1", "3", "all"]
)


@MODES
@pytest.mark.parametrize(
    ("text", "heard"),
    [
        (f"Let me check. {TOOL_CALL} Done.", "Let me check. Done."),
        ('Result: {"text": "a } b", "inner": {"q": "\\"}"}} ok.', "Result: ok."),
        ('Here { "a": 1 } it is.', "Here it is."),
        ("(Note: do not say this.) Hello.", "Hello."),
        ("Hello (nb : see (this) later) !", "Hello !"),
        ("--- Next part.", "Next part."),
        ("First.\n\n-----\n\nSecond.", "First. Second."),
    ],
    ids=["tool-call", "braces-in-strings", "spaced-key", "note", "nested-note", "dashes", "rule"],
)
def test_technical_text_is_not_heard(spoken: Callable[[str], str], text: str, heard: str) -> None:
    assert spoken(text) == heard


@MODES
@pytest.mark.parametrize(
    "text",
    [
        "It takes (about ten minutes).",
        "(no problem) (n) (nota bene) {x} {} done.",
        "A well-known fix -- really.",
        "Prices go from 10-20 dollars.",
    ],
)
def test_ordinary_text_passes(spoken: Callable[[str], str], text: str) -> None:
    assert spoken(text) == text


@MODES
@pytest.mark.parametrize(
    ("text", "heard"),
    [
        ('I will call it now. {"name": "delegate", "arguments": {', "I will call it now."),
        ("All set (Note: remember to", "All set"),
        ("Ends with a brace {", "Ends with a brace {"),
        ("Ends on a dash -", "Ends on a dash -"),
        ("Ends on a rule ---", "Ends on a rule"),
    ],
)
def test_the_end_of_the_stream_decides_what_is_still_open(
    spoken: Callable[[str], str], text: str, heard: str
) -> None:
    assert spoken(text) == heard


@MODES
def test_a_removal_glued_to_the_text_before_keeps_the_space_after(
    spoken: Callable[[str], str],
) -> None:
    assert spoken(f"Let me check.{TOOL_CALL} The shop opens.") == "Let me check. The shop opens."


def test_a_whole_text_filtered_mid_stream_leaves_the_stream_as_it_was() -> None:
    """Review of RMK-624: ``__call__`` reset the instance a reply was streaming
    through, which then read the end of an open object aloud."""
    f = StripTechnicalText()
    out = f.feed('Let me look. {"name": "lookup", ')

    assert f("Welcome.") == "Welcome."
    out += f.feed('"arguments": {"q": "x"}} Done.') + f.flush()

    assert out == "Let me look. Done."


@pytest.mark.parametrize("size", [1, 3, 10_000])
def test_the_spacing_a_removal_leaves_is_not_streamed(size: int) -> None:
    f = StripTechnicalText()
    text = f"Let me look. {TOOL_CALL}  ---\n The shop opens at ten."
    out = "".join(f.feed(text[i : i + size]) for i in range(0, len(text), size))

    assert out + f.flush() == "Let me look. The shop opens at ten."


def test_a_reused_filter_starts_each_reply_afresh() -> None:
    f = StripTechnicalText()
    f.feed('Cut {"name": "x", ')  # a reply cut mid-object
    f.reset()

    assert f.feed("Next reply.") + f.flush() == "Next reply."


def test_what_it_removes_is_logged_without_its_text(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("roomkit.telemetry.redaction._content_logging", False)
    with caplog.at_level(logging.DEBUG, logger="roomkit.voice.tts.filters"):
        _whole(f"Done. {TOOL_CALL} (Note: secret plan.) ---")

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [
        f"A JSON object ({len(TOOL_CALL)} chars) was kept out of speech",
        "A note (20 chars) was kept out of speech",
    ]
    assert "delegate" not in caplog.text and "secret" not in caplog.text


def test_it_chains_with_the_other_filters() -> None:
    chain = TTSFilterChain(StripTechnicalText(), StripEmoji())

    assert chain(f"Done \U0001f60a {TOOL_CALL} bye.") == "Done bye."


class _SlowStream(AIProvider):
    """Streams its tokens with a pause after each, so two replies overlap."""

    def __init__(self, tokens: list[str], gap: float = 0.0) -> None:
        self._tokens, self._gap = tokens, gap

    @property
    def model_name(self) -> str:
        return "slow"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        return AIResponse(content="".join(self._tokens))

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        for token in self._tokens:
            yield token
            await asyncio.sleep(self._gap)


async def _voice_rooms(
    *replies: list[str], gap: float = 0.0
) -> tuple[RoomKit, VoiceChannel, MockVoiceBackend, _StreamingMockTTS, list[VoiceSession]]:
    """One voice channel filtering with StripTechnicalText, a room per reply,
    each with an agent that streams that reply."""
    tts, backend = _StreamingMockTTS(), MockVoiceBackend()
    stt = MockSTTProvider(transcripts=["A question."] * len(replies))
    vad = MockVADProvider(events=_speech_events() * len(replies))
    voice = VoiceChannel(
        "voice-1",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=AudioPipelineConfig(vad=vad),
        tts_filter=StripTechnicalText(),
    )
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(voice)
    sessions = []
    for n, tokens in enumerate(replies):
        kit.register_channel(AIChannel(f"ai-{n}", provider=_SlowStream(tokens, gap)))
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice-1")
        await kit.attach_channel(room.id, f"ai-{n}")
        sessions.append(await kit.connect_voice(room.id, f"user-{n}", "voice-1"))
    return kit, voice, backend, tts, sessions


async def _ask(backend: MockVoiceBackend, session: VoiceSession) -> None:
    for data in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        await backend.simulate_audio_received(session, AudioFrame(data=data))


async def _spoken(tts: _StreamingMockTTS, replies: int) -> list[str]:
    for _ in range(100):
        if len(tts.stream_input_texts) >= replies:
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.2)
    return [" ".join(chunks) for chunks in tts.stream_input_texts]


async def test_a_streamed_reply_is_spoken_without_it_and_stored_with_it() -> None:
    """Through the voice channel: the object is split over tokens and holds a
    full stop, which a per-sentence hook would have cut in two."""
    tokens = [
        "I'll look. ",
        '{"name": "delegate", ',
        '"arguments": {"task": "Rain. ',
        'Today"}}',
        " Done.",
    ]
    kit, _, backend, tts, [session] = await _voice_rooms(tokens)

    await _ask(backend, session)
    [spoken] = await _spoken(tts, 1)

    assert "delegate" not in spoken and "{" not in spoken
    assert "I'll look." in spoken and "Done." in spoken
    final = [text for _, text, role in backend.sent_transcriptions if role == "assistant"]
    assert final and "delegate" not in final[-1]
    events = await kit.store.list_events(session.room_id)
    stored = [e.content.body for e in events if e.source.channel_id == "ai-0"]
    assert any("delegate" in body for body in stored)  # the slip stays visible in the room
    await kit.close()


async def test_replies_streamed_at_once_in_two_rooms_keep_their_own_filtering() -> None:
    """Review of RMK-624: one filter instance shared by two streams swallowed
    the other room's sentences while an object was open, and a ``say()`` in the
    middle reset it."""
    room_a = ["Room A looks. ", '{"name": "lookup", ', '"arguments": {"q": "x"}}', " A done."]
    room_b = ["Room B hello. ", "Room B keeps talking. ", "Room B done."]
    kit, voice, backend, tts, [a, b] = await _voice_rooms(room_a, room_b, gap=0.05)

    await _ask(backend, a)
    await asyncio.sleep(0.03)
    await _ask(backend, b)
    await asyncio.sleep(0.05)
    await voice.say(b, "Welcome to room B.")
    spoken = await _spoken(tts, 2)

    assert sorted(spoken) == [
        "Room A looks. A done.",
        "Room B hello. Room B keeps talking. Room B done.",
    ]
    await kit.close()


async def test_say_keeps_it_out_of_the_voice_too() -> None:
    kit, voice, backend, tts, [session] = await _voice_rooms(["unused"])

    await voice.say(session, f"One moment. {TOOL_CALL} Done.")

    assert tts.calls[-1]["text"] == "One moment. Done."
    await kit.close()
