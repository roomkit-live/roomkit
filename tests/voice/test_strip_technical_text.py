"""StripTechnicalText: a tool call, a note to self or a separator the model wrote
into a spoken reply is never read aloud (RMK-624)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import pytest

from roomkit import RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.voice import StripTechnicalText, TTSFilterChain
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.filters import StripEmoji
from tests.test_voice_streaming_ai_tts import _speech_events, _StreamingAIProvider
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
        ("Bonjour (nb : voir (ceci) plus tard) !", "Bonjour !"),
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


def test_what_it_removes_is_logged_without_its_text(caplog: pytest.LogCaptureFixture) -> None:
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
    tts = _StreamingMockTTS()
    stt = MockSTTProvider(transcripts=["What's the weather?"])
    backend = MockVoiceBackend()
    voice = VoiceChannel(
        "voice-1",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=AudioPipelineConfig(vad=MockVADProvider(events=_speech_events())),
        tts_filter=StripTechnicalText(),
    )
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(voice)
    kit.register_channel(AIChannel("ai-1", provider=_StreamingAIProvider(tokens)))
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    await kit.attach_channel(room.id, "ai-1")
    session = await kit.connect_voice(room.id, "user-1", "voice-1")

    for data in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        await backend.simulate_audio_received(session, AudioFrame(data=data))
    for _ in range(50):
        if tts.stream_input_texts:
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.2)

    spoken = " ".join(" ".join(call) for call in tts.stream_input_texts)
    assert "delegate" not in spoken and "{" not in spoken
    assert "I'll look." in spoken and "Done." in spoken
    final = [text for _, text, role in backend.sent_transcriptions if role == "assistant"]
    assert final and "delegate" not in final[-1]
    stored = [
        e.content.body
        for e in await kit.store.list_events(room.id)
        if e.source.channel_id == "ai-1"
    ]
    assert any("delegate" in body for body in stored)  # the slip stays visible in the room
    await kit.close()
