"""StripTechnicalText: a tool call, a note to self or a separator the model wrote
into a spoken reply is never read aloud (RMK-624)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import pytest

from roomkit.voice import StripTechnicalText, TTSFilterChain
from roomkit.voice.tts.filters import StripEmoji
from tests.voice.streamed_replies import ask, said, voice_rooms

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
        (
            'Searching.{agent: "meteo", task: "rain"}{"status": "ok"}Back soon.',
            "Searching. Back soon.",
        ),
        ('I check{"name": "x"}, then I come back.', "I check, then I come back."),
    ],
    ids=[
        "tool-call",
        "braces-in-strings",
        "spaced-key",
        "note",
        "nested-note",
        "dashes",
        "rule",
        "bare-key-glued",
        "glued-before-comma",
    ],
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
    kit, _, backend, tts, [session] = await voice_rooms(tokens, tts_filter=StripTechnicalText())

    await ask(backend, session)
    [spoken] = await said(tts, 1)

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
    kit, voice, backend, tts, [a, b] = await voice_rooms(
        room_a, room_b, gap=0.05, tts_filter=StripTechnicalText()
    )

    await ask(backend, a)
    await asyncio.sleep(0.03)
    await ask(backend, b)
    await asyncio.sleep(0.05)
    await voice.say(b, "Welcome to room B.")
    spoken = await said(tts, 2)

    assert sorted(spoken) == [
        "Room A looks. A done.",
        "Room B hello. Room B keeps talking. Room B done.",
    ]
    await kit.close()


async def test_say_keeps_it_out_of_the_voice_too() -> None:
    kit, voice, backend, tts, [session] = await voice_rooms(
        ["unused"], tts_filter=StripTechnicalText()
    )

    await voice.say(session, f"One moment. {TOOL_CALL} Done.")

    assert tts.calls[-1]["text"] == "One moment. Done."
    await kit.close()
