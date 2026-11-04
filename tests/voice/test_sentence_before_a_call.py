"""The text before a tool call is said before the call runs, and never runs on
into the next round's text (RMK-695, RFC §12.2 step 12s.a)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import AIChannel, RoomKit, VoiceChannel
from roomkit.channels._voice_tts import _ResponseText
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType
from roomkit.models.event import TextContent
from roomkit.models.streaming import SegmentBreakMarker, ThinkingDeltaMarker, UsageMarker
from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    AITool,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.tts.filters import StripInternalTags, TTSStreamFilter, filtered_stream
from roomkit.voice.tts.sentence_splitter import split_sentences

ANNOUNCED = "Je relance le dé."
RESULT = "Le d20 donne 9."


async def _items(*items: Any) -> AsyncIterator[Any]:
    for item in items:
        yield item


class _HeardTTS(TTSProvider):
    """Keeps each sentence as the session's synthesis reads it."""

    def __init__(self) -> None:
        self.heard: list[str] = []

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize(self, text: str, *, voice: str | None = None) -> object:
        raise NotImplementedError

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        async for sentence in text_stream:
            self.heard.append(sentence)
            yield AudioChunk(data=b"\x00\x00", sample_rate=16000)


class _RollsAfterSaying(AIProvider):
    """Says *before* (a sentence per item), calls ``roll_dice``, then says *after*."""

    def __init__(self, before: list[str], after: str) -> None:
        self._before = before
        self._after = after
        self.rounds = 0

    @property
    def model_name(self) -> str:
        return "rolls-after-saying"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        return AIResponse(content="unused")

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        self.rounds += 1
        if self.rounds > 1:
            yield StreamTextDelta(text=self._after)
            yield StreamDone(finish_reason="stop")
            return
        for text in self._before:
            yield StreamTextDelta(text=text)
        yield StreamToolCall(id="t1", name="roll_dice")
        yield StreamDone(finish_reason="tool_calls")


class _Dice:
    """``roll_dice``: notes what the voice had said when it ran."""

    def __init__(self, tts: _HeardTTS) -> None:
        self._tts = tts
        self.heard_when_rolled: list[str] | None = None

    async def roll(self, name: str, arguments: dict[str, Any]) -> str:
        for _ in range(50):
            if self._tts.heard:
                break
            await asyncio.sleep(0.01)
        self.heard_when_rolled = list(self._tts.heard)
        return "9"


async def _voice_turn(
    before: list[str], after: str = RESULT, **voice_options: Any
) -> tuple[RoomKit, MockVoiceBackend, _HeardTTS, _Dice, _RollsAfterSaying, str]:
    backend, tts = MockVoiceBackend(), _HeardTTS()
    model, dice = _RollsAfterSaying(before, after), _Dice(tts)
    kit = RoomKit(voice=backend)
    kit.register_channel(VoiceChannel("voice-1", tts=tts, backend=backend, **voice_options))
    roll = AITool(name="roll_dice", description="Roll a die.", parameters={"type": "object"})
    kit.register_channel(AIChannel("ai-0", provider=model, tools=[roll], tool_handler=dice.roll))
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    await kit.attach_channel(room.id, "ai-0")
    await kit.join(room.id, "voice-1", participant_id="user-0")
    asked = InboundMessage(
        channel_id="voice-1", sender_id="user-0", content=TextContent(body="Roll for me.")
    )
    await asyncio.wait_for(kit.process_inbound(asked, room_id=room.id), 5)
    async with asyncio.timeout(5):  # the reply said to its end: its transcript is sent
        while not any(role == "assistant" for _, _, role in backend.sent_transcriptions):
            await asyncio.sleep(0.01)
    return kit, backend, tts, dice, model, room.id


async def test_a_segment_break_ends_the_sentence_however_short() -> None:
    tokens = _items("Let me check.", SegmentBreakMarker(), "It is 9.")

    assert [s async for s in split_sentences(tokens, min_chunk_chars=20)] == [
        "Let me check.",
        "It is 9.",
    ]


@pytest.mark.parametrize(
    "told", [UsageMarker(usage={"input_tokens": 1}), ThinkingDeltaMarker("x")]
)
async def test_a_marker_that_says_nothing_of_the_text_ends_no_sentence(told: Any) -> None:
    text = _ResponseText()

    read = [item async for item in text.read(_items("Let me ", told, "check."))]

    assert read == ["Let me ", "check."]


async def test_a_break_with_nothing_held_says_nothing() -> None:
    tokens = _items(SegmentBreakMarker(), "The invoice has three parts. ", SegmentBreakMarker())

    assert [s async for s in split_sentences(tokens)] == ["The invoice has three parts."]


class _HoldsAll(TTSStreamFilter):
    """Holds every chunk until it is flushed: a filter written before ``end_segment``."""

    def __init__(self) -> None:
        self._held = ""

    def feed(self, chunk: str) -> str:
        self._held += chunk
        return ""

    def flush(self) -> str:
        held, self._held = self._held, ""
        return held

    def reset(self) -> None:
        self._held = ""


class _HoldsAllUntilASegmentEnds(_HoldsAll):
    def end_segment(self) -> str:
        return self.flush()


async def test_a_streaming_filter_lets_out_what_it_decided_at_a_break() -> None:
    marker = SegmentBreakMarker()
    chunks = _items("Let me ", "check.", marker, "Nine.")

    out = [c async for c in filtered_stream(chunks, _HoldsAllUntilASegmentEnds())]

    assert out == ["Let me check.", marker, "Nine."]


async def test_a_filter_that_lets_nothing_out_at_a_break_keeps_it_for_the_end() -> None:
    """``flush`` is the end of the stream's: a break never calls it."""
    marker = SegmentBreakMarker()
    chunks = _items("Let me ", "check.", marker, "Nine.")

    out = [c async for c in filtered_stream(chunks, _HoldsAll())]

    assert out == [marker, "Let me check.Nine."]


async def test_a_span_open_at_a_break_stays_open_across_it() -> None:
    marker = SegmentBreakMarker()
    chunks = _items("Sure. [internal]I need ", marker, "the dice.[/internal] It is 9.")

    out = [c async for c in filtered_stream(chunks, StripInternalTags())]

    assert out == ["Sure. ", marker, " It is 9."]


async def test_the_sentence_before_a_call_is_said_while_the_tool_runs() -> None:
    kit, backend, tts, dice, _, _ = await _voice_turn([ANNOUNCED])

    assert dice.heard_when_rolled == [ANNOUNCED]
    assert tts.heard == [ANNOUNCED, RESULT]  # not "Je relance le dé.Le d20 donne 9."
    final = [text for _, text, role in backend.sent_transcriptions if role == "assistant"]
    assert final[-1] == f"{ANNOUNCED}\n\n{RESULT}"  # its segments apart, as ON_AI_RESPONSE
    await kit.close()


async def test_a_span_the_filter_removes_is_not_spoken_across_the_call() -> None:
    """An ``[internal]`` block opened before the call and closed after it."""
    kit, backend, tts, _, _, _ = await _voice_turn(
        ["Sure. [internal]I need the dice "],
        after="first.[/internal] It is 9.",
        tts_filter=StripInternalTags(),
    )

    assert tts.heard == ["Sure.", "It is 9."]
    final = [text for _, text, role in backend.sent_transcriptions if role == "assistant"]
    assert final[-1] == "Sure. It is 9."
    await kit.close()


async def test_a_reply_of_exactly_its_budget_keeps_its_segments_apart() -> None:
    """Not cut, the reply was said as read: its transcript is the read text's."""
    kit, backend, tts, dice, _, _ = await _voice_turn([ANNOUNCED], max_sentences=2)

    assert tts.heard == [ANNOUNCED, RESULT]
    assert dice.heard_when_rolled == [ANNOUNCED]
    final = [text for _, text, role in backend.sent_transcriptions if role == "assistant"]
    assert final[-1] == f"{ANNOUNCED}\n\n{RESULT}"
    await kit.close()


async def test_a_call_after_the_budgets_last_sentence_never_starts() -> None:
    """RFC §12.2 step 12s.e: a reply of exactly its budget that goes on with a
    call stops there; the sentence the call's start let out is the budget's last."""
    kit, _, tts, dice, model, room_id = await _voice_turn(
        ["The invoice has three parts. ", "Let me check."], max_sentences=2
    )

    assert tts.heard == ["The invoice has three parts.", "Let me check."]
    assert (dice.heard_when_rolled, model.rounds) == (None, 1)
    events = await kit.store.list_events(room_id)
    [reply] = [e for e in events if e.source.channel_id == "ai-0" and e.type == EventType.MESSAGE]
    # Written at the call's start, its text was whole; the turn ended cancelled.
    assert reply.metadata["loop_end_reason"] == "cancelled"
    [end] = [e for e in events if e.type == EventType.TOOL_CALL_END]
    assert end.content.status == "failed"  # announced, never run (RFC §12.2 step 13s)
    await kit.close()
