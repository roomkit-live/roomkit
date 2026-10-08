"""A voice channel's sentence budget: at most N sentences spoken per reply, the
reply ended at the first sentence over it (RMK-623, RFC §12.2 step 12s.e)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import AIChannel, HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
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
from roomkit.voice.tts.mock import MockTTSProvider
from tests.test_voice_stream_barge_in import _SentenceTTS, _StoppableBackend
from tests.voice.streamed_replies import StreamedReply, ask, said, voice_rooms

SIX = [
    "The invoice has three parts. ",
    "The first part is your plan. ",
    "The second part is the calls. ",
    "The third part is the taxes. ",
    "Taxes depend on your province. ",
    "That is the whole invoice.",
]


async def _stored(kit: RoomKit, room_id: str) -> Any:
    [reply] = [e for e in await kit.store.list_events(room_id) if e.source.channel_id == "ai-0"]
    return reply


async def test_a_reply_past_its_budget_ends_at_the_first_sentence_over_it() -> None:
    kit, _, backend, tts, [session] = await voice_rooms(SIX, max_sentences=2)

    await ask(backend, session)
    [heard] = await said(tts, 1)

    assert heard == "The invoice has three parts. The first part is your plan."
    provider = kit.get_channel("ai-0")._provider  # type: ignore[union-attr]
    assert isinstance(provider, StreamedReply) and provider.read < len(SIX)
    reply = await _stored(kit, session.room_id)
    assert reply.metadata.get("cancelled") is True  # ended like a barge-in (13s)
    assert "taxes" not in reply.content.body  # the room keeps about what was heard
    final = [text for _, text, role in backend.sent_transcriptions if role == "assistant"]
    assert final[-1] == heard
    await kit.close()


async def test_a_last_sentence_over_the_budget_ends_the_reply_too() -> None:
    """Review of RMK-623: a sentence ends only when what follows it comes, and
    the last one never had anything after it; the reply stops at its first
    word, so it is cut like any other, and AFTER_TTS reports what was said."""
    kit, _, backend, tts, [session] = await voice_rooms(SIX, max_sentences=5)
    after: list[str] = []
    responses: list[Any] = []

    @kit.hook(HookTrigger.AFTER_TTS, execution=HookExecution.ASYNC)
    async def spoken(text: str, ctx: Any) -> None:
        after.append(text)

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def answered(event: Any, ctx: Any) -> None:
        responses.append(event)

    await ask(backend, session)
    [heard] = await said(tts, 1)

    assert heard == "".join(SIX[:5]).strip()
    assert after == [heard]
    assert (await _stored(kit, session.room_id)).metadata.get("cancelled") is True
    assert responses == []  # a turn its reader stopped fires nothing
    await kit.close()


class _BooksAfterTalking(AIProvider):
    """Says two sentences, a third, then calls ``book_table``."""

    def __init__(self) -> None:
        self.booked = 0
        self.rounds = 0

    @property
    def model_name(self) -> str:
        return "books-after-talking"

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
            yield StreamTextDelta(text=" Your table is booked.")
            yield StreamDone(finish_reason="stop")
            return
        for text in SIX[:2]:
            yield StreamTextDelta(text=text)
        yield StreamTextDelta(text="Let me book that table for you.")
        yield StreamToolCall(id="t1", name="book_table")
        yield StreamDone(finish_reason="tool_calls")

    async def book(self, name: str, arguments: dict[str, Any]) -> str:
        self.booked += 1
        return "booked"


async def test_a_tool_call_after_the_sentence_over_the_budget_never_starts() -> None:
    """Review of RMK-623: the person heard two sentences; the reply stops before
    the turn asks for the call announced in the third, which is never made."""
    backend, model = _StoppableBackend(), _BooksAfterTalking()
    kit = RoomKit(voice=backend)
    kit.register_channel(
        VoiceChannel("voice-1", tts=_SentenceTTS(), backend=backend, max_sentences=2)
    )
    book = AITool(name="book_table", description="Book a table.", parameters={"type": "object"})
    kit.register_channel(AIChannel("ai-0", provider=model, tools=[book], tool_handler=model.book))
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    await kit.attach_channel(room.id, "ai-0")
    await kit.join(room.id, "voice-1", participant_id="user-0")

    asked = InboundMessage(
        channel_id="voice-1", sender_id="user-0", content=TextContent(body="Hi")
    )
    await asyncio.wait_for(kit.process_inbound(asked, room_id=room.id), 5)
    await asyncio.sleep(0.2)

    assert (model.booked, model.rounds) == (0, 1)
    reply = await _stored(kit, room.id)
    assert reply.metadata.get("cancelled") is True
    await kit.close()


async def test_a_reply_of_exactly_its_budget_runs_to_its_end() -> None:
    kit, _, backend, tts, [session] = await voice_rooms(SIX[:2], max_sentences=2)

    await ask(backend, session)
    [heard] = await said(tts, 1)

    assert heard == "The invoice has three parts. The first part is your plan."
    reply = await _stored(kit, session.room_id)
    assert not reply.metadata.get("cancelled")
    await kit.close()


async def test_a_sentence_before_tts_drops_does_not_count() -> None:
    kit, _, backend, tts, [session] = await voice_rooms(SIX, max_sentences=2)

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def no_intro(sentence: str, ctx: Any) -> HookResult:
        if sentence.startswith("The invoice"):
            return HookResult.block("intro")
        return HookResult.allow()

    await ask(backend, session)
    [heard] = await said(tts, 1)

    assert heard == "The first part is your plan. The second part is the calls."
    await kit.close()


async def test_a_reply_delivered_whole_is_spoken_to_its_budget() -> None:
    """A TTS without streamed input: the reply reaches the voice whole, stored
    as it is, and its first sentences are spoken."""
    tts = MockTTSProvider()
    kit, _, backend, _, [session] = await voice_rooms(SIX, tts=tts, max_sentences=2)

    await ask(backend, session)
    for _ in range(50):
        if tts.calls:
            break
        await asyncio.sleep(0.02)

    assert tts.calls[-1]["text"] == "The invoice has three parts. The first part is your plan."
    assert (await _stored(kit, session.room_id)).content.body == "".join(SIX)
    await kit.close()


async def test_say_has_no_budget() -> None:
    tts, backend = MockTTSProvider(), MockVoiceBackend()
    voice = VoiceChannel("voice-1", tts=tts, backend=backend, max_sentences=1)
    session = await backend.connect("room-1", "user-1", "voice-1")

    await voice.say(session, "".join(SIX[:3]))

    assert tts.calls[-1]["text"] == "".join(SIX[:3])


@pytest.mark.parametrize("budget", [0, -1])
def test_a_budget_below_one_is_refused(budget: int) -> None:
    with pytest.raises(ValueError, match="max_sentences"):
        VoiceChannel("voice-1", max_sentences=budget)
