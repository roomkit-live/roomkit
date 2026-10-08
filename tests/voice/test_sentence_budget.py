"""A voice channel's sentence budget: at most N sentences spoken per reply, the
reply ended at the first sentence over it (RMK-623, RFC §12.2 step 12s.e)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.tts.mock import MockTTSProvider
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
