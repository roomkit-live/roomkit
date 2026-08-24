"""Speaker attribution in the AI context (multi-speaker rooms).

``_build_context`` flattens every non-self event into an anonymous "user"
stream. In a room where several people speak, that erases who said what: the
model can only guess the addressee, and it guesses wrong (a reply opening with
the wrong colleague's name). The property pinned here: whenever the history
window holds two or more distinct speakers, every attributable user turn the
model receives names its speaker, and the system prompt says how to read the
prefixes — while a single-speaker room (a 1:1 DM) is byte-identical to before.
"""

from __future__ import annotations

from roomkit.channels import SMSChannel
from roomkit.channels._ai_context import (
    _SPEAKER_ATTRIBUTION_NOTE,
    _event_speaker,
    _with_speaker_prefix,
)
from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIImagePart, AITextPart
from roomkit.providers.ai.mock import MockAIProvider


async def _kit(responses: list[str]) -> tuple[RoomKit, MockAIProvider]:
    kit = RoomKit()
    provider = MockAIProvider(responses=responses)
    kit.register_channel(SMSChannel("sms1"))
    kit.register_channel(AIChannel("ai1", provider=provider))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    return kit, provider


async def _say(kit: RoomKit, sender_id: str, name: str | None, body: str) -> None:
    await kit.process_inbound(
        InboundMessage(
            channel_id="sms1",
            sender_id=sender_id,
            content=TextContent(body=body),
            metadata={"sender_name": name} if name else {},
        )
    )


def _user_texts(context) -> list[str]:
    return [str(m.content) for m in context.messages if m.role == "user"]


class TestMultiSpeakerAttribution:
    async def test_two_speakers_prefix_every_attributable_user_turn(self) -> None:
        kit, provider = await _kit(["a1", "a2", "a3"])
        await _say(kit, "u-alice", "Alice", "Tuesday works for me.")
        await _say(kit, "u-bob", "Bob", "I would rather ship Thursday.")
        await _say(kit, "u-alice", "Alice", "Who proposed what?")

        last = provider.calls[-1]
        texts = _user_texts(last)
        assert any(t == "Alice: Tuesday works for me." for t in texts)
        assert any(t == "Bob: I would rather ship Thursday." for t in texts)
        # The trigger turn is attributed too.
        assert any(t == "Alice: Who proposed what?" for t in texts)
        # The model is told how to read the prefixes, once.
        assert last.system_prompt is not None
        assert last.system_prompt.count(_SPEAKER_ATTRIBUTION_NOTE) == 1

    async def test_assistant_turns_are_never_prefixed(self) -> None:
        kit, provider = await _kit(["first answer", "a2"])
        await _say(kit, "u-alice", "Alice", "hello")
        await _say(kit, "u-bob", "Bob", "hi again")

        last = provider.calls[-1]
        assistant_texts = [str(m.content) for m in last.messages if m.role == "assistant"]
        assert assistant_texts == ["first answer"]

    async def test_single_speaker_room_is_untouched(self) -> None:
        kit, provider = await _kit(["a1", "a2"])
        await _say(kit, "u-alice", "Alice", "first message")
        await _say(kit, "u-alice", "Alice", "second message")

        last = provider.calls[-1]
        texts = _user_texts(last)
        assert "first message" in texts
        assert "second message" in texts
        assert not any(t.startswith("Alice:") for t in texts)
        assert _SPEAKER_ATTRIBUTION_NOTE not in (last.system_prompt or "")

    async def test_unnamed_turn_stays_bare_in_a_multi_speaker_room(self) -> None:
        kit, provider = await _kit(["a1", "a2", "a3"])
        await _say(kit, "u-alice", "Alice", "named one")
        await _say(kit, "u-ghost", None, "nameless interjection")
        await _say(kit, "u-bob", "Bob", "named two")

        last = provider.calls[-1]
        texts = _user_texts(last)
        assert any(t == "nameless interjection" for t in texts)
        assert any(t == "Alice: named one" for t in texts)


class TestSpeakerResolution:
    def _event(self, *, metadata: dict | None = None, participant_id: str | None = None):
        return RoomEvent(
            room_id="r1",
            source=EventSource(
                channel_id="sms1",
                channel_type=ChannelType.SMS,
                participant_id=participant_id,
            ),
            content=TextContent(body="x"),
            metadata=metadata or {},
        )

    def _context(self, participants: list[Participant]) -> RoomContext:
        return RoomContext(room=Room(id="r1"), participants=participants)

    def test_sender_name_metadata_wins(self) -> None:
        event = self._event(metadata={"sender_name": "  Alice  "}, participant_id="p1")
        assert _event_speaker(event, self._context([])) == "Alice"

    def test_participant_display_name_is_the_fallback(self) -> None:
        event = self._event(participant_id="p1")
        ctx = self._context(
            [Participant(id="p1", room_id="r1", channel_id="sms1", display_name="Bob")]
        )
        assert _event_speaker(event, ctx) == "Bob"

    def test_no_name_anywhere_resolves_to_none(self) -> None:
        event = self._event(participant_id="p1")
        ctx = self._context([Participant(id="p1", room_id="r1", channel_id="sms1")])
        assert _event_speaker(event, ctx) is None

    def test_multimodal_content_gets_a_lead_text_part(self) -> None:
        parts = [AIImagePart(url="data:image/png;base64,x", mime_type="image/png")]
        out = _with_speaker_prefix(parts, "Alice")
        assert isinstance(out, list)
        assert isinstance(out[0], AITextPart)
        assert out[0].text == "Alice:"
        assert out[1:] == parts
