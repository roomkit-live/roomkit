"""A message with a media, read outside the AI channel (RFC §6.6, §19.7.4, RMK-767).

Every reader that takes a room's message as text reads it the way the AI
channel does: its text, each media as its label (a voice message as its
transcript), a composite's parts. One that reads names in it, or a search
query, reads the author's words alone, a caption included, never a file name.
A Loop, like a one-pass supervisor, answers a message its tasks cannot carry
through its lead agent alone, which sees the media.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from roomkit import RoomKit
from roomkit._media import content_text, content_words
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.memory.retrieval import RetrievalMemory
from roomkit.memory.token_estimator import extract_event_text
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import (
    AudioContent,
    CompositeContent,
    EventContent,
    EventSource,
    MediaContent,
    RichContent,
    RoomEvent,
    TextContent,
    VideoContent,
)
from roomkit.models.room import Room
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.strategies.discussion._driver import TurnDriver
from roomkit.orchestration.strategies.discussion.dispatch import classifier as dispatch
from roomkit.orchestration.strategies.loop import Loop
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIContext, AITextPart
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.scoring.hook import ScoringHook
from roomkit.speaking import classifier as speaking
from roomkit.tasks.brief import conversation_block
from tests.orchestration.test_loop_strategy import _make_mock_kit
from tests.orchestration.test_strategy_discussion import _provider, _room, _settle
from tests.tool_loop_modes import read_reply, respond

_PHOTO = MediaContent(
    url="app-media://tenant-a/photo-1", mime_type="image/png", filename="lamp.png"
)
_COMPOSITE = CompositeContent(parts=[TextContent(body="compare with mine"), _PHOTO])
_VOICE = AudioContent(url="data:audio/ogg;base64,T2dnUw==", transcript="what time is the meeting?")
_SILENT_VOICE = _VOICE.model_copy(update={"transcript": None})
_AI = ChannelBinding(
    channel_id="ai1",
    room_id="r1",
    channel_type=ChannelType.AI,
    category=ChannelCategory.INTELLIGENCE,
)
_MEMBER = ChannelBinding(channel_id="sms1", room_id="r1", channel_type=ChannelType.SMS)


def _event(content: EventContent, channel_id: str = "sms1") -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id=channel_id, channel_type=ChannelType.SMS),
        content=content,
    )


class TestSharedReadings:
    def test_the_whole_message_reads_its_text_and_its_media_label(self) -> None:
        assert content_text(_COMPOSITE) == "compare with mine\n[Media: lamp.png]"

    def test_the_authors_words_keep_a_caption_and_never_a_file_name(self) -> None:
        captioned = _PHOTO.model_copy(update={"caption": "my lamp", "filename": "@nova.png"})

        assert content_words(captioned) == "my lamp"
        assert content_words(_PHOTO) == ""
        assert content_words(_COMPOSITE) == "compare with mine"

    def test_a_voice_message_reads_its_transcript_and_a_video_its_label(self) -> None:
        video = VideoContent(url="data:video/mp4;base64,AAAA")

        assert content_text(_VOICE) == content_words(_VOICE) == "what time is the meeting?"
        assert content_text(_SILENT_VOICE) == "[Voice message: audio/ogg]"
        assert content_text(video) == "[Video: video/mp4]"
        # Neither a type nor a payload is the author's words.
        assert content_words(_SILENT_VOICE) == content_words(video) == ""

    def test_rich_content_reads_as_its_plain_text(self) -> None:
        rich = RichContent(body="**hi**", plain_text="hi")

        assert content_text(rich) == content_words(rich) == "hi"


def _loop(lead: Agent, helper: Agent) -> Orchestration:
    return Loop(agent=lead, reviewers=[helper])


def _one_pass(lead: Agent, helper: Agent) -> Orchestration:
    return Supervisor(
        supervisor=lead,
        workers=[helper],
        strategy="sequential",
        auto_delegate=True,
        refine_task=False,
    )


class TestLeadAnswersAlone:
    @pytest.mark.parametrize("strategy", [_loop, _one_pass], ids=["loop", "one-pass"])
    @pytest.mark.parametrize(
        "content",
        [_COMPOSITE, _PHOTO, TextContent(body="")],
        ids=["composite", "photo", "empty text"],
    )
    async def test_a_message_the_tasks_cannot_carry_is_answered_by_the_lead_alone(
        self, strategy: Callable[[Agent, Agent], Orchestration], content: EventContent
    ) -> None:
        lead_provider = MockAIProvider(responses=["a red lamp, like yours"], vision=True)
        helper_provider = MockAIProvider(responses=["approved"])
        lead = Agent("lead", provider=lead_provider)
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock()
        await strategy(lead, Agent("helper", provider=helper_provider)).install(kit, "r1")
        binding = _AI.model_copy(update={"channel_id": "lead"})

        output = await lead.on_event(_event(content), binding, RoomContext(room=Room(id="r1")))
        run = await read_reply(output)

        # The lead saw the message itself and answered; no task was delegated.
        assert run.text == "a red lamp, like yours"
        assert len(lead_provider.calls) == 1 and helper_provider.calls == []
        kit.delegate.assert_not_awaited()


class TestTextReaders:
    def test_a_workers_brief_carries_a_composite_a_media_and_a_voice_message(self) -> None:
        bindings = [_MEMBER, _AI.model_copy(update={"channel_id": "worker"})]
        context = RoomContext(
            room=Room(id="r1"),
            bindings=bindings,
            recent_events=[_event(_COMPOSITE), _event(_PHOTO), _event(_VOICE)],
        )

        block = conversation_block(context, ["worker"], 5)

        assert block is not None
        assert "compare with mine\\n[Media: lamp.png]" in block  # a JSON string
        assert '"[Media: lamp.png]"' in block
        assert '"what time is the meeting?"' in block
        assert "app-media://" not in block and "data:" not in block

    @pytest.mark.parametrize("reader", [speaking._text, dispatch._text], ids=["speak", "dispatch"])
    def test_a_classifier_reads_a_composite(self, reader: object) -> None:
        assert reader(_event(_COMPOSITE)) == "compare with mine\n[Media: lamp.png]"  # type: ignore[operator]

    def test_a_discussion_turn_quotes_the_message_it_answers(self) -> None:
        stub = SimpleNamespace(
            _room=SimpleNamespace(agents={}, label=lambda channel_id: channel_id)
        )

        note = TurnDriver._answering(stub, _event(_COMPOSITE), RoomContext(room=Room(id="r1")))  # type: ignore[arg-type]

        assert "compare with mine [Media: lamp.png]" in note

    def test_a_scoring_query_is_the_latest_message_a_voice_message_included(self) -> None:
        recent = [_event(TextContent(body="an older question")), _event(_VOICE)]
        answer = SimpleNamespace(channel_id="ai1")

        query = ScoringHook._extract_last_user_query(
            answer,
            RoomContext(  # type: ignore[arg-type]
                room=Room(id="r1"), recent_events=recent
            ),
        )

        assert query == "what time is the meeting?"

    def test_a_memory_summary_reads_a_voice_message_as_its_label(self) -> None:
        assert extract_event_text(_event(_SILENT_VOICE)) == "[Voice message: audio/ogg]"
        assert extract_event_text(_event(_VOICE)) == "what time is the meeting?"

    def test_a_search_query_reads_the_caption_never_the_file_name(self) -> None:
        captioned = _PHOTO.model_copy(update={"caption": "my red lamp"})

        assert RetrievalMemory._extract_query(_event(captioned)) == "my red lamp"
        assert RetrievalMemory._extract_query(_event(_COMPOSITE)) == "compare with mine"
        assert RetrievalMemory._extract_query(_event(_VOICE)) == "what time is the meeting?"


def _said(context: AIContext) -> str:
    texts: list[str] = []
    for message in context.messages:
        content = message.content
        for part in content if isinstance(content, list) else [content]:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, AITextPart):
                texts.append(part.text)
    return "\n".join(texts)


class TestAIChannelHistory:
    @pytest.mark.parametrize("vision", [False, True], ids=["text model", "vision model"])
    async def test_history_reads_a_voice_message_and_rich_content_as_live(
        self, vision: bool
    ) -> None:
        # Live, the router hands the channel a voice message as its transcript
        # and keeps rich content (RFC §6.6); history reads them the same (§6.4).
        provider = MockAIProvider(responses=["noted"], vision=vision)
        channel = AIChannel("ai1", provider=provider)
        rich = RichContent(body="**bold ask**", plain_text="bold ask")
        history = [
            _event(_VOICE),
            _event(rich),
            _event(CompositeContent(parts=[RichContent(body="*see*", plain_text="see"), _VOICE])),
        ]
        current = _event(TextContent(body="and now?"))
        context = RoomContext(
            room=Room(id="r1"), bindings=[_MEMBER, _AI], recent_events=[*history, current]
        )

        await respond(channel, current, _AI, context)
        said = _said(provider.calls[-1])

        assert said.count("what time is the meeting?") == 2
        assert "bold ask" in said and "**bold ask**" not in said
        assert "see" in said and "*see*" not in said


class TestDiscussionAddressing:
    @staticmethod
    async def _say(kit: RoomKit, content: EventContent) -> None:
        await kit.process_inbound(
            InboundMessage(channel_id="ops", sender_id="oncall", content=content)
        )

    @pytest.mark.parametrize(
        "content",
        [
            CompositeContent(parts=[TextContent(body="@a look at this"), _PHOTO]),
            _PHOTO.model_copy(update={"caption": "@a look at this"}),
        ],
        ids=["composite text", "caption"],
    )
    async def test_a_name_the_author_wrote_asks_that_agent_alone(
        self, content: EventContent
    ) -> None:
        a, b = _provider("a answers"), _provider("b answers")
        kit = RoomKit()
        await _room(kit, {"a": a, "b": b})

        await self._say(kit, content)
        await _settle(kit)

        assert (len(a.calls), len(b.calls)) == (1, 0)
        await kit.close()

    @pytest.mark.parametrize(
        "filename", ["@b notes.png", "@carol notes.png"], ids=["agent", "person"]
    )
    async def test_a_name_in_a_file_name_addresses_no_one(self, filename: str) -> None:
        a, b = _provider("a answers"), _provider("b answers")
        kit = RoomKit()
        await _room(kit, {"a": a, "b": b}, people=["carol"])
        named_file = _PHOTO.model_copy(update={"filename": filename})

        await self._say(
            kit, CompositeContent(parts=[TextContent(body="what is this?"), named_file])
        )
        await _settle(kit)

        # Unaddressed: everyone is asked; neither an agent is singled out nor
        # the message left to a person by the file name.
        assert (len(a.calls), len(b.calls)) == (1, 1)
        await kit.close()
