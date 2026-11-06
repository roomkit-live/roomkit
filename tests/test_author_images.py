"""What an image an author sent becomes in a model's input (RFC §6.4, RMK-763).

An application that keeps its media in private storage stores a reference (a
URL in a scheme it chooses) and hands the framework a ``media_resolver``: the
channel asks it for the bytes as it builds each turn, so the store and every
broadcast carry the reference alone. The newest images go as images, with
their author's words; the rest, and anything a provider cannot take, as a
mention, never a failed turn.
"""

from __future__ import annotations

import base64
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit._media import canonical_data_uri, media_label
from roomkit.channels._acp_context import acp_event_text
from roomkit.channels._turn_notes import TURN_IMAGES_HEADER
from roomkit.channels.ai import AIChannel
from roomkit.core.transcoder import DefaultContentTranscoder
from roomkit.memory.token_estimator import estimate_event_tokens
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelMediaType, ChannelType
from roomkit.models.event import (
    AudioContent,
    CompositeContent,
    EventContent,
    MediaContent,
    RoomEvent,
    TextContent,
    VideoContent,
)
from roomkit.models.room import Room
from roomkit.providers.ai import ResponseSchemaError, ToolNameRule
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AIResponse,
    AITextPart,
    AITool,
    AIToolCall,
    ProviderError,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.providers.ai.image_parts import image_part_base64
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.utils import to_data_uri
from tests.conftest import make_event
from tests.test_framework import SimpleChannel
from tests.tool_loop_modes import read_reply, respond

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)
_PNG = to_data_uri(_PNG_BYTES, "image/png")
_REF = "app-media://photo-1"
_BINDING = ChannelBinding(
    channel_id="ai1",
    room_id="r1",
    channel_type=ChannelType.AI,
    category=ChannelCategory.INTELLIGENCE,
)
_MEMBER = ChannelBinding(channel_id="sms1", room_id="r1", channel_type=ChannelType.SMS)


class _MediaWatcher(SimpleChannel):
    """A transport that takes media: what a broadcast hands it, untranscoded."""

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(media_types=[ChannelMediaType.TEXT, ChannelMediaType.MEDIA])


class _Host:
    """An application's media store: the bytes of each reference it holds."""

    def __init__(self, media: dict[str, bytes] | None = None, *, fails: bool = False) -> None:
        self.media = media if media is not None else {_REF: _PNG_BYTES}
        self.fails = fails
        self.asked: list[tuple[str, str]] = []

    async def __call__(self, media: MediaContent, room_id: str) -> bytes | None:
        self.asked.append((media.url, room_id))
        if self.fails:
            raise RuntimeError("storage unreachable")
        return self.media.get(media.url)


def _photo(
    url: str = _REF,
    *,
    filename: str | None = "lamp.png",
    caption: str | None = None,
    mime_type: str = "image/png",
) -> MediaContent:
    return MediaContent(url=url, mime_type=mime_type, filename=filename, caption=caption)


def _event(content: EventContent) -> RoomEvent:
    return make_event(room_id="r1", channel_id="sms1").model_copy(update={"content": content})


def _channel(
    streaming: bool = False,
    *,
    vision: bool = True,
    resolver: _Host | None = None,
    **kwargs: object,
) -> AIChannel:
    provider = MockAIProvider(responses=["ok"] * 5, streaming=streaming, vision=vision)
    channel = AIChannel("ai1", provider=provider, **kwargs)  # type: ignore[arg-type]
    channel._media_resolver = resolver  # noqa: SLF001 - the framework wires it at registration
    return channel


async def _turn(
    channel: AIChannel, event: RoomEvent, history: Sequence[RoomEvent] = ()
) -> AIContext:
    # The member's channel is bound too: an unbound source's events are
    # visible to no other channel.
    context = RoomContext(
        room=Room(id="r1"), bindings=[_MEMBER, _BINDING], recent_events=[*history, event]
    )
    await respond(channel, event, _BINDING, context)
    return channel.provider.calls[-1]  # type: ignore[attr-defined]


def _parts(context: AIContext) -> list[object]:
    parts: list[object] = []
    for message in context.messages:
        parts.extend(message.content if isinstance(message.content, list) else [message.content])
    return parts


def _images(context: AIContext) -> list[str]:
    return [part.url for part in _parts(context) if isinstance(part, AIImagePart)]


def _text(context: AIContext) -> str:
    return "\n".join(
        part if isinstance(part, str) else part.text
        for part in _parts(context)
        if isinstance(part, str | AITextPart)
    )


class TestReference:
    async def test_a_reference_reaches_the_model_as_the_hosts_image(self, streaming: bool) -> None:
        host = _Host()
        channel = _channel(streaming, resolver=host)
        event = _event(CompositeContent(parts=[TextContent(body="compare with mine"), _photo()]))

        context = await _turn(channel, event)

        assert _images(context) == [_PNG]
        assert host.asked == [(_REF, "r1")]
        content = context.messages[-1].content
        assert isinstance(content, list)
        assert isinstance(content[0], AITextPart) and content[0].text.startswith(
            "compare with mine"
        )
        assert isinstance(content[1], AIImagePart)

    async def test_the_store_and_broadcasts_carry_the_reference_alone(self) -> None:
        host = _Host()
        provider = MockAIProvider(responses=["a nice lamp"], vision=True)
        kit = RoomKit(media_resolver=host)
        watcher = _MediaWatcher("watcher")
        kit.register_channel(SimpleChannel("user"))
        kit.register_channel(watcher)
        kit.register_channel(AIChannel("ai1", provider=provider))
        try:
            await kit.create_room(room_id="room")
            await kit.attach_channel("room", "user")
            await kit.attach_channel("room", "watcher")
            await kit.attach_channel("room", "ai1", category=ChannelCategory.INTELLIGENCE)
            message = InboundMessage(channel_id="user", sender_id="person", content=_photo())

            await kit.process_inbound(message, room_id="room")

            assert _images(provider.calls[-1]) == [_PNG]
            stored = [e.content for e in await kit.store.list_events("room")]
            assert _photo() in stored
            delivered = [
                e.content for e in watcher.delivered if isinstance(e.content, MediaContent)
            ]
            assert delivered == [_photo()]
            assert "base64" not in repr(stored) + repr(watcher.delivered)
        finally:
            await kit.close()

    async def test_the_image_is_still_there_on_the_next_turn(self, streaming: bool) -> None:
        host = _Host()
        channel = _channel(streaming, resolver=host)
        photo = _event(_photo())

        context = await _turn(
            channel, _event(TextContent(body="what colour is its foot?")), [photo]
        )

        assert _images(context) == [_PNG]
        first, last = context.messages[0], context.messages[-1]
        assert isinstance(first.content, list) and isinstance(first.content[-1], AIImagePart)
        assert last.content == "what colour is its foot?"
        assert host.asked == [(_REF, "r1")]  # resolved as the turn is built, nothing kept

    async def test_a_link_goes_to_the_provider_as_it_is(self) -> None:
        host = _Host()
        link = "https://cdn.example.com/lamp.png"

        context = await _turn(_channel(resolver=host), _event(_photo(link)))

        assert _images(context) == [link]
        assert host.asked == []

    async def test_the_authors_image_comes_before_the_replayed_tool_images(self) -> None:
        channel = _channel(resolver=_Host())
        tool_image = to_data_uri(b"tool-image", "image/png")
        channel._tool_usage.record(  # noqa: SLF001
            "r1", "generate_image", {"prompt": "a lamp"}, [AIImagePart(url=tool_image)]
        )
        event = _event(CompositeContent(parts=[TextContent(body="compare with mine"), _photo()]))

        context = await _turn(channel, event)

        content = context.messages[-1].content
        assert isinstance(content, list)
        urls = [p.url for p in content if isinstance(p, AIImagePart)]
        header = next(
            i
            for i, p in enumerate(content)
            if isinstance(p, AITextPart) and p.text.startswith(TURN_IMAGES_HEADER)
        )
        assert urls == [_PNG, tool_image]
        assert content.index(next(p for p in content if isinstance(p, AIImagePart))) < header


class TestBound:
    @staticmethod
    def _history(count: int) -> tuple[_Host, list[RoomEvent]]:
        host = _Host({f"app-media://p{i}": bytes([i]) * 8 for i in range(count)})
        events = [_event(_photo(f"app-media://p{i}", filename=f"p{i}.png")) for i in range(count)]
        return host, events

    async def test_only_the_newest_four_images_go(self, streaming: bool) -> None:
        host, history = self._history(5)

        context = await _turn(
            _channel(streaming, resolver=host), _event(TextContent(body="?")), history
        )

        assert _images(context) == [to_data_uri(bytes([i]) * 8, "image/png") for i in range(1, 5)]
        assert "[Earlier image not shown: “p0.png”]" in _text(context)
        assert ("app-media://p0", "r1") not in host.asked

    @pytest.mark.parametrize(("bound", "sent"), [(1, 1), (0, 0), (None, 5)])
    async def test_the_bound_is_the_channels_to_change(self, bound: int | None, sent: int) -> None:
        host, history = self._history(5)
        channel = _channel(resolver=host, max_author_images=bound)

        context = await _turn(channel, _event(TextContent(body="?")), history)

        assert len(_images(context)) == sent

    async def test_inline_images_count_the_same(self) -> None:
        history = [_event(_photo(to_data_uri(bytes([i]) * 8, "image/png"))) for i in range(5)]

        context = await _turn(_channel(), _event(TextContent(body="?")), history)

        assert len(_images(context)) == 4

    def test_a_negative_bound_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_author_images"):
            _channel(max_author_images=-1)


class TestMention:
    @pytest.mark.parametrize(
        "host",
        [None, _Host({}), _Host(fails=True), _Host({_REF: b"x" * (5 * 1024 * 1024 + 1)})],
        ids=["no resolver", "resolver has none", "resolver fails", "too large"],
    )
    async def test_an_image_that_does_not_resolve_leaves_the_turn_a_mention(
        self, host: _Host | None, streaming: bool
    ) -> None:
        context = await _turn(_channel(streaming, resolver=host), _event(_photo()))

        assert _images(context) == []
        assert "[Image unavailable: “lamp.png”]" in _text(context)

    async def test_a_document_is_a_mention_not_an_image(self) -> None:
        pdf = _photo(
            to_data_uri(b"%PDF-1.7", "application/pdf"),
            filename="contract.pdf",
            mime_type="application/pdf",
        )

        context = await _turn(_channel(), _event(pdf))

        assert _images(context) == []
        assert "[Media: contract.pdf]" in _text(context)
        assert "base64" not in _text(context)


class TestModelThatDoesNotSee:
    async def test_it_reads_a_mention_never_the_bytes(self, streaming: bool) -> None:
        host = _Host()
        history = [_event(_photo(_PNG, caption="my lamp")), _event(_photo())]
        channel = _channel(streaming, vision=False, resolver=host)

        context = await _turn(channel, _event(TextContent(body="what was that?")), history)

        assert "[Image not shown, this model does not see images: \u201cmy lamp\u201d]" in _text(
            context
        )
        assert "[Image not shown, this model does not see images: \u201clamp.png\u201d]" in _text(
            context
        )
        assert "base64" not in repr(context.messages)
        assert host.asked == []

    async def test_a_composite_keeps_its_words(self) -> None:
        composite = CompositeContent(parts=[TextContent(body="compare with mine"), _photo()])

        context = await _turn(
            _channel(vision=False), _event(TextContent(body="?")), [_event(composite)]
        )

        not_seen = "[Image not shown, this model does not see images: \u201clamp.png\u201d]"
        assert f"compare with mine\n{not_seen}" in _text(context)

    async def test_a_document_reads_as_its_label(self) -> None:
        pdf = _photo(_REF, filename="contract.pdf", mime_type="application/pdf")
        composite = CompositeContent(parts=[TextContent(body="sign this"), pdf])

        context = await _turn(
            _channel(vision=False), _event(TextContent(body="?")), [_event(composite)]
        )

        assert "sign this\n[Media: contract.pdf]" in _text(context)

    async def test_a_captionless_document_reads_as_its_label_without_a_describer(
        self,
    ) -> None:
        pdf = _photo(_REF, filename="contract.pdf", mime_type="application/pdf")

        context = await _turn(_channel(vision=False), _event(TextContent(body="?")), [_event(pdf)])

        assert "[Media: contract.pdf]" in _text(context)


class TestTranscoding:
    @pytest.mark.parametrize(
        ("url", "label"),
        [
            (_PNG, "[Media: image/png]"),
            (_REF, "[Media: image/png]"),
            ("https://cdn.example.com/a.png", "[Media: https://cdn.example.com/a.png]"),
        ],
    )
    def test_a_media_label_never_shows_a_data_uri_or_a_reference(
        self, url: str, label: str
    ) -> None:
        assert media_label(_photo(url, filename=None)) == label

    async def test_a_text_only_channel_reads_the_label(self) -> None:
        text_only = ChannelBinding(
            channel_id="sms",
            room_id="r1",
            channel_type=ChannelType.SMS,
            capabilities=ChannelCapabilities(media_types=[ChannelMediaType.TEXT]),
        )

        content = await DefaultContentTranscoder().transcode(
            _photo(_PNG, filename=None), _BINDING, text_only
        )

        assert content == TextContent(body="[Media: image/png]")


class TestBudget:
    def test_a_document_costs_its_label_not_an_image(self) -> None:
        pdf = _event(_photo(_REF, filename="contract.pdf", mime_type="application/pdf"))
        image = _event(_photo())

        assert estimate_event_tokens(pdf) < 50
        assert estimate_event_tokens(image) >= 1000


class TestReferenceURL:
    @pytest.mark.parametrize("url", ["app-media://photo-1", "store+v2://bucket/a.png", _PNG])
    def test_a_reference_link_or_data_uri_is_accepted(self, url: str) -> None:
        assert MediaContent(url=url, mime_type="image/png").url == url

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "FILE:///etc/passwd",
            "ftp://example.com/a.png",
            "javascript://x",
            "/etc/passwd",
            "lamp.png",
            "C://Windows/win.ini",
            "app-media://../../etc/passwd",
            "app-media://photos/../../../etc/passwd",
            "app-media://a\\b",
        ],
    )
    def test_a_url_a_transport_could_read_or_run_is_refused(self, url: str) -> None:
        # Some transports read a URL they are handed (a path, a file: URI).
        with pytest.raises(ValueError, match="reference"):
            MediaContent(url=url, mime_type="image/png")


class _Scripted(MockAIProvider):
    """Answers each request with the next step: a response, or an error raised."""

    def __init__(
        self,
        steps: list[AIResponse | Exception],
        *,
        streaming: bool = False,
        response_schema: bool = False,
    ) -> None:
        super().__init__(vision=True, streaming=streaming, response_schema=response_schema)
        self._steps = list(steps)

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        return self._next_response()

    def _next_response(self) -> AIResponse:
        # A turn constrained to a schema streams this one, unchecked.
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class _BreaksMidStream(_Scripted):
    """Streams the start of its first answer, then fails: the request was taken."""

    def __init__(
        self,
        steps: list[AIResponse | Exception],
        *,
        first: StreamEvent,
        response_schema: bool = False,
    ) -> None:
        super().__init__(steps, streaming=True, response_schema=response_schema)
        self._first = first

    async def _scripted_events(self, context: AIContext) -> Any:
        if len(self.calls) > 0:
            async for event in super()._scripted_events(context):
                yield event
            return
        self.calls.append(context)
        yield self._first
        raise ProviderError("400 overloaded mid-answer", retryable=False, status_code=400)


_OK = AIResponse(content="ok", finish_reason="stop")
_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "string"}},
    "required": ["a"],
    "additionalProperties": False,
}
_LOOKUP = AITool(name="lookup", description="Look something up", parameters={})


async def _lookup(name: str, arguments: dict[str, Any]) -> str:
    return "found"


async def _openai_refusal(model: str, *, tools: bool) -> ProviderError:
    """What OpenAI's provider raises before the request for *model*."""
    provider = OpenAIAIProvider(OpenAIConfig(api_key="sk-test", model=model))
    context = AIContext(
        messages=[AIMessage(role="user", content="hi")], tools=[_LOOKUP] if tools else []
    )
    with pytest.raises(ProviderError) as refused:
        await provider.generate(context)
    return refused.value


def _name_refusal(rule: ToolNameRule) -> ProviderError:
    """What a vendor's tool name rule raises for a name it refuses."""
    with pytest.raises(ProviderError) as refused:
        rule.check(["1abc"])
    return refused.value


async def _turns(
    provider: MockAIProvider, *, tool: bool = False, schema: dict[str, Any] | None = None
) -> AIContext:
    """A photo's turn, which may fail, then a question: the context the question got."""
    channel = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=_lookup if tool else None,
        response_schema=schema,
    )
    channel._media_resolver = _Host()  # noqa: SLF001
    binding = _BINDING.model_copy(
        update={"metadata": {"tools": [_LOOKUP.model_dump()]} if tool else {}}
    )
    photo = _event(_photo())
    first = RoomContext(room=Room(id="r1"), bindings=[_MEMBER, binding], recent_events=[photo])
    with contextlib.suppress(ProviderError):
        await read_reply(await channel.on_event(photo, binding, first))
    question = _event(TextContent(body="and now?"))
    after = RoomContext(
        room=Room(id="r1"), bindings=[_MEMBER, binding], recent_events=[photo, question]
    )
    await read_reply(await channel.on_event(question, binding, after))
    return provider.calls[-1]


class TestRefusal:
    async def test_an_image_the_provider_refused_reads_as_refused_after(
        self, streaming: bool
    ) -> None:
        refused = ProviderError(
            "400 the image could not be processed", retryable=False, status_code=400
        )

        context = await _turns(_Scripted([refused, _OK], streaming=streaming))

        assert _images(context) == []
        assert "[Image refused by the model's provider: \u201clamp.png\u201d]" in _text(context)

    @pytest.mark.parametrize("status", [401, 402, 403, 404, 429])
    async def test_an_error_that_is_not_about_the_content_refuses_nothing(
        self, status: int
    ) -> None:
        # A key refused, a bill unpaid (PolarGrid, Cerebras), a permission, a
        # missing model, a rate limit.
        error = ProviderError(f"{status} invalid x-api-key", retryable=False, status_code=status)

        context = await _turns(_Scripted([error, _OK]))

        assert _images(context) == [_PNG]

    @pytest.mark.parametrize(
        "refusal",
        [
            # The turn's schema a model that takes none is refused (RMK-1436).
            lambda: ResponseSchemaError(
                "mock does not support a response schema", reason="unsupported", provider="mock"
            ),
            # A model OpenAI's Chat Completions does not serve, refused before
            # the request as its catalogue states it.
            lambda: _openai_refusal("gpt-5.5-pro", tools=False),
            lambda: _openai_refusal("gpt-6-astra", tools=True),
            # A tool name the vendor refuses, checked before the request.
            lambda: _name_refusal(ToolNameRule("acme", r"[a-z]+")),
        ],
        ids=[
            "unsupported-schema",
            "responses-only-model",
            "model-without-chat-tools",
            "tool-name",
        ],
    )
    async def test_a_refusal_before_any_request_refuses_nothing(
        self, refusal: Callable[[], Awaitable[ProviderError] | ProviderError], streaming: bool
    ) -> None:
        """No request went out: nothing says the provider refused the content,
        and the image goes on the next turn as its author sent it (RFC §6.4)."""
        error = refusal()
        if not isinstance(error, ProviderError):
            error = await error

        context = await _turns(_Scripted([error, _OK], streaming=streaming))

        assert _images(context) == [_PNG]

    async def test_a_refusal_raised_from_one_before_any_request_is_its_own(self) -> None:
        """Only the error raised says it came before any request: a provider
        that raises its own from such a refusal is read as itself."""
        wrapped = ProviderError("an integrator's provider failed")
        wrapped.__cause__ = _name_refusal(ToolNameRule("acme", r"[a-z]+"))

        context = await _turns(_Scripted([wrapped, _OK]))

        assert _images(context) == []

    async def test_a_later_round_failing_refuses_nothing(self) -> None:
        # Round 0 carried the image and was answered: the provider took it.
        call = AIResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[AIToolCall(id="c0", name="lookup", arguments={})],
        )
        error = ProviderError("400 bad request", retryable=False, status_code=400)

        context = await _turns(_Scripted([call, error, _OK]), tool=True)

        assert _images(context) == [_PNG]

    @pytest.mark.parametrize(
        ("first", "schema"),
        [
            (StreamTextDelta(text="Let me look"), None),
            # RMK-1220: a call the loop serves reaches no reader, yet the
            # provider began its answer.
            (StreamToolCall(id="c0", name="lookup", arguments={}), None),
            # A turn constrained to a schema holds its text until checked:
            # the provider began its answer all the same.
            (StreamTextDelta(text='{"a": "x'), _SCHEMA),
        ],
        ids=["text", "tool-call", "schema"],
    )
    async def test_a_stream_under_way_refuses_nothing(
        self, first: StreamEvent, schema: dict[str, Any] | None
    ) -> None:
        answer = _OK if schema is None else AIResponse(content='{"a": "x"}', finish_reason="stop")
        provider = _BreaksMidStream([answer], first=first, response_schema=schema is not None)

        context = await _turns(provider, tool=schema is None, schema=schema)

        assert _images(context) == [_PNG]


class TestLivePath:
    """Through the router, which transcodes what a binding cannot take."""

    @staticmethod
    async def _first_turn(
        provider: MockAIProvider, content: MediaContent | None = None, **channel: Any
    ) -> AIContext:
        kit = RoomKit()
        kit.register_channel(SimpleChannel("user"))
        kit.register_channel(AIChannel("ai1", provider=provider, **channel))
        try:
            await kit.create_room(room_id="room")
            await kit.attach_channel("room", "user")
            await kit.attach_channel("room", "ai1", category=ChannelCategory.INTELLIGENCE)
            photo = content or MediaContent(url=_PNG, mime_type="image/png")
            await kit.process_inbound(
                InboundMessage(channel_id="user", sender_id="person", content=photo),
                room_id="room",
            )
            return provider.calls[-1]
        finally:
            await kit.close()

    async def test_a_live_captionless_document_goes_to_the_describer_first(self) -> None:
        provider = MockAIProvider(responses=["ok"], vision=False)
        pdf = MediaContent(url=_REF, mime_type="application/pdf", filename="contract.pdf")

        context = await self._first_turn(
            provider, pdf, describe_empty_event=lambda event: "[a file the user sent]"
        )

        assert context.messages[-1].content == "[a file the user sent]"

    async def test_a_live_captionless_image_says_the_model_does_not_see_it(self) -> None:
        provider = MockAIProvider(responses=["ok"], vision=False)

        context = await self._first_turn(
            provider, describe_empty_event=lambda event: "[a photo the user sent]"
        )

        assert (
            context.messages[-1].content
            == "[Image not shown, this model does not see images: \u201cimage/png\u201d]"
        )

    async def test_a_model_that_does_not_see_never_reads_the_payload(self) -> None:
        context = await self._first_turn(MockAIProvider(responses=["ok"], vision=False))

        assert "base64" not in repr(context.messages)

    async def test_a_model_that_sees_gets_the_image(self) -> None:
        context = await self._first_turn(MockAIProvider(responses=["ok"], vision=True))

        assert _images(context) == [_PNG]


class TestConstraints:
    @pytest.mark.parametrize("mime_type", ["image/jpg", "image/jpeg; charset=binary", "IMAGE/PNG"])
    async def test_a_type_spelled_another_way_still_goes_as_an_image(self, mime_type: str) -> None:
        link = "https://cdn.example.com/lamp.jpg"

        context = await _turn(_channel(), _event(_photo(link, mime_type=mime_type)))

        assert _images(context) == [link]

    @pytest.mark.parametrize(
        ("url", "mime_type", "sent"),
        [
            (to_data_uri(_PNG_BYTES, "image/jpg"), "image/jpg", "image/jpeg"),
            ("DATA:IMAGE/PNG;BASE64," + _PNG.split(",", 1)[1], "IMAGE/PNG", "image/png"),
            (_REF, "image/jpg", "image/jpeg"),
            (_REF, "Image/PNG; charset=binary", "image/png"),
        ],
    )
    async def test_an_image_goes_with_its_type_folded(
        self, url: str, mime_type: str, sent: str
    ) -> None:
        # Anthropic takes image/jpeg, never image/jpg, and sent a "DATA:" URI
        # as a link: the first turn failed, then the image read as refused.
        channel = _channel(resolver=_Host())

        context = await _turn(channel, _event(_photo(url, mime_type=mime_type)))

        [part] = [part for part in _parts(context) if isinstance(part, AIImagePart)]
        assert part.url.startswith(f"data:{sent};base64,")
        assert part.mime_type == sent
        assert image_part_base64(part, provider="test")[0] == sent

    def test_a_data_uri_is_spelled_as_every_vendor_reads_it(self) -> None:
        assert canonical_data_uri("DATA:image/JPG;BASE64,AA==") == "data:image/jpeg;base64,AA=="
        assert canonical_data_uri("data:image/png;base64,AA==") == "data:image/png;base64,AA=="

    async def test_an_empty_data_uri_reads_as_unavailable(self) -> None:
        # A transport whose download failed hands an empty payload.
        context = await _turn(_channel(), _event(_photo("data:image/jpeg;base64,")))

        assert _images(context) == []
        assert "[Image unavailable: \u201clamp.png\u201d]" in _text(context)

    @pytest.mark.parametrize(
        ("content", "label"),
        [
            (AudioContent(url="data:audio/ogg;base64,T2dnUw=="), "[Voice message: audio/ogg]"),
            (VideoContent(url="data:video/mp4;base64,AAAA"), "[Video: video/mp4]"),
            (
                AudioContent(url="https://x.example/a.ogg"),
                "[Voice message: https://x.example/a.ogg]",
            ),
        ],
    )
    async def test_a_voice_or_video_label_never_shows_a_payload(
        self, content: EventContent, label: str
    ) -> None:
        text_only = ChannelBinding(
            channel_id="sms",
            room_id="r1",
            channel_type=ChannelType.SMS,
            capabilities=ChannelCapabilities(media_types=[ChannelMediaType.TEXT]),
        )

        transcoded = await DefaultContentTranscoder().transcode(content, _BINDING, text_only)

        assert transcoded == TextContent(body=label)

    @pytest.mark.parametrize(
        ("url", "stored"),
        [
            ("HTTP://cdn.example.com/A.png", "http://cdn.example.com/A.png"),
            ("Https://cdn.example.com/a.png", "https://cdn.example.com/a.png"),
            ("DATA:image/png;base64,AA", "data:image/png;base64,AA"),
            ("App-Media://Photo-1", "App-Media://Photo-1"),  # a reference is the host's
        ],
    )
    def test_a_scheme_in_capitals_is_stored_as_every_reader_tests_it(
        self, url: str, stored: str
    ) -> None:
        # Readers test the lower-case scheme: neonize opened "HTTP://…" as a
        # path, and a provider sent "DATA:…" as a link to fetch.
        assert MediaContent(url=url, mime_type="image/png").url == stored


def test_an_acp_agent_reads_a_composite_with_its_media() -> None:
    composite = CompositeContent(parts=[TextContent(body="compare with mine"), _photo()])

    assert acp_event_text(_event(composite)) == "compare with mine\n[Media: lamp.png]"
