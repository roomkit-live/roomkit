"""A response that fails mid-stream is not handed again to the channel that streamed it.

RFC §12.2 step 13s (RMK-467): the text the response produced is stored,
reaches the streaming channel through the stream, as a completed response's
last row does, then the failure; every other channel gets it as an ordinary
event and ON_ERROR fires. A voice speaks all of it, its last partial sentence
included (step 15s). Only a failure of the streaming channel itself sends the
text to it again, since it may not have rendered it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from roomkit import AIChannel, RoomKit, VoiceChannel
from roomkit.channels.base import Channel
from roomkit.channels.cli import CLIChannel
from roomkit.channels.websocket import StreamMessage, WebSocketChannel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import RoomEvent, TextContent, ToolCallContent
from roomkit.models.hook import HookResult
from roomkit.models.room import Room
from roomkit.models.streaming import ToolCallStartMarker
from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    StreamEvent,
    StreamTextDelta,
)
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.base import TTSProvider
from tests.conftest import make_event
from tests.test_framework import SimpleChannel

SAID = "Your table is booked. "


class _FailsMidAnswer(AIProvider):
    """Streams some text, then its vendor goes down."""

    def __init__(self, said: str = SAID) -> None:
        self._said = said

    @property
    def model_name(self) -> str:
        return "mock-fails-mid-answer"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:  # pragma: no cover
        return AIResponse(content="unused")

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        yield StreamTextDelta(text=self._said)
        raise RuntimeError("ai down")


class _OpensCallsThenFails(Channel):
    """An intelligence channel whose response announces two calls, then fails."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput(responded=True, response_stream=self._stream())

    async def _stream(self) -> AsyncIterator[Any]:
        yield "Checking. "
        yield ToolCallStartMarker(tool_name="a", tool_id="c1", arguments={})
        yield ToolCallStartMarker(tool_name="b", tool_id="c2", arguments={})
        raise RuntimeError("ai down")

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class _CountingTTS(TTSProvider):
    """Records what each path synthesizes: streamed sentences and standard replays."""

    def __init__(self) -> None:
        self.streamed: list[str] = []
        self.standard: list[str] = []

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize(self, text: str, *, voice: str | None = None) -> object:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        self.standard.append(text)
        yield AudioChunk(data=b"\x00\x00", sample_rate=16000)

    async def synthesize_stream_input(
        self, text_stream: AsyncIterator[str], *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        async for sentence in text_stream:
            self.streamed.append(sentence)
            yield AudioChunk(data=b"\x00\x00", sample_rate=16000)


class _BufferingChannel(SimpleChannel):
    """A transport that takes a streamed answer through the default deliver_stream."""

    @property
    def supports_streaming_delivery(self) -> bool:
        return True


class _BrokenTransport(_BufferingChannel):
    """Takes the first delta, then its own transport drops."""

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        async for chunk in text_stream:
            if isinstance(chunk, str):
                raise ConnectionError("socket gone")
        return ChannelOutput.empty()  # pragma: no cover


class _SwallowingTransport(_BufferingChannel):
    """Renders the stream as it comes and swallows whatever it raises."""

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        with contextlib.suppress(Exception):
            async for _chunk in text_stream:
                pass
        return ChannelOutput.empty()


class _DeliverBreaks(_BufferingChannel):
    """The default deliver_stream, over a deliver that fails on the answer."""

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if event.source.channel_id == "ai-1":
            raise ConnectionError("deliver broke")
        return await super().deliver(event, binding, context)


class _CancelsItsPull(_BufferingChannel):
    """Cancels its pull of the failed response, as a voice cancels its read once
    every session barged in: after the failure is read (the pull ended) or while
    the response closes its calls, never before, which would leave the response
    stopped rather than failed."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.ending = asyncio.Event()

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        starts = 0
        while starts < 2:
            item = await anext(text_stream)
            if isinstance(item, RoomEvent) and item.type == EventType.TOOL_CALL_START:
                starts += 1
        pull = asyncio.ensure_future(anext(text_stream))
        ending = asyncio.ensure_future(self.ending.wait())
        await asyncio.wait({pull, ending}, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
        pull.cancel()
        ending.cancel()
        await asyncio.gather(pull, ending, return_exceptions=True)
        return ChannelOutput.empty()


class _KeepsItsStream(_BufferingChannel):
    """A transport that keeps the stream it is handed, to see whether it was closed."""

    handed: Any = None

    def _closed(self) -> bool:
        return self.handed is not None and self.handed.ag_frame is None


class _StopsAtTheRow(_KeepsItsStream):
    """Renders the deltas, then stops reading at the first row (the failed text's)."""

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        self.handed = text_stream
        async for chunk in text_stream:
            if isinstance(chunk, RoomEvent):
                break
        return ChannelOutput.empty()


class _DropsMidStream(_KeepsItsStream):
    """Takes the first delta, then its own transport drops."""

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        self.handed = text_stream
        await anext(text_stream)
        raise ConnectionError("socket gone")


class _HangsAfterADelta(_KeepsItsStream):
    """Takes the first delta, then waits on its transport until the turn is cancelled."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.pulled = asyncio.Event()

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        self.handed = text_stream
        await anext(text_stream)
        self.pulled.set()
        await asyncio.Event().wait()
        return ChannelOutput.empty()  # pragma: no cover


class _VoiceBench:
    def __init__(self) -> None:
        self.backend = MockVoiceBackend()
        self.tts = _CountingTTS()
        self.kit = RoomKit(voice=self.backend)
        self.channel: Channel = VoiceChannel("stream", tts=self.tts, backend=self.backend)

    async def ready(self, room_id: str) -> None:
        await self.kit.join(room_id, "stream", participant_id="user")

    def transcripts(self) -> list[str]:
        return [text for _, text, role in self.backend.sent_transcriptions if role == "assistant"]


class _WebSocketBench:
    """One client speaking the streaming protocol, one taking whole events."""

    def __init__(self) -> None:
        self.kit = RoomKit()
        self.channel: Channel = WebSocketChannel("stream")
        self.live: list[tuple[str, str | None]] = []
        self.plain: list[str | None] = []

    async def ready(self, room_id: str) -> None:
        async def live_event(_conn: str, event: RoomEvent) -> None:
            self.live.append(("event", _body(event)))

        async def live_stream(_conn: str, message: StreamMessage) -> None:
            self.live.append((message.type, None))

        async def plain_event(_conn: str, event: RoomEvent) -> None:
            self.plain.append(_body(event))

        assert isinstance(self.channel, WebSocketChannel)
        self.channel.register_connection(
            "live", live_event, room_id=room_id, stream_send_fn=live_stream
        )
        self.channel.register_connection("plain", plain_event, room_id=room_id)


class _CLIBench:
    markdown = False

    def __init__(self) -> None:
        self.kit = RoomKit()
        self.channel: Channel = CLIChannel("stream", use_color=False, markdown=self.markdown)

    async def ready(self, room_id: str) -> None:
        return None


class _MarkdownCLIBench(_CLIBench):
    markdown = True


class _BufferingBench:
    def __init__(self) -> None:
        self.kit = RoomKit()
        self.channel: Channel = _BufferingChannel("stream")

    async def ready(self, room_id: str) -> None:
        return None


Bench = _VoiceBench | _WebSocketBench | _CLIBench | _BufferingBench

BENCHES = [_VoiceBench, _WebSocketBench, _CLIBench, _MarkdownCLIBench, _BufferingBench]
BENCH_IDS = ["voice", "websocket", "cli", "cli-markdown", "buffering"]


@dataclass
class _Turn:
    result: InboundResult
    rows: list[RoomEvent]
    sms: list[str | None]
    errors: list[RoomEvent]
    after_tts: list[str]


def _body(event: RoomEvent) -> str | None:
    return event.content.body if isinstance(event.content, TextContent) else None


async def _run(
    kit: RoomKit,
    streaming: Channel,
    ready: Any = None,
    *,
    intelligence: Channel | None = None,
) -> _Turn:
    """One turn the user starts on *streaming*, which also renders the answer."""
    sms = SimpleChannel("sms1")
    kit.register_channel(streaming)
    kit.register_channel(sms)
    kit.register_channel(intelligence or AIChannel("ai-1", provider=_FailsMidAnswer()))
    room = await kit.create_room()
    await kit.attach_channel(room.id, "stream")
    await kit.attach_channel(room.id, "sms1")
    await kit.attach_channel(room.id, "ai-1", category=ChannelCategory.INTELLIGENCE)
    if ready is not None:
        await ready(room.id)
    errors: list[RoomEvent] = []
    after_tts: list[str] = []

    @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC)
    async def on_error(event: RoomEvent, _ctx: RoomContext) -> None:
        errors.append(event)

    @kit.hook(HookTrigger.AFTER_TTS, execution=HookExecution.ASYNC)
    async def on_after_tts(text: str, _ctx: RoomContext) -> None:
        after_tts.append(text)

    result = await kit.process_inbound(
        InboundMessage(
            channel_id="stream", sender_id="user", content=TextContent(body="Book a table")
        ),
        room_id=room.id,
    )
    events = await kit.store.list_events(room.id, offset=0, limit=50)
    await kit.close()
    return _Turn(
        result=result,
        rows=[e for e in events if e.source.channel_id == "ai-1"],
        sms=[_body(e) for e in sms.delivered if e.source.channel_id == "ai-1"],
        errors=errors,
        after_tts=after_tts,
    )


async def _bench_turn(bench: Bench, said: str = SAID) -> _Turn:
    ai = AIChannel("ai-1", provider=_FailsMidAnswer(said))
    return await _run(bench.kit, bench.channel, bench.ready, intelligence=ai)


class TestTheChannelThatStreamedIsNotHandedTheTextAgain:
    async def test_a_voice_channel_does_not_speak_it_again(self) -> None:
        bench = _VoiceBench()
        await _bench_turn(bench)

        assert bench.tts.streamed == [SAID.strip()]
        assert bench.tts.standard == []

    async def test_a_websocket_client_gets_the_row_inside_the_stream_and_never_after(
        self,
    ) -> None:
        bench = _WebSocketBench()
        await _bench_turn(bench)

        # As a completed response: the row inline, then the end of the stream.
        assert bench.live == [
            ("stream_start", None),
            ("stream_chunk", None),
            ("event", SAID),
            ("stream_error", None),
        ]
        assert bench.plain == [SAID]

    @pytest.mark.parametrize("bench_type", [_CLIBench, _MarkdownCLIBench], ids=["plain", "md"])
    async def test_the_cli_does_not_print_it_again(
        self, bench_type: type[_CLIBench], capsys: pytest.CaptureFixture
    ) -> None:
        await _bench_turn(bench_type())

        assert capsys.readouterr().out.count(SAID.strip()) == 1

    async def test_a_buffering_channel_delivers_what_it_buffered_once(self) -> None:
        bench = _BufferingBench()
        await _bench_turn(bench)

        assert isinstance(bench.channel, _BufferingChannel)
        assert [_body(e) for e in bench.channel.delivered if _body(e) == SAID] == [SAID]


async def test_a_buffering_channel_delivers_no_text_for_a_blank_stream() -> None:
    """A stream of line breaks is no message in the room (RMK-1300): the default
    ``deliver_stream`` delivers the empty answer; failing, it delivers nothing."""

    async def blank(*, fails: bool) -> AsyncIterator[str]:
        yield "\n\n"
        if fails:
            raise RuntimeError("ai down")

    channel = SimpleChannel("buffering")
    binding = ChannelBinding(
        channel_id="buffering", room_id="r1", channel_type=channel.channel_type
    )
    context = RoomContext(room=Room(id="r1"))

    await channel.deliver_stream(blank(fails=False), make_event(), binding, context)
    assert [event.content.body for event in channel.delivered] == [""]

    with pytest.raises(RuntimeError):
        await channel.deliver_stream(blank(fails=True), make_event(), binding, context)
    assert len(channel.delivered) == 1


@pytest.mark.parametrize(
    "said",
    ["Your table is booked. ", "Your table is booked.", "Booked. "],
    ids=["sentence-flushed", "no-trailing-space", "under-min-chunk"],
)
async def test_a_voice_speaks_all_the_failed_response_produced(said: str) -> None:
    """The splitter's last partial sentence is spoken too, then reported as heard."""
    bench = _VoiceBench()
    turn = await _bench_turn(bench, said)

    assert bench.tts.streamed == [said.strip()]
    assert bench.tts.standard == []
    # The whole streamed text, as a completed response reports it.
    assert bench.transcripts() == [said]
    assert turn.after_tts == [said]
    assert str(turn.result.error) == "ai down"


@pytest.mark.parametrize("bench_type", BENCHES, ids=BENCH_IDS)
async def test_the_others_get_the_text_and_on_error_fires(bench_type: type[Bench]) -> None:
    turn = await _bench_turn(bench_type())

    assert turn.sms == [SAID]
    assert len(turn.errors) == 1
    assert str(turn.result.error) == "ai down"
    (row,) = turn.rows
    assert _body(row) == SAID
    assert "cancelled" not in row.metadata


async def test_the_failure_is_logged_with_its_traceback(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="roomkit.framework"):
        await _bench_turn(_BufferingBench())

    (record,) = [r for r in caplog.records if "streaming delivery of ai-1" in r.getMessage()]
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], RuntimeError)


async def test_a_streaming_channel_that_fails_itself_gets_the_text_again() -> None:
    """Its transport dropped while rendering: what it was handed may be unseen."""
    channel = _BrokenTransport("stream")
    turn = await _run(RoomKit(), channel)

    assert [_body(e) for e in channel.delivered if e.source.channel_id == "ai-1"] == [SAID]
    assert turn.sms == [SAID]
    assert isinstance(turn.result.error, ConnectionError)
    assert len(turn.errors) == 1


async def test_a_channel_that_swallows_the_failure_does_not_make_the_turn_a_success() -> None:
    channel = _SwallowingTransport("stream")
    turn = await _run(RoomKit(), channel)

    assert [e for e in channel.delivered if e.source.channel_id == "ai-1"] == []
    assert turn.sms == [SAID]
    assert str(turn.result.error) == "ai down"
    assert len(turn.errors) == 1
    (row,) = turn.rows
    assert "cancelled" not in row.metadata


async def test_the_response_failure_stays_the_turns_when_the_channel_fails_on_top(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger="roomkit.framework"):
        turn = await _run(RoomKit(), _DeliverBreaks("stream"))

    assert str(turn.result.error) == "ai down"
    assert turn.sms == [SAID]
    assert any("deliver broke" in str(r.exc_info[1]) for r in caplog.records if r.exc_info)


async def test_a_pull_cancelled_while_the_failed_response_ends_leaves_no_call_pending() -> None:
    """The open calls are closed outside the read the channel cancels."""
    kit = RoomKit()
    channel = _CancelsItsPull("stream")

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def slow_ends(event: RoomEvent, _ctx: RoomContext) -> HookResult:
        if event.type == EventType.TOOL_CALL_END:
            channel.ending.set()
            await asyncio.sleep(0.015)
        return HookResult.allow()

    turn = await _run(kit, channel, intelligence=_OpensCallsThenFails("ai-1"))

    ends = [
        r.content
        for r in turn.rows
        if r.type == EventType.TOOL_CALL_END and isinstance(r.content, ToolCallContent)
    ]
    assert sorted(c.tool_id for c in ends) == ["c1", "c2"]
    assert all(c.status == "failed" for c in ends)
    assert str(turn.result.error) == "ai down"


@pytest.mark.parametrize(
    "channel_type", [_StopsAtTheRow, _DropsMidStream], ids=["stops-at-the-row", "channel-fails"]
)
async def test_the_stream_handed_to_the_channel_is_closed_when_the_turn_fails(
    channel_type: type[_KeepsItsStream],
) -> None:
    channel = channel_type("stream")
    turn = await _run(RoomKit(), channel)

    assert turn.result.error is not None
    assert channel._closed()


async def test_the_stream_handed_to_the_channel_is_closed_when_the_turn_is_cancelled() -> None:
    kit, channel = RoomKit(), _HangsAfterADelta("stream")
    kit.register_channel(channel)
    kit.register_channel(AIChannel("ai-1", provider=_FailsMidAnswer()))
    room = await kit.create_room()
    await kit.attach_channel(room.id, "stream")
    await kit.attach_channel(room.id, "ai-1", category=ChannelCategory.INTELLIGENCE)
    turn = asyncio.create_task(
        kit.process_inbound(
            InboundMessage(channel_id="stream", sender_id="user", content=TextContent(body="Hi")),
            room_id=room.id,
        )
    )
    await asyncio.wait_for(channel.pulled.wait(), 2)

    turn.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await turn

    assert channel._closed()
    await kit.close()
