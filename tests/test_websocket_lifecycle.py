"""Tests for WebSocket lifecycle integration (Area 4.7)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.channels.websocket import (
    StreamChunk,
    StreamEnd,
    StreamMessage,
    StreamStart,
    WebSocketChannel,
)
from roomkit.core.framework import ChannelNotRegisteredError, RoomKit
from roomkit.core.mixins._streaming_segments import SegmentWriter
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType, EventStatus, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.hook import HookResult
from roomkit.models.room import Room
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    AITool,
    AIToolCall,
    StreamEvent,
    StreamTextDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


def _make_event(room_id: str = "room-1") -> RoomEvent:
    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id="ai", channel_type=ChannelType.AI),
        content=TextContent(body=""),
    )


def _make_binding(channel_id: str = "ws1") -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id="room-1",
        channel_type=ChannelType.WEBSOCKET,
        capabilities=ChannelCapabilities(media_types=[ChannelMediaType.TEXT]),
    )


def _make_context() -> RoomContext:
    return RoomContext(room=Room(id="room-1"), bindings=[_make_binding()])


async def _noop_send(conn_id: str, event: RoomEvent) -> None:
    pass


async def _noop_stream_send(conn_id: str, msg: StreamMessage) -> None:
    pass


async def _text_stream(*chunks: str) -> Any:
    for chunk in chunks:
        yield chunk


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


def _row(event: RoomEvent) -> str:
    """A row as a connection reads it: its text, else its type."""
    if isinstance(event.content, TextContent):
        return event.content.body
    return event.type.value


async def _block_the_agent(event: RoomEvent, context: RoomContext) -> HookResult:
    if event.source.channel_id == "ai":
        return HookResult.block("withheld")
    return HookResult.allow()


class _Stalls(MockAIProvider):
    """Streams its first words, then waits until its turn is cut."""

    def __init__(self) -> None:
        super().__init__(streaming=True)

    async def _scripted_events(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        yield StreamTextDelta(text="Partial ")
        await asyncio.Event().wait()


class _Fails(MockAIProvider):
    """Streams its first words, then fails."""

    def __init__(self) -> None:
        super().__init__(streaming=True)

    async def _scripted_events(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        yield StreamTextDelta(text="Partial ")
        raise RuntimeError("provider down")


class _WorksOn(WebSocketChannel):
    """Reads its stream to its end, or to its failure, then keeps working."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.cuts: list[RoomEvent | None] = []
        self.read_all = asyncio.Event()

    async def deliver_stream(
        self, text_stream: Any, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> Any:
        with contextlib.suppress(RuntimeError):
            async for _ in text_stream:
                pass
        self.read_all.set()
        await asyncio.Event().wait()

    async def deliver_cut(
        self,
        event: RoomEvent,
        row: RoomEvent | None,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> None:
        self.cuts.append(row)


async def _cut_room(provider: MockAIProvider, ws: WebSocketChannel) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(AIChannel("ai", provider=provider))
    kit.register_channel(ws)
    await kit.create_room(room_id="room-1")
    for channel_id in ("ai", ws.channel_id):
        await kit.attach_channel("room-1", channel_id)
    return kit


def _keyed(key: str) -> InboundMessage:
    return InboundMessage(
        channel_id="ws", sender_id="alice", content=TextContent(body="Hi"), delivery_key=key
    )


class TestConnectWebSocket:
    async def test_connect_registers_connection(self, kit: RoomKit) -> None:
        """connect_websocket registers a connection on the channel."""
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        await kit.connect_websocket("ws1", "conn1", send, room_id="room-1")
        assert ws.connection_count == 1

    async def test_connect_emits_framework_event(self, kit: RoomKit) -> None:
        """connect_websocket emits a 'channel_connected' framework event."""
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        received: list[FrameworkEvent] = []

        @kit.on("channel_connected")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        await kit.connect_websocket("ws1", "conn1", send, room_id="room-1")
        assert len(received) == 1
        assert received[0].type == "channel_connected"
        assert received[0].data["connection_id"] == "conn1"

    async def test_connect_non_websocket_raises(self, kit: RoomKit) -> None:
        """connect_websocket raises for non-WebSocket channels."""
        ch = SimpleChannel("sms1")
        kit.register_channel(ch)

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        with pytest.raises(ChannelNotRegisteredError):
            await kit.connect_websocket("sms1", "conn1", send, room_id="room-1")

    async def test_connect_unregistered_raises(self, kit: RoomKit) -> None:
        """connect_websocket raises for unregistered channel."""

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        with pytest.raises(ChannelNotRegisteredError):
            await kit.connect_websocket("nope", "conn1", send, room_id="room-1")


class TestDisconnectWebSocket:
    async def test_disconnect_unregisters_connection(self, kit: RoomKit) -> None:
        """disconnect_websocket removes the connection."""
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        await kit.connect_websocket("ws1", "conn1", send, room_id="room-1")
        assert ws.connection_count == 1

        await kit.disconnect_websocket("ws1", "conn1")
        assert ws.connection_count == 0

    async def test_disconnect_emits_framework_event(self, kit: RoomKit) -> None:
        """disconnect_websocket emits 'channel_disconnected' framework event."""
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        received: list[FrameworkEvent] = []

        @kit.on("channel_disconnected")
        async def handler(fe: FrameworkEvent) -> None:
            if fe.type == "channel_disconnected":
                received.append(fe)

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        await kit.connect_websocket("ws1", "conn1", send, room_id="room-1")
        await kit.disconnect_websocket("ws1", "conn1")
        assert len(received) == 1
        assert received[0].data["connection_id"] == "conn1"

    async def test_disconnect_nonexistent_is_safe(self, kit: RoomKit) -> None:
        """disconnect_websocket does not raise for unknown channel/conn."""
        # Should not raise even for unregistered channel
        await kit.disconnect_websocket("nope", "conn1")

    async def test_multiple_connections(self, kit: RoomKit) -> None:
        """Multiple WebSocket connections can be registered and tracked."""
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)

        async def send(conn_id: str, event: RoomEvent) -> None:
            pass

        await kit.connect_websocket("ws1", "conn1", send, room_id="room-1")
        await kit.connect_websocket("ws1", "conn2", send, room_id="room-1")
        assert ws.connection_count == 2

        await kit.disconnect_websocket("ws1", "conn1")
        assert ws.connection_count == 1


class TestStreamingDelivery:
    """Tests for WebSocket streaming text delivery."""

    async def test_supports_streaming_delivery_false_by_default(self) -> None:
        """No stream_send_fn registered → supports_streaming_delivery is False."""
        ws = WebSocketChannel("ws1")
        ws.register_connection("c1", _noop_send, room_id="room-1")
        assert ws.supports_streaming_delivery is False

    async def test_supports_streaming_delivery_true(self) -> None:
        """At least one stream_send_fn → supports_streaming_delivery is True."""
        ws = WebSocketChannel("ws1")
        ws.register_connection(
            "c1", _noop_send, room_id="room-1", stream_send_fn=_noop_stream_send
        )
        assert ws.supports_streaming_delivery is True

    async def test_deliver_stream_sends_protocol_messages(self) -> None:
        """deliver_stream sends stream_start, stream_chunk(s), stream_end."""
        ws = WebSocketChannel("ws1")
        received: list[StreamMessage] = []

        async def stream_send(conn_id: str, msg: StreamMessage) -> None:
            received.append(msg)

        ws.register_connection("c1", _noop_send, room_id="room-1", stream_send_fn=stream_send)

        event = _make_event()
        binding = _make_binding()
        context = _make_context()

        await ws.deliver_stream(_text_stream("Hello", " world"), event, binding, context)

        assert len(received) == 4  # start + 2 chunks + end
        assert isinstance(received[0], StreamStart)
        assert isinstance(received[1], StreamChunk)
        assert isinstance(received[2], StreamChunk)
        assert isinstance(received[3], StreamEnd)

        # All share the same stream_id
        stream_id = received[0].stream_id
        assert all(m.stream_id == stream_id for m in received)

    async def test_deliver_stream_accumulated_text(self) -> None:
        """StreamChunk.text accumulates progressively."""
        ws = WebSocketChannel("ws1")
        chunks: list[StreamChunk] = []

        async def stream_send(conn_id: str, msg: StreamMessage) -> None:
            if isinstance(msg, StreamChunk):
                chunks.append(msg)

        ws.register_connection("c1", _noop_send, room_id="room-1", stream_send_fn=stream_send)
        await ws.deliver_stream(
            _text_stream("a", "b", "c"), _make_event(), _make_binding(), _make_context()
        )

        assert [c.text for c in chunks] == ["a", "ab", "abc"]

    async def test_deliver_stream_delta_text(self) -> None:
        """StreamChunk.delta contains only the new text for each chunk."""
        ws = WebSocketChannel("ws1")
        chunks: list[StreamChunk] = []

        async def stream_send(conn_id: str, msg: StreamMessage) -> None:
            if isinstance(msg, StreamChunk):
                chunks.append(msg)

        ws.register_connection("c1", _noop_send, room_id="room-1", stream_send_fn=stream_send)
        await ws.deliver_stream(
            _text_stream("Hello", " ", "world"), _make_event(), _make_binding(), _make_context()
        )

        assert [c.delta for c in chunks] == ["Hello", " ", "world"]

    @pytest.mark.parametrize("chunks", [("Hi", " there"), ("\n", "\n")], ids=["text", "blank"])
    async def test_a_plain_connection_reads_nothing_of_a_stream_that_hands_no_row(
        self, chunks: tuple[str, ...]
    ) -> None:
        """A connection without stream_send_fn reads the rows a stream hands,
        and no event the stream's text would make, which the room never held:
        a row blocked, kept out or narrowed away is not handed, and a blank
        text is no row (RFC Appendix A.8, RMK-994, RMK-1300)."""
        ws = WebSocketChannel("ws1")
        final_events: list[RoomEvent] = []

        async def regular_send(conn_id: str, event: RoomEvent) -> None:
            final_events.append(event)

        ws.register_connection("c1", regular_send, room_id="room-1")

        await ws.deliver_stream(
            _text_stream(*chunks), _make_event(), _make_binding(), _make_context()
        )

        assert final_events == []

    async def test_deliver_stream_error_removes_connection(self) -> None:
        """Streaming send failures increment error count and remove after threshold."""
        ws = WebSocketChannel("ws1")

        async def failing_stream_send(conn_id: str, msg: StreamMessage) -> None:
            raise ConnectionError("gone")

        ws.register_connection(
            "c1", _noop_send, room_id="room-1", stream_send_fn=failing_stream_send
        )
        assert ws.connection_count == 1

        # stream_start (err 1), then 3 chunks each cause an error
        # After 3 consecutive errors the connection is removed
        await ws.deliver_stream(
            _text_stream("a", "b", "c"), _make_event(), _make_binding(), _make_context()
        )

        assert ws.connection_count == 0

    async def test_deliver_stream_empty_stream(self) -> None:
        """Empty text_stream sends start + end with empty text content."""
        ws = WebSocketChannel("ws1")
        received: list[StreamMessage] = []

        async def stream_send(conn_id: str, msg: StreamMessage) -> None:
            received.append(msg)

        ws.register_connection("c1", _noop_send, room_id="room-1", stream_send_fn=stream_send)

        await ws.deliver_stream(_text_stream(), _make_event(), _make_binding(), _make_context())

        assert len(received) == 2  # start + end only
        assert isinstance(received[0], StreamStart)
        assert isinstance(received[1], StreamEnd)
        assert isinstance(received[1].event.content, TextContent)
        assert received[1].event.content.body == ""

    async def test_deliver_stream_mixed_connections(self) -> None:
        """Mixed: streaming connections get protocol, a non-streaming one gets
        nothing of a stream that hands no row."""
        ws = WebSocketChannel("ws1")
        stream_msgs: list[StreamMessage] = []
        regular_events: list[RoomEvent] = []

        async def stream_send(conn_id: str, msg: StreamMessage) -> None:
            stream_msgs.append(msg)

        async def regular_send(conn_id: str, event: RoomEvent) -> None:
            regular_events.append(event)

        # c1 = streaming, c2 = regular
        ws.register_connection("c1", _noop_send, room_id="room-1", stream_send_fn=stream_send)
        ws.register_connection("c2", regular_send, room_id="room-1")

        await ws.deliver_stream(
            _text_stream("foo", "bar"), _make_event(), _make_binding(), _make_context()
        )

        # Streaming connection got start + 2 chunks + end
        assert len(stream_msgs) == 4
        assert isinstance(stream_msgs[0], StreamStart)
        assert isinstance(stream_msgs[-1], StreamEnd)

        assert regular_events == []

    @pytest.mark.parametrize(
        ("responses", "blocked", "rows"),
        [
            pytest.param([AIResponse(content="Answer.")], False, ["Answer."], id="answer"),
            pytest.param(
                [
                    AIResponse(
                        content="Looking.",
                        tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})],
                    ),
                    AIResponse(content="Done."),
                ],
                False,
                ["Looking.", "tool_call_start", "tool_call_end", "Done."],
                id="after-a-call",
            ),
            pytest.param([AIResponse(content="\n")], False, [], id="blank"),
            pytest.param([AIResponse(content="Secret.")], True, [], id="blocked"),
        ],
    )
    async def test_a_plain_connection_reads_a_streamed_answer_as_its_rows_once(
        self, responses: list[AIResponse], blocked: bool, rows: list[str]
    ) -> None:
        """A connection that does not stream reads the rows the stream hands
        it, each once, and no event the answer streamed under: not the empty
        one after its rows, nor its text when no row was handed (RFC Appendix
        A.8, RMK-994); the streaming connection reads its deltas, then an end
        that carries no text, none a hook refused (RMK-1760)."""
        kit = RoomKit()
        kit.register_channel(
            AIChannel(
                "ai",
                provider=MockAIProvider(ai_responses=responses, streaming=True),
                tools=[AITool(name="lookup", description="Look it up.")],
                tool_handler=_found,
            )
        )
        ws = WebSocketChannel("ws")
        kit.register_channel(ws)
        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "ai")
        await kit.attach_channel("room-1", "ws")
        if blocked:
            kit.hook(HookTrigger.BEFORE_BROADCAST)(_block_the_agent)
        plain: list[RoomEvent] = []
        streamed: list[str] = []

        async def send(conn_id: str, event: RoomEvent) -> None:
            plain.append(event)

        async def stream_send(conn_id: str, msg: StreamMessage) -> None:
            streamed.append(f"end {_row(msg.event)!r}" if isinstance(msg, StreamEnd) else msg.type)

        ws.register_connection("plain", send, room_id="room-1")
        ws.register_connection(
            "streaming", _noop_send, room_id="room-1", stream_send_fn=stream_send
        )
        try:
            await kit.process_inbound(
                InboundMessage(channel_id="ws", sender_id="alice", content=TextContent(body="Hi")),
                room_id="room-1",
            )

            delivered = [
                event.id
                for event in await kit.store.list_events("room-1")
                if event.source.channel_id == "ai" and event.status == EventStatus.DELIVERED
            ]
            assert [_row(event) for event in plain] == rows
            assert [event.id for event in plain] == delivered
            assert streamed == ["stream_start", *["stream_chunk"] * len(responses), "end ''"]
        finally:
            await kit.close()

    @pytest.mark.parametrize(
        ("blocked", "streaming_reads", "plain_reads"),
        [
            pytest.param(False, "end 'Partial ' cancelled", ["Partial "], id="kept"),
            pytest.param(True, "end '' None", [], id="blocked"),
        ],
    )
    async def test_a_stream_cut_from_outside_ends_with_its_cut_row(
        self, blocked: bool, streaming_reads: str, plain_reads: list[str]
    ) -> None:
        """A streamed answer cut from outside ends for every connection with
        what the room holds: the streaming one reads an end carrying the cut
        row, marked cancelled, the plain one reads the row; a cut row a hook
        refused leaves an end with no text, and nothing for the plain one
        (RFC §12.2 step 13s, Appendix A.8, RMK-1756)."""
        kit = RoomKit()
        kit.register_channel(AIChannel("ai", provider=_Stalls()))
        ws = WebSocketChannel("ws")
        sms = SimpleChannel("sms")
        for channel in (ws, sms):
            kit.register_channel(channel)
        await kit.create_room(room_id="room-1")
        for channel_id in ("ai", "ws", "sms"):
            await kit.attach_channel("room-1", channel_id)
        if blocked:
            kit.hook(HookTrigger.BEFORE_BROADCAST)(_block_the_agent)
        plain: list[RoomEvent] = []
        streamed: list[str] = []
        shown = asyncio.Event()

        async def send(conn_id: str, event: RoomEvent) -> None:
            if event.source.channel_id == "ai":
                plain.append(event)

        async def stream_send(conn_id: str, msg: StreamMessage) -> None:
            if isinstance(msg, StreamEnd):
                cancelled = msg.event.metadata.get("cancelled")
                streamed.append(f"end {_row(msg.event)!r} {'cancelled' if cancelled else None}")
            else:
                streamed.append(msg.type)
                if isinstance(msg, StreamChunk):
                    shown.set()

        ws.register_connection("plain", send, room_id="room-1")
        ws.register_connection(
            "streaming", _noop_send, room_id="room-1", stream_send_fn=stream_send
        )
        try:
            call = asyncio.ensure_future(
                kit.process_inbound(
                    InboundMessage(
                        channel_id="sms",
                        sender_id="alice",
                        content=TextContent(body="Hi"),
                        delivery_key="k",
                    ),
                    room_id="room-1",
                )
            )
            await asyncio.wait_for(shown.wait(), 5)
            assert await kit.cancel_delivery("k") == 1
            await asyncio.wait_for(call, 5)

            assert streamed == ["stream_start", "stream_chunk", streaming_reads]
            assert [_row(event) for event in plain] == plain_reads
            if plain_reads:
                stored = (await kit.store.list_events("room-1"))[-1]
                assert stored.metadata.get("cancelled") is True
                assert plain[0].id == stored.id
        finally:
            await kit.close()

    @pytest.mark.parametrize(
        "provider",
        [
            pytest.param(lambda: MockAIProvider(streaming=True), id="read-to-end"),
            pytest.param(_Fails, id="failed"),
        ],
    )
    async def test_a_cut_after_the_channel_read_its_stream_hands_it_no_end(
        self, provider: Any
    ) -> None:
        """A channel handed its stream's end, or its failure, no longer reads
        it: a cut while it works on hands it no deliver_cut, as in Rust
        (RFC §12.2 step 13s, RMK-1756)."""
        ws = _WorksOn("ws")
        kit = await _cut_room(provider(), ws)
        ws.register_connection(
            "streaming", _noop_send, room_id="room-1", stream_send_fn=_noop_stream_send
        )
        try:
            call = asyncio.ensure_future(kit.process_inbound(_keyed("k"), room_id="room-1"))
            await asyncio.wait_for(ws.read_all.wait(), 5)
            assert await kit.cancel_delivery("k") == 1
            with contextlib.suppress(Exception):
                await asyncio.wait_for(call, 5)

            assert ws.cuts == []
        finally:
            await kit.close()

    async def test_a_cut_whose_text_could_not_be_stored_still_ends_the_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Storing what a cut leaves failed: the streaming connection still
        reads the stream's end, with no text (RMK-1756)."""
        stored = SegmentWriter.end_cancelled

        async def fails(self: SegmentWriter, reader: Any, reason: str | None = None) -> None:
            await stored(self, reader, reason)
            raise RuntimeError("store down")

        monkeypatch.setattr(SegmentWriter, "end_cancelled", fails)
        ws = WebSocketChannel("ws")
        kit = await _cut_room(_Stalls(), ws)
        streamed: list[str] = []
        shown = asyncio.Event()

        async def stream_send(conn_id: str, msg: StreamMessage) -> None:
            streamed.append(f"end {_row(msg.event)!r}" if isinstance(msg, StreamEnd) else msg.type)
            if isinstance(msg, StreamChunk):
                shown.set()

        ws.register_connection(
            "streaming", _noop_send, room_id="room-1", stream_send_fn=stream_send
        )
        try:
            call = asyncio.ensure_future(kit.process_inbound(_keyed("k"), room_id="room-1"))
            await asyncio.wait_for(shown.wait(), 5)
            assert await kit.cancel_delivery("k") == 1
            with contextlib.suppress(Exception):
                await asyncio.wait_for(call, 5)

            assert streamed == ["stream_start", "stream_chunk", "end ''"]
        finally:
            await kit.close()

    async def test_unregister_clears_stream_send_fn(self) -> None:
        """unregister_connection removes stream_send_fn too."""
        ws = WebSocketChannel("ws1")
        ws.register_connection(
            "c1", _noop_send, room_id="room-1", stream_send_fn=_noop_stream_send
        )
        assert ws.supports_streaming_delivery is True

        ws.unregister_connection("c1")
        assert ws.supports_streaming_delivery is False
