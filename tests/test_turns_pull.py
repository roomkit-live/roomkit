"""How a turn's stream is pulled across the turn backend (RFC §10.6.4): the
executor reads deltas ahead of the owner's reader, never past any other item
before the reader read it; a cut closes the channel's stream wherever it
finds the stream; text bursts cross whole, or refuse the turn."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit.core._turn_scope import current_turn_id
from roomkit.core.exceptions import TurnLinkError
from roomkit.models.steering import Cancel
from roomkit.models.streaming import (
    SegmentBreakMarker,
    ThinkingDeltaMarker,
    ToolCallStartMarker,
    UsageMarker,
)
from roomkit.turns import InMemoryTurnBackend, TurnClient, TurnExecutor, memory
from roomkit.turns._codec import encode_item
from roomkit.turns._frame_sender import FrameSender
from roomkit.turns.wire import Item
from tests.test_turns_backend import ScriptedChannel, _client, _run


@pytest.mark.parametrize("deltas", [10, 40])
async def test_a_call_never_runs_before_its_reader_read_its_start(deltas: int) -> None:
    """Past half its window the owner pulls again while the executor may
    already hold the call's start: that pull grants nothing past it."""
    ran: list[str] = []

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        for i in range(deltas):
            yield f"w{i} "
        yield ToolCallStartMarker(tool_name="t", tool_id="c1", arguments={})
        ran.append("the call ran")
        yield "after"

    client = _client(ScriptedChannel(script))
    stream = (await _run(client)).response_stream
    async for item in stream:
        if isinstance(item, ToolCallStartMarker):
            break  # the reader stops at the call
    await asyncio.sleep(0.05)
    await stream.aclose()
    await client.close()

    assert ran == []


def _closing_script(log: list[str]) -> Any:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        try:
            yield "text"
            yield SegmentBreakMarker()  # past it, the executor waits for a pull
            yield "never"
        finally:
            log.append(f"closed under {current_turn_id()}")

    return script


async def test_a_cut_while_the_executor_waits_for_a_pull_closes_the_stream_in_the_turn() -> None:
    """A steering Cancel to a channel without steering cuts the turn: its
    stream is closed under the turn, not left to the garbage collector."""
    log: list[str] = []
    client = _client(ScriptedChannel(_closing_script(log)))
    output = await _run(client)
    stream = output.response_stream
    assert await anext(stream) == "text"
    assert isinstance(await anext(stream), SegmentBreakMarker)
    await asyncio.sleep(0.05)

    assert await client.steer("room", "agent", Cancel()) == 1
    assert [item async for item in stream] == []
    turn_id = client._started[0].request.turn_id
    await client.close()

    assert log == [f"closed under {turn_id}"]


async def test_an_owner_lost_while_the_executor_waits_for_a_pull_closes_the_stream() -> None:
    log: list[str] = []
    channels = {"agent": ScriptedChannel(_closing_script(log))}
    backend = InMemoryTurnBackend()
    executor = TurnExecutor(channels.get, hosts=channels.__contains__)
    await backend.serve(executor)
    client = TurnClient(backend)
    stream = (await _run(client)).response_stream
    assert await anext(stream) == "text"
    assert isinstance(await anext(stream), SegmentBreakMarker)
    await asyncio.sleep(0.05)

    await client.close()  # the owner goes
    for _ in range(50):
        if log:
            break
        await asyncio.sleep(0.01)
    await executor.close()

    assert len(log) == 1 and log[0].startswith("closed under turn-")


async def test_a_text_delta_no_frame_can_carry_refuses_the_turn() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "Hello "
        await asyncio.sleep(0.01)
        yield "bad \ud800 text "
        yield "world."

    client = _client(ScriptedChannel(script))
    stream = (await _run(client)).response_stream
    read: list[str] = []
    with pytest.raises(TypeError, match="stream item cannot cross"):
        async for item in stream:
            read.append(item)
    await client.close()

    assert read == ["Hello "]


async def test_text_kept_while_a_send_waits_crosses_without_another_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over a link whose send waits (a network write), a delta kept while
    the burst before it was being sent goes as soon as that send ends."""
    send = memory._ExecutorLink.send

    async def network_send(self: Any, frame: Any) -> int:
        await asyncio.sleep(0.005)
        return await send(self, frame)

    monkeypatch.setattr(memory._ExecutorLink, "send", network_send)

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        for i in range(5):
            yield f"t{i} "
            await asyncio.sleep(0.002)
        await asyncio.Event().wait()  # then the model thinks

    client = _client(ScriptedChannel(script))
    stream = (await _run(client)).response_stream
    started, text = time.monotonic(), ""
    while "t4" not in text:
        text += await asyncio.wait_for(anext(stream), 1.0)
    took = time.monotonic() - started
    await stream.aclose()
    await client.close()

    assert text == "t0 t1 t2 t3 t4 "
    assert took < 1.0  # not carried by the next heartbeat


async def test_the_record_a_reader_reads_is_the_one_sent_before_its_item() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "a"
        channel.record["x"] = 1
        yield "b"
        yield "b2"
        yield SegmentBreakMarker()
        channel.record["x"] = 2
        yield "c"
        channel.record["x"] = 3

    client = _client(ScriptedChannel(script))
    output = await _run(client)
    seen = []
    async for item in output.response_stream:
        await asyncio.sleep(0.02)  # a slow reader: the turn's end comes first
        name = item if isinstance(item, str) else type(item).__name__
        seen.append((name, output.response_metadata.get("x")))
    seen.append(("end", output.response_metadata.get("x")))
    await client.close()

    assert seen == [
        ("a", None),
        ("bb2", 1),
        ("SegmentBreakMarker", 1),
        ("c", 2),
        ("end", 3),
    ]


async def test_a_burst_naming_no_positive_number_of_deltas_is_an_invalid_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def foreign_burst(self: FrameSender, texts: list[str]) -> None:
        if texts:
            await self._send(Item(item={"kind": "text", "text": "".join(texts), "deltas": "two"}))

    monkeypatch.setattr(FrameSender, "_send_text", foreign_burst)

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "a"
        yield "b"
        await asyncio.Event().wait()

    client = _client(ScriptedChannel(script))
    stream = (await _run(client)).response_stream
    with pytest.raises(TurnLinkError) as raised:
        await anext(stream)
    [turn] = client._started

    assert raised.value.reason == "frame_invalid"
    assert turn.over  # the turn ends, its reader told why
    await client.close()


async def test_what_a_generation_used_waits_on_no_link_and_keeps_its_place() -> None:
    sent: list[dict[str, object]] = []
    link_open = asyncio.Event()

    async def link(frame: Any) -> int:
        await link_open.wait()
        sent.append(frame.item)
        return len(sent)

    frames = FrameSender(link)
    frames.add_text("a")
    frames.add_text("b")
    # No await: the loop goes on to decide its round whatever the link does.
    frames.add_item(Item(item=encode_item(UsageMarker(usage={"input_tokens": 1}))))
    frames.add_text("c")
    await asyncio.sleep(0)
    link_open.set()
    await frames.send(Item(item={"kind": "stream_end"}))

    assert [item.get("text") or item["kind"] for item in sent] == [
        "ab",
        "usage",
        "c",
        "stream_end",
    ]


class _DirectChannel(ScriptedChannel):
    """A channel whose stream is its script's generator itself: closing the
    stream runs the script's own cleanup, awaits included."""

    def _stream(self) -> AsyncIterator[Any]:  # type: ignore[override]
        return self.script(self)


async def _ends_its_text(channel: ScriptedChannel) -> AsyncIterator[Any]:
    yield "a"
    await asyncio.sleep(0)  # a burst per delta
    yield "b"
    channel.record["final"] = True  # how the turn ended


@pytest.mark.parametrize("stop", ["close", "drop"])
async def test_a_reader_that_stops_after_its_turn_ended_reads_the_turn_s_last_record(
    stop: str,
) -> None:
    """Text only: the executor reads to the end and its end comes before the
    reader stops; the record the reader then holds is still the last one."""
    client = _client(ScriptedChannel(_ends_its_text))
    output = await _run(client)
    stream = output.response_stream
    assert await anext(stream) == "a"
    await asyncio.sleep(0.05)
    if stop == "close":
        await stream.aclose()
    else:
        stream.drop()
        await asyncio.sleep(0.01)
    await client.close()

    assert output.response_metadata.get("final") is True


async def test_a_steering_cut_s_last_record_reaches_its_reader() -> None:
    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        try:
            yield "text"
            yield SegmentBreakMarker()
            yield "never"
        finally:
            channel.record["interrupted"] = True

    client = _client(_DirectChannel(script))
    output = await _run(client)
    stream = output.response_stream
    assert await anext(stream) == "text"
    assert isinstance(await anext(stream), SegmentBreakMarker)
    await asyncio.sleep(0.02)
    assert await client.steer("room", "agent", Cancel()) == 1
    await asyncio.sleep(0.05)  # the turn's end comes before the reader reads on

    assert [item async for item in stream] == []
    assert output.response_metadata.get("interrupted") is True
    await client.close()


@pytest.mark.parametrize("second", ["reader_cancelled", "second_steer"])
async def test_a_second_cut_lets_the_first_cut_s_close_run_the_stream_s_cleanup(
    second: str,
) -> None:
    log: list[str] = []

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        try:
            yield "text"
            yield SegmentBreakMarker()
            yield "never"
        finally:
            await asyncio.sleep(0.05)  # a session closed, a span ended
            log.append(f"cleanup done under {current_turn_id()}")

    client = _client(_DirectChannel(script))
    stream = (await _run(client)).response_stream
    assert await anext(stream) == "text"
    assert isinstance(await anext(stream), SegmentBreakMarker)
    await asyncio.sleep(0.02)
    turn_id = client._started[0].request.turn_id
    await client.steer("room", "agent", Cancel())  # the cut closes the stream
    await asyncio.sleep(0.01)
    if second == "reader_cancelled":
        reader = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0.005)
        reader.cancel("kit_closed")
        await asyncio.gather(reader, return_exceptions=True)
    else:
        await client.steer("room", "agent", Cancel())
    async with asyncio.timeout(1):  # the first cut's close runs the cleanup
        while not log:
            await asyncio.sleep(0.005)
    await client.close()  # a second cleanup, were one run, would be logged by then

    assert log == [f"cleanup done under {turn_id}"]


async def test_a_burst_the_link_failed_to_carry_fails_the_turn_rather_than_stall_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its deltas were counted read on the executor and never reached the
    reader: no later pull could grant past the next call's start."""
    send = memory._ExecutorLink.send
    failed: list[Any] = []

    async def flaky(self: Any, frame: Any) -> int:
        lost = getattr(frame, "type", None) == "item" and frame.item.get("text") == "lost "
        if lost and not failed:
            failed.append(frame)
            raise ConnectionError("the link hiccupped")
        return await send(self, frame)

    monkeypatch.setattr(memory._ExecutorLink, "send", flaky)

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "first "
        await asyncio.sleep(0.01)
        yield "lost "
        await asyncio.sleep(0.01)  # the burst's flush fails
        yield SegmentBreakMarker()
        yield "after"

    client = _client(ScriptedChannel(script))
    stream = (await _run(client)).response_stream
    read: list[Any] = []
    with pytest.raises(TurnLinkError):
        async with asyncio.timeout(2.0):
            async for item in stream:
                read.append(item)
    await client.close()

    assert failed and read == ["first "]


async def test_a_pull_sent_while_the_send_of_a_break_waits_for_its_ack_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A link whose send delivers the frame, then waits for its ack: the
    reader reads the break and pulls before the executor's send returns.
    That pull came after the reader read the break: the stream reads on."""
    send = memory._ExecutorLink.send

    async def acked_send(self: Any, frame: Any) -> int:
        seq = await send(self, frame)
        await asyncio.sleep(0.01)  # the ack
        return seq

    monkeypatch.setattr(memory._ExecutorLink, "send", acked_send)

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        yield "a"
        yield SegmentBreakMarker()
        yield "b"

    client = _client(ScriptedChannel(script))
    stream = (await _run(client)).response_stream
    async with asyncio.timeout(2.0):
        read = [item async for item in stream]
    await client.close()

    assert read[0] == "a" and isinstance(read[1], SegmentBreakMarker) and read[2:] == ["b"]


@pytest.mark.parametrize("deltas", [1, 32, 33, 64, 65, 130])
async def test_deltas_bursts_and_thinking_before_each_break_never_stall_the_stream(
    deltas: int,
) -> None:
    """The reader and the executor count the same items however the deltas
    cross (one by one, joined, reasoning apart): every pull is honoured."""

    async def script(channel: ScriptedChannel) -> AsyncIterator[Any]:
        for i in range(deltas):
            yield f"t{i} "
            if i % 3 == 0:
                yield ThinkingDeltaMarker(thinking="x")
            if i % 7 == 0:
                await asyncio.sleep(0)
        yield SegmentBreakMarker()
        for i in range(deltas):
            yield f"u{i} "
        yield ToolCallStartMarker(tool_name="t", tool_id="c", arguments={})
        yield "z"

    client = _client(ScriptedChannel(script))
    stream = (await _run(client)).response_stream
    read: list[Any] = []
    async with asyncio.timeout(3.0):
        async for item in stream:
            read.append(item)
            if len(read) % 5 == 0:
                await asyncio.sleep(0.001)
    await client.close()

    assert read[-1] == "z"
