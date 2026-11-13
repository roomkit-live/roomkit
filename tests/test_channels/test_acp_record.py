"""An ACP callback writes its turn's record without borrowing a local AI loop."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import acp
import pytest
from acp.schema import PromptResponse

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.models.channel import ChannelOutput
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.streaming import ToolCallStartMarker
from roomkit.providers.ai.base import ProviderError
from roomkit.tools import ToolCallContext, current_response_metadata, tool_turn_context
from roomkit.tools.context import current_tool_chain_depth
from tests.conftest import make_event
from tests.test_channels.test_acp import _binding, _channel, _context, _RecordingToolHandler
from tests.test_framework import SimpleChannel
from tests.test_tool_turn_context import _realtime


class _RecordingContext(_RecordingToolHandler):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[ResponseMetadata | None] = []
        self.depths: list[int] = []
        self.nested: ResponseMetadata | None = None

    async def on_tool_result(
        self, tool_name: str, tool_input: dict[str, Any], result: str, **kwargs: Any
    ) -> None:
        self._record()

    async def on_tool_cancelled(
        self, tool_name: str, tool_input: dict[str, Any], **kwargs: Any
    ) -> None:
        self._record()

    def _record(self) -> None:
        self.seen.append(current_response_metadata())
        self.depths.append(current_tool_chain_depth())
        with tool_turn_context(room_id="nested"):
            self.nested = current_response_metadata()
            assert self.nested is not None
            self.nested["nested"] = True
        # A realtime call has no response record, even inside an ACP observer.
        with _realtime(ToolCallContext(room_id="r1", tool_call_id="voice", channel_id="voice")):
            assert current_response_metadata() is None
            with tool_turn_context(room_id="nested"):
                nested = current_response_metadata()
                assert nested is not None and nested is not self.seen[0]
            assert current_response_metadata() is None
        self.seen.append(current_response_metadata())
        if self.seen[-1] is not None:
            self.seen[-1]["observed"] = True


@pytest.mark.parametrize("ending", ["complete", "error", "cancelled", "abandoned"])
async def test_an_acp_callback_keeps_its_record_across_a_nested_local_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    handler = _RecordingContext()
    channel, connection, _ = _channel(tmp_path, handler=handler)
    if ending != "complete":
        _leave_open_call(connection, monkeypatch, ending)
    try:
        with tool_turn_context(room_id="parent", chain_depth=3):
            parent = current_response_metadata()
            assert parent is not None
            parent["parent"] = True
            output = await channel.on_event(make_event(), _binding(), _context())
            assert output.response_stream is not None

            # The consumer and its callbacks inherit the parent's context.
            consumed = asyncio.wait_for(asyncio.create_task(_consume(output, ending)), 5)
            if ending == "error":
                with pytest.raises(ProviderError, match="agent failed"):
                    await consumed
            else:
                await consumed
            assert current_response_metadata() is parent
            assert dict(parent) == {"parent": True}

        assert current_response_metadata() is None
        assert len(handler.seen) == 2
        assert all(record is output.response_metadata for record in handler.seen)
        assert output.response_metadata["observed"] is True
        assert "nested" not in output.response_metadata
        assert handler.nested is not output.response_metadata
        assert handler.depths == [3]
    finally:
        await channel.close()


async def _consume(output: ChannelOutput, ending: str) -> None:
    assert output.response_stream is not None
    if ending == "abandoned":
        marker = await anext(output.response_stream)
        assert isinstance(marker, ToolCallStartMarker)
        await asyncio.create_task(output.response_stream.aclose())
        return
    async for _ in output.response_stream:
        pass


def _leave_open_call(connection: Any, monkeypatch: pytest.MonkeyPatch, ending: str) -> None:
    async def prompt(session_id: str, prompt: Any, **kwargs: Any) -> PromptResponse:
        await connection.client.session_update(
            session_id,
            acp.start_tool_call("open", title="Read", status="in_progress"),
        )
        if ending == "error":
            raise RuntimeError("agent failed")
        if ending == "abandoned":
            await asyncio.Event().wait()
        return PromptResponse(stop_reason="cancelled")

    monkeypatch.setattr(connection, "prompt", prompt)


async def test_the_acp_response_observer_reads_the_record_its_tool_hook_wrote(
    tmp_path: Path,
) -> None:
    channel, _, _ = _channel(tmp_path)
    async with RoomKit() as kit:
        kit.register_channel(SimpleChannel("source"))
        kit.register_channel(channel)
        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "source")
        await kit.attach_channel("room-1", "acp-agent", category=ChannelCategory.INTELLIGENCE)
        records: dict[str, ResponseMetadata | None] = {}
        written, observed = asyncio.Event(), asyncio.Event()

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
        async def write(event: Any, context: Any) -> None:
            record = current_response_metadata()
            records["tool"] = record
            if record is not None:
                record["from_tool"] = True
            written.set()

        @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
        async def read(event: Any, context: Any) -> None:
            records["response"] = current_response_metadata()
            observed.set()

        await kit.process_inbound(
            InboundMessage(
                channel_id="source", sender_id="alice", content=TextContent(body="read")
            )
        )
        await asyncio.wait_for(asyncio.gather(written.wait(), observed.wait()), 5)
        assert records["tool"] is not None and records["response"] is records["tool"]
        assert records["response"]["from_tool"] is True
        assert current_response_metadata() is None
