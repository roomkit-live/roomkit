"""A received old prompt result must not acknowledge a replacement session.

Reusable public ACPChannel/ACPTransport regression. The old connection has
already obtained its response, but delays returning it while a different
room discovers the dead pipe and reconnects. No old-wire message is invented
following close. New connections deliberately reuse their per-room IDs.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from acp.schema import NewSessionResponse, PromptResponse

from roomkit import ACPChannel, ACPTransport, Room, RoomContext
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from tests.conftest import make_event
from tests.test_channels.test_acp import _binding, _FakeACPConnection


class ReceivedResultConnection(_FakeACPConnection):
    def __init__(self, client: Any, transport: ReconnectingTransport, generation: int) -> None:
        super().__init__(client, emit_updates=False)
        self.transport = transport
        self.generation = generation

    async def new_session(self, **kwargs: Any) -> NewSessionResponse:
        self.new_session_calls.append(kwargs)
        return NewSessionResponse(
            session_id=f"session-{kwargs['roomkit.live/roomId']}",
            config_options=list(self.config_options),
        )

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> Any:
        await super().prompt(session_id, prompt, **kwargs)
        response = PromptResponse(stop_reason="end_turn")
        if self.generation == 0:
            self.transport.received.set()
            await self.transport.return_received.wait()
        return response


class ReconnectingTransport(ACPTransport):
    def __init__(self, *, reuse_connection: bool) -> None:
        self.reuse_connection = reuse_connection
        self.connections: list[ReceivedResultConnection] = []
        self.alive = False
        self.received = asyncio.Event()
        self.return_received = asyncio.Event()

    @property
    def name(self) -> str:
        return "received-result-reconnect"

    async def open(self, client: Any, *, queue: Any) -> ReceivedResultConnection:
        if self.reuse_connection and self.connections:
            connection = self.connections[0]
            connection.generation = len(self.connections)
            connection.client = client
        else:
            connection = ReceivedResultConnection(client, self, len(self.connections))
        self.connections.append(connection)
        self.alive = True
        return connection

    async def close(self) -> None:
        self.alive = False

    def is_alive(self) -> bool:
        return self.alive


def context(room: str, *events: Any) -> RoomContext:
    return RoomContext(
        room=Room(id=room),
        bindings=[
            _binding(room),
            ChannelBinding(channel_id="ch1", room_id=room, channel_type=ChannelType.SMS),
        ],
        recent_events=list(events),
        history_start_index=0,
    )


async def prompt(channel: ACPChannel, event: Any, *history: Any) -> None:
    output = await channel.on_event(
        event, _binding(event.room_id), context(event.room_id, *history)
    )
    assert output.response_stream is not None
    async for _ in output.response_stream:
        pass
    assert output.response_metadata["acp"]["prompt_returned"] is True
    assert "stop_reason" not in output.response_metadata["acp"]
    assert not output.response_metadata["acp"].get("interrupted")


@pytest.mark.asyncio
@pytest.mark.parametrize("open_replacement_before_old_ack", [False, True])
@pytest.mark.parametrize("reuse_connection", [False, True])
async def test_a_received_old_answer_does_not_acknowledge_the_new_connection(
    tmp_path: Any, open_replacement_before_old_ack: bool, reuse_connection: bool
) -> None:
    transport = ReconnectingTransport(reuse_connection=reuse_connection)
    channel = ACPChannel(
        "acp-agent", cwd=tmp_path, transport=transport, instructions="Standing policy"
    )
    first = make_event(room_id="room-a", body="old request", index=0)
    old = asyncio.create_task(prompt(channel, first, first))
    try:
        await asyncio.wait_for(transport.received.wait(), 10)
        transport.alive = False
        other = make_event(room_id="room-b", body="other room", index=0)
        await asyncio.wait_for(prompt(channel, other, other), 10)
        assert len(transport.connections) == 2
        assert (transport.connections[0] is transport.connections[1]) == reuse_connection
        if open_replacement_before_old_ack:
            await channel.set_config_option("room-a", "model", "sonnet")
            assert channel.session_id("room-a") == "session-room-a"
        transport.return_received.set()
        await asyncio.wait_for(old, 10)
        second = make_event(room_id="room-a", body="new request", index=1)
        await asyncio.wait_for(prompt(channel, second, first, second), 10)
        sent = transport.connections[1].prompt_calls[-1]["prompt"][0].text
        assert "old request" in sent, sent
        assert "Standing policy" in sent, sent
        assert sent.endswith("new request")
    finally:
        transport.return_received.set()
        if not old.done():
            old.cancel()
        await asyncio.gather(old, return_exceptions=True)
        await channel.close()
