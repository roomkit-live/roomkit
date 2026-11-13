"""External permission services are owned by the ACP channel, not a session."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import acp
import pytest

from roomkit import ACPChannel, HookResult, HookTrigger, RoomKit
from roomkit.tools import PolicyExternalToolHandler
from tests.test_channels.test_acp import _FakeACPConnection, _FakeTransport
from tests.test_channels.test_acp_decided_calls import _OPTIONS


class Service(PolicyExternalToolHandler):
    def __init__(self, *, start_failures: int = 0, stop_fails: bool = False) -> None:
        super().__init__()
        self.starts = 0
        self.stops = 0
        self.start_failures = start_failures
        self.stop_fails = stop_fails
        self.connection: Any = None
        self.transport: Any = None

    async def start(self) -> None:
        assert self.connection.initialize_calls
        assert self.connection.authenticated_with == "test-auth"
        self.starts += 1
        if self.starts <= self.start_failures:
            raise ValueError("approval service down")

    async def stop(self) -> None:
        assert not self.transport.is_alive()
        self.stops += 1
        if self.stop_fails:
            raise RuntimeError("service cleanup failed")


def channel(tmp_path: Path, service: Service) -> ACPChannel:
    connection = _FakeACPConnection(None, emit_updates=False)
    transport = _FakeTransport(connection)
    result = ACPChannel(
        "acp-agent",
        transport=transport,
        cwd=tmp_path,
        authentication_method="test-auth",
        external_tool_handler=service,
    )
    service.connection, service.transport = connection, transport
    connection.client = result._client
    return result


async def test_registration_and_unused_close_do_not_start_a_handler(tmp_path: Path) -> None:
    service = Service()
    agent = channel(tmp_path, service)
    assert service.channel_id == ""
    kit = RoomKit()
    kit.register_channel(agent)
    assert service.channel_id == "acp-agent"
    assert service.starts == 0
    await kit.close()
    assert (service.starts, service.stops) == (0, 0)


async def test_a_permission_during_graceful_close_keeps_its_room(tmp_path: Path) -> None:
    service = Service()
    agent = channel(tmp_path, service)
    kit = RoomKit()
    kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    seen: list[str] = []

    @kit.hook(HookTrigger.BEFORE_TOOL_USE)
    async def deny(event: Any, context: Any) -> HookResult:
        seen.append(context.room.id)
        return HookResult.block("no reads")

    await agent.set_config_option("r1", "model", "opus")
    session = agent.session_id("r1")
    closing, resume = asyncio.Event(), asyncio.Event()

    async def close_session(session_id: str) -> None:
        assert session_id == session
        closing.set()
        await resume.wait()

    service.connection.close_session = close_session
    task = asyncio.create_task(agent.close())
    try:
        await asyncio.wait_for(closing.wait(), 3)
        request = acp.update_tool_call("read", title="Read", raw_input={})
        answer = await service.connection.client.request_permission(session, request, _OPTIONS)
        assert answer.outcome.option_id == "r"
        assert seen == ["r1"]
    finally:
        resume.set()
        await asyncio.wait_for(task, 3)
        await kit.close()


async def test_reconnect_keeps_the_started_service_and_close_stops_it_once(tmp_path: Path) -> None:
    service = Service()
    agent = channel(tmp_path, service)
    await agent.set_config_option("r1", "model", "opus")
    first = agent.session_id("r1")
    await agent.set_config_option("r2", "model", "opus")
    assert service.starts == 1
    service.transport.alive = False
    await agent.set_config_option("r1", "model", "opus")
    assert agent.session_id("r1") != first
    assert service.starts == 1
    await agent.close()
    await agent.close()
    assert service.stops == 1


async def test_failed_handler_start_closes_transport_and_can_be_retried(tmp_path: Path) -> None:
    service = Service(start_failures=1)
    agent = channel(tmp_path, service)
    with pytest.raises(ValueError, match="approval service down"):
        await agent.set_config_option("r1", "model", "opus")
    assert not service.transport.is_alive()
    assert service.stops == 0
    await agent.set_config_option("r1", "model", "opus")
    assert service.starts == 2
    await agent.close()
    assert service.stops == 1


async def test_failed_handler_start_is_not_stopped(tmp_path: Path) -> None:
    service = Service(start_failures=1)
    agent = channel(tmp_path, service)
    with pytest.raises(ValueError, match="approval service down"):
        await agent.set_config_option("r1", "model", "opus")
    await agent.close()
    assert (service.starts, service.stops) == (1, 0)


async def test_handler_stop_failure_does_not_fail_or_repeat_close(tmp_path: Path) -> None:
    service = Service(stop_fails=True)
    agent = channel(tmp_path, service)
    await agent.set_config_option("r1", "model", "opus")
    await agent.close()
    await agent.close()
    assert (service.starts, service.stops) == (1, 1)


async def test_failed_authentication_never_starts_the_handler(tmp_path: Path) -> None:
    service = Service()
    agent = channel(tmp_path, service)

    async def fail(method_id: str) -> None:
        raise ValueError("authentication failed")

    service.connection.authenticate = fail
    with pytest.raises(ValueError, match="authentication failed"):
        await agent.set_config_option("r1", "model", "opus")
    await agent.close()
    assert not service.transport.is_alive()
    assert (service.starts, service.stops) == (0, 0)
