"""A channel owns its sandbox teardown after ending its calls."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from roomkit import AIChannel, RoomKit, SandboxResult
from roomkit.providers.ai import MockAIProvider
from tests.test_sandbox_contract import RecordingSandbox, answer, channel_with_call


class ClosingSandbox(RecordingSandbox):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.order: list[str] = []

    async def execute(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> SandboxResult:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.order.append("call ended")
        return SandboxResult(exit_code=0)

    async def close(self) -> None:
        self.order.append("sandbox closed")


async def test_close_ends_the_active_command_before_releasing_the_sandbox(streaming: bool) -> None:
    sandbox = ClosingSandbox()
    channel, provider = channel_with_call(sandbox, streaming)
    turn = asyncio.create_task(answer(channel))
    try:
        await asyncio.wait_for(sandbox.started.wait(), 3)
        await asyncio.wait_for(channel.close(), 3)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(turn, 3)
        assert sandbox.order == ["call ended", "sandbox closed"]
        assert len(provider.calls) == 1
    finally:
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)


async def test_an_empty_catalogue_still_belongs_to_the_channel() -> None:
    sandbox = ClosingSandbox()
    sandbox.definitions = []
    channel = AIChannel("ai1", provider=MockAIProvider(), sandbox=sandbox)
    await channel.close()
    assert sandbox.order == ["sandbox closed"]


class FailingClose(RecordingSandbox):
    async def close(self) -> None:
        raise ConnectionError("sandbox cleanup failed")


async def test_direct_close_propagates_a_sandbox_cleanup_error() -> None:
    channel = AIChannel("ai1", provider=MockAIProvider(), sandbox=FailingClose())
    with pytest.raises(ConnectionError, match="sandbox cleanup failed"):
        await channel.close()


async def test_kit_close_collects_a_sandbox_cleanup_error() -> None:
    kit = RoomKit()
    kit.register_channel(AIChannel("ai1", provider=MockAIProvider(), sandbox=FailingClose()))
    with pytest.raises(ExceptionGroup) as caught:
        await kit.close()
    assert any(
        isinstance(error, ConnectionError) and str(error) == "sandbox cleanup failed"
        for error in caught.value.exceptions
    )


class OwnClose(ClosingSandbox):
    channel: AIChannel

    async def execute(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> SandboxResult:
        self.order.append("call entered")
        await self.channel.close()
        self.order.append("own close returned")
        return SandboxResult(exit_code=0, output="returned")


async def test_a_command_can_close_its_channel_without_awaiting_itself(streaming: bool) -> None:
    sandbox = OwnClose()
    channel, provider = channel_with_call(sandbox, streaming)
    sandbox.channel = channel
    run = await asyncio.wait_for(answer(channel), 3)
    assert sandbox.order == ["call entered", "sandbox closed", "own close returned"]
    assert len(provider.calls) == 1
    assert len(run.calls) == 1 and not run.calls[0].failed
