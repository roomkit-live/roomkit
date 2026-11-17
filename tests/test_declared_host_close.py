"""A declared owner callback keeps its own shutdown work until close returns."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookResult, HookTrigger, RoomEvent, TextContent, TurnHost
from tests.test_turns_host import _kit, _say


@pytest.mark.usefixtures("close_grace", "cut_unwind")
@pytest.mark.parametrize("through_hook", [False, True], ids=["direct", "send-event-hook"])
async def test_a_declared_callback_finishes_the_close_it_started(through_hook: bool) -> None:
    closed = asyncio.Event()
    stages: list[str] = []

    async def use(host: TurnHost) -> Any:
        return await host.call("app.close")

    kit, _, _ = await _kit(use)

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def closing_hook(event: RoomEvent, context: Any) -> HookResult:
        if getattr(event.content, "body", None) == "close owner":
            stages.append("hook entered")
            await kit.close()
            stages.append("hook returned")
        return HookResult.allow()

    async def owner(arguments: dict[str, Any]) -> str:
        stages.append("callback entered")
        try:
            if through_hook:
                try:
                    await kit.send_event("r1", "agent", TextContent(body="close owner"))
                except asyncio.CancelledError:
                    stages.append("send_event raised CancelledError")
                    raise
                except BaseException as error:
                    stages.append(f"send_event raised {type(error).__name__}")
                    raise
                stages.append("send_event returned")
            else:
                await kit.close()
                stages.append("close returned")
            return "closed"
        finally:
            stages.append("callback finished")
            closed.set()

    kit.declare_host_verb("app.close", owner)
    try:
        await asyncio.wait_for(asyncio.gather(_say(kit), return_exceptions=True), 5)
        try:
            await asyncio.wait_for(closed.wait(), 5)
        except TimeoutError:
            pytest.fail(f"declared close did not return: {stages}")
        expected = ["callback entered"]
        if through_hook:
            expected += ["hook entered", "hook returned", "send_event raised KitClosedError"]
        else:
            expected += ["close returned"]
        assert stages == [*expected, "callback finished"]
    finally:
        await kit.close()
