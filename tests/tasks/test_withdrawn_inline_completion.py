"""A withdrawn call's cleanup does not delay a normally completed turn."""

from __future__ import annotations

import asyncio

import pytest

from roomkit.tasks import DelegateHandler, setup_delegation
from roomkit.tasks.models import TaskStatus
from tests.bounds import CUT, expire, opened_with_cut
from tests.tasks.test_delegate import _Held, _hold_child_terminal_write, _kit_delegating_in_turn


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_normal_turn_does_not_join_a_withdrawn_inline_completion(
    streaming: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    kit, researcher = _kit_delegating_in_turn(_Held(started), tool_timeouts={"delegate_task": CUT})
    researcher._provider._streaming = streaming
    _hold_child_terminal_write(kit, cleaning, release, monkeypatch)
    setup_delegation(researcher, DelegateHandler(kit, wait_in_task_rooms=True))
    async with asyncio.timeout(5):
        bounds = opened_with_cut(monkeypatch)
        await kit.create_room(room_id="call")
        task = await kit.delegate("call", "researcher", "Look it up.")
        await started.wait()
        expire(bounds)
        await cleaning.wait()
        waiting = asyncio.create_task(task.wait())
        try:
            done, _ = await asyncio.wait({waiting}, timeout=0.05)
        finally:
            release.set()
            ended = await waiting
            await kit.close()
        assert done, f"streaming={streaming}: normal turn waited for withdrawn child cleanup"
        assert ended.status == TaskStatus.COMPLETED
        assert "drainage incomplete" not in caplog.text
