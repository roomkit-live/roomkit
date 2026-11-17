"""Concurrent served reads account against the live shared byte budget."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from tests.mcp_skill_fakes import FakeSkillServer, ServedSkill, skill_md

from roomkit.skills import SkillRegistry
from roomkit.skills.mcp.catalog import SkillChangedError
from roomkit.skills.mcp.entry import MAX_TOTAL_SIZE


async def test_dynamic_reads_check_bytes_committed_while_their_io_was_pending() -> None:
    started = {name: asyncio.Event() for name in ("first.txt", "second.txt")}
    release = {name: asyncio.Event() for name in started}
    size = MAX_TOTAL_SIZE // 2
    skill = ServedSkill(
        "guide",
        {
            "SKILL.md": skill_md("guide", "Guide", "Body"),
            "first.txt": "a" * size,
            "second.txt": "b" * size,
        },
        dynamic=True,
    )

    class WaitingServer(FakeSkillServer):
        async def read_resource(self, uri: str) -> Any:
            name = uri.rsplit("/", 1)[-1]
            if name in started:
                started[name].set()
                await release[name].wait()
            return await super().read_resource(uri)

    registry = SkillRegistry()
    await registry.add_mcp_server(WaitingServer([skill]), "docs", allow_dynamic=True)
    await registry.load_skill("docs:guide")
    first = asyncio.create_task(registry.read_file("docs:guide", "first.txt"))
    second = asyncio.create_task(registry.read_file("docs:guide", "second.txt"))
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), 1)
        release["first.txt"].set()
        assert len(await first) == size
        release["second.txt"].set()
        with pytest.raises(SkillChangedError):
            await second
        assert registry.get_skill("docs:guide") is None
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(first, second, return_exceptions=True)
