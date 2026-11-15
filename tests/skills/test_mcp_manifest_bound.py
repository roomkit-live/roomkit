"""Served manifest changes are checked before registry or catalog writes."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from tests.mcp_skill_fakes import FakeSkillServer, ServedSkill, skill_md
from tests.skills.test_manifest_bound import LIMIT

from roomkit.skills import SkillManifestTooLargeError, SkillRegistry
from roomkit.skills.models import Skill, SkillMetadata


def _skill(name: str = "hidden", *, dynamic: bool = False) -> ServedSkill:
    return ServedSkill(name, {"SKILL.md": skill_md(name, "d" * 1024)}, dynamic=dynamic)


def _fill(registry: SkillRegistry, *, remaining: int = 0) -> None:
    registry.mark_unavailable("padding", "")
    registry.mark_unavailable("padding", "x" * (LIMIT - remaining - len(registry.to_prompt_xml())))


async def test_a_new_label_counts_aliased_names_outside_its_listing(tmp_path: Path) -> None:
    registry = SkillRegistry()
    skill = Skill(SkillMetadata(name="local", description="small"), "body", tmp_path)
    registry.add(skill)
    skill.metadata.name = "hf:ghost"
    _fill(registry)
    before = registry.to_prompt_xml()
    with pytest.raises(SkillManifestTooLargeError) as raised:
        await registry.add_mcp_server(FakeSkillServer(), "hf")
    assert raised.value.characters > LIMIT
    assert registry.to_prompt_xml() == before
    assert not registry.has_served
    assert registry.get_metadata("local") is skill.metadata


@pytest.mark.parametrize("kind", ["available", "dynamic", "malformed"])
async def test_oversized_first_listing_registers_neither_skills_nor_server(kind: str) -> None:
    server = FakeSkillServer([_skill(f"skill-{i}", dynamic=kind == "dynamic") for i in range(800)])
    if kind == "malformed":
        server.entry_overrides = {skill.uri: {"uri": skill.uri} for skill in server.skills}
    registry = SkillRegistry()
    with pytest.raises(SkillManifestTooLargeError):
        await registry.add_mcp_server(server, "hf", execution_tools=["exec_*"])
    assert registry.to_prompt_xml() == ""
    assert registry.skill_names == []
    assert registry.unavailable_skills == {}
    assert not registry.has_served
    assert registry.origin("hf:hidden") is None
    assert registry.execution_tool_patterns == frozenset()
    assert await registry.load_skill("hf:hidden") is None
    assert server.reads == []
    assert all(method == "skills/list" for method, _ in server.requests)


async def test_rejected_relisting_keeps_old_entries_marks_and_shared_origin() -> None:
    server = FakeSkillServer([_skill("old")])
    registry = SkillRegistry()
    await registry.add_mcp_server(server, "hf", execution_tools=["old_*"])
    loaded = await registry.load_skill("hf:old")
    registry.mark_unlisted("hf:old")
    copied = registry.copy(["hf:old"])
    before = registry.to_prompt_xml()
    origin = registry.origin("hf:old")
    server.skills = [_skill(f"skill-{i}") for i in range(70)]

    with pytest.raises(SkillManifestTooLargeError):
        await registry.add_mcp_server(server, "hf", execution_tools=["new_*"])

    assert registry.to_prompt_xml() == before
    assert registry.skill_names == ["hf:old"]
    assert registry.listed_names == []
    assert registry.origin("hf:old") is origin
    assert registry.execution_tool_patterns == frozenset({"old_*"})
    assert registry.get_skill("hf:old") is loaded
    assert copied.get_skill("hf:old") is loaded


@pytest.mark.parametrize("excess", [0, 1])
async def test_new_origin_is_counted_with_its_final_quoted_xml(excess: int) -> None:
    server = FakeSkillServer([_skill()])
    measured = SkillRegistry()
    await measured.add_mcp_server(server, "hf")
    _fill(measured)
    reason = measured.get_unavailable_reason("padding")
    assert reason is not None
    registry = SkillRegistry()
    registry.mark_unavailable("padding", reason + "x" * excess)
    before = registry.to_prompt_xml()
    if excess:
        with pytest.raises(SkillManifestTooLargeError) as raised:
            await registry.add_mcp_server(server, "hf")
        assert raised.value.characters == LIMIT + 1
        assert registry.to_prompt_xml() == before
        assert not registry.has_served
    else:
        await registry.add_mcp_server(server, "hf")
        assert len(registry.to_prompt_xml()) == LIMIT


async def test_fetched_available_skill_is_hidden_before_measuring() -> None:
    server = FakeSkillServer([_skill()], listed=set())
    registry = SkillRegistry()
    await registry.add_mcp_server(server, "hf")
    _fill(registry)
    before = registry.to_prompt_xml()
    assert await registry.load_skill("hf:hidden") is not None
    assert registry.to_prompt_xml() == before
    assert registry.listed_names == []
    assert registry.skill_names == ["hf:hidden"]


async def test_rejected_fetch_does_not_modify_the_shared_catalog() -> None:
    server = FakeSkillServer([_skill()], listed=set())
    registry = SkillRegistry()
    await registry.add_mcp_server(server, "hf")
    copied = registry.copy()
    loaded = await copied.load_skill("hf:hidden")
    _fill(registry)
    before = registry.to_prompt_xml()
    server.skills = [_skill(dynamic=True)]

    with pytest.raises(SkillManifestTooLargeError):
        await registry.load_skill("hf:hidden")

    assert registry.to_prompt_xml() == before
    assert registry.skill_names == []
    assert copied.get_skill("hf:hidden") is loaded
    assert registry.get_unavailable_reason("hf:hidden") is None


@pytest.mark.parametrize("operation", ["get", "load", "read"])
async def test_following_a_shared_refusal_fails_closed_without_partial_local_changes(
    operation: str,
) -> None:
    server = FakeSkillServer([_skill()])
    registry = SkillRegistry()
    await registry.add_mcp_server(server, "hf")
    await registry.load_skill("hf:hidden")
    copied = registry.copy()
    copied.mark_unlisted("hf:hidden")
    _fill(copied)
    before = copied.to_prompt_xml()
    server.skills = [_skill(dynamic=True)]
    await registry.add_mcp_server(server, "hf")
    reads = list(server.reads)

    for _ in range(2):
        with pytest.raises(SkillManifestTooLargeError):
            if operation == "get":
                copied.get_skill("hf:hidden")
            elif operation == "load":
                await copied.load_skill("hf:hidden")
            else:
                await copied.read_file("hf:hidden", "SKILL.md")
        assert copied.to_prompt_xml() == before
        assert copied.skill_names == ["hf:hidden"]
        assert copied.get_unavailable_reason("hf:hidden") is None
        assert server.reads == reads
    copied.mark_unavailable("padding", "short")
    assert await copied.load_skill("hf:hidden") is None
    assert copied.get_unavailable_reason("hf:hidden") is not None


@pytest.mark.parametrize("operation", ["listing", "fetch"])
async def test_remote_reads_check_current_state_after_io(operation: str) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    class WaitingServer(FakeSkillServer):
        async def request(
            self, method: str, params: Mapping[str, Any] | None = None
        ) -> dict[str, Any]:
            if method == ("skills/list" if operation == "listing" else "skills/get"):
                started.set()
                await release.wait()
            return await super().request(method, params)

    server = WaitingServer([_skill(dynamic=operation == "fetch")])
    registry = SkillRegistry()
    if operation == "fetch":
        server.listed = set()
        await registry.add_mcp_server(server, "hf")
        pending = asyncio.create_task(registry.load_skill("hf:hidden"))
    else:
        pending = asyncio.create_task(registry.add_mcp_server(server, "hf"))
    try:
        await asyncio.wait_for(started.wait(), 1)
        _fill(registry)
        before = registry.to_prompt_xml()
        release.set()
        with pytest.raises(SkillManifestTooLargeError):
            await pending
        assert registry.to_prompt_xml() == before
        assert registry.has_served == (operation == "fetch")
        assert registry.get_metadata("hf:hidden") is None
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)


async def test_manifest_pressure_never_undoes_a_shared_verification_retirement() -> None:
    from roomkit.skills.mcp.catalog import SkillChangedError

    skill = _skill()
    skill.files["guide.md"] = "verified guide"
    server = FakeSkillServer([skill])
    registry = SkillRegistry()
    await registry.add_mcp_server(server, "hf")
    await registry.load_skill("hf:hidden")
    copied = registry.copy()
    copied.mark_unlisted("hf:hidden")
    _fill(copied)
    before = copied.to_prompt_xml()
    server.served[skill.file_uri("guide.md")] = "tampered"
    server.skills = [_skill(dynamic=True)]
    with pytest.raises(SkillChangedError):
        await registry.read_file("hf:hidden", "guide.md")
    reads = list(server.reads)
    with pytest.raises(SkillManifestTooLargeError):
        copied.get_skill("hf:hidden")
    assert copied.to_prompt_xml() == before
    assert server.reads == reads
    copied.mark_unavailable("padding", "short")
    assert await copied.load_skill("hf:hidden") is None
