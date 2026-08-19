"""Skill tool gating actually gates (RFC §24.2).

`allowed_tools` entries are ToolPolicy globs. Two things broke that: the
frontmatter parser stringified a YAML list into its own repr, and both gating
sites compared tool names for exact membership. A skill written the way the
RFC's own example writes it therefore restricted nothing at all — silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AITool, AIToolCall
from roomkit.skills.models import SkillMetadata
from roomkit.skills.parser import parse_frontmatter
from roomkit.skills.registry import SkillRegistry
from roomkit.tools.policy import ToolPolicy, matches_any_pattern
from tests.test_skills_integration import ToolCallMockProvider, _binding, _ctx, make_event

# The shape §24.1 uses.
RFC_EXAMPLE = """---
name: research-helper
description: Searches and fetches reference material
allowed_tools:
  - search_*
  - fetch_*
---

Body.
"""


class TestFrontmatterKeepsTheList:
    def test_the_rfc_example_parses_into_a_list(self) -> None:
        data, _body = parse_frontmatter(RFC_EXAMPLE)
        assert data["allowed_tools"] == ["search_*", "fetch_*"]

    def test_the_scalar_form_still_works(self) -> None:
        data, _ = parse_frontmatter(
            "---\nname: s\ndescription: d\nallowed_tools: search_*, fetch_docs\n---\n\nBody.\n"
        )
        assert data["allowed_tools"] == "search_*, fetch_docs"

    def test_a_scalar_field_written_as_a_list_is_joined_not_refused(self) -> None:
        """An odd `license:` line should not stop a skill from loading."""
        data, _ = parse_frontmatter(
            "---\nname: s\ndescription: d\nlicense:\n  - MIT\n  - Apache-2.0\n---\n\nBody.\n"
        )
        assert data["license"] == ["MIT", "Apache-2.0"]


class TestGlobsGate:
    @pytest.mark.parametrize("allowed", [["search_*"], "search_*"])
    def test_a_glob_covers_every_matching_tool(self, allowed) -> None:  # noqa: ANN001
        meta = SkillMetadata(name="s", description="d", allowed_tools=allowed)
        assert meta.gates("search_web") is True
        assert meta.gates("search_docs") is True

    def test_a_glob_does_not_cover_everything_else(self) -> None:
        meta = SkillMetadata(name="s", description="d", allowed_tools=["search_*"])
        assert meta.gates("delete_everything") is False
        assert meta.gates("fetch_url") is False

    def test_an_exact_name_still_gates_exactly(self) -> None:
        meta = SkillMetadata(name="s", description="d", allowed_tools=["run_report"])
        assert meta.gates("run_report") is True
        assert meta.gates("run_report_v2") is False

    def test_no_allowed_tools_gates_nothing(self) -> None:
        meta = SkillMetadata(name="s", description="d")
        assert meta.gated_tool_names == []
        assert meta.gates("anything") is False

    def test_the_rfc_example_gates_what_it_names(self) -> None:
        """The whole point: written the RFC's way, the skill restricts."""
        data, _ = parse_frontmatter(RFC_EXAMPLE)
        meta = SkillMetadata(
            name="research-helper",
            description="d",
            allowed_tools=data["allowed_tools"],
        )
        assert meta.gates("search_web") is True
        assert meta.gates("fetch_url") is True
        assert meta.gates("send_email") is False


class TestGatingSitesUseTheSameMatcher:
    """Skill gating, ``allowed_tools`` and ToolPolicy decide the same question.

    They must therefore answer it identically: a pattern that covers a tool for
    one has to cover it for the others, or a skill's declaration and a policy's
    disagree about which tools exist.
    """

    def test_the_shared_matcher_is_glob_aware(self) -> None:
        assert matches_any_pattern("search_web", {"search_*"}) is True
        assert matches_any_pattern("send_email", {"search_*"}) is False

    @pytest.mark.parametrize(
        ("tool", "expected"),
        [("search_web", True), ("search_", True), ("websearch", False), ("send_email", False)],
    )
    def test_a_skill_and_a_policy_agree_on_a_pattern(self, tool: str, expected: bool) -> None:
        pattern = "search_*"
        skill = SkillMetadata(name="s", description="d", allowed_tools=[pattern])
        policy = ToolPolicy(allow=[pattern])

        assert skill.gates(tool) is expected
        assert policy.is_allowed(tool) is expected
        assert matches_any_pattern(tool, {pattern}) is expected


class TestAGlobGatedToolDoesNotRun:
    """The matcher being glob-aware is not enough — the callers must call it.

    A test that only exercises ``matches_any_pattern`` passes while a guard
    beside it asks ``name in gated_names``, which no glob ever satisfies. Two
    filters stand between a gated tool and its handler: the tool is kept out of
    the catalogue the model sees, and the execution guard refuses it anyway.
    What a host can observe is the tool not running.
    """

    @staticmethod
    def _registry(tmp_path: Path, allowed_tools: str) -> SkillRegistry:
        skill_dir = tmp_path / "research"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: research\ndescription: Research\n"
            f"allowed_tools: {allowed_tools}\n---\nSearch things.",
            encoding="utf-8",
        )
        registry = SkillRegistry()
        registry.discover(tmp_path)
        return registry

    async def test_a_glob_gated_tool_does_not_execute(self, tmp_path: Path) -> None:
        ran: list[str] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            ran.append(name)
            return "searched"

        provider = ToolCallMockProvider(
            tool_calls=[AIToolCall(id="tc1", name="search_web", arguments={})]
        )
        ch = AIChannel(
            "ai1",
            provider=provider,
            skills=self._registry(tmp_path, "search_*"),
            tool_handler=handler,
            tools=[AITool(name="search_web", description="Search the web", parameters={})],
        )
        await ch.on_event(make_event(body="find something"), _binding(), _ctx())

        assert ran == []
        tool_msg = [m for m in provider.calls[1].messages if m.role == "tool"]
        error = json.loads(tool_msg[0].content[0].result)["error"]
        assert "not available to this agent" in error
        assert "gated behind a skill" in error

    async def test_a_tool_the_glob_does_not_cover_still_runs(self, tmp_path: Path) -> None:
        ran: list[str] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            ran.append(name)
            return "done"

        provider = ToolCallMockProvider(
            tool_calls=[AIToolCall(id="tc1", name="send_email", arguments={})]
        )
        ch = AIChannel(
            "ai1",
            provider=provider,
            skills=self._registry(tmp_path, "search_*"),
            tool_handler=handler,
            tools=[AITool(name="send_email", description="Send an email", parameters={})],
        )
        await ch.on_event(make_event(body="mail it"), _binding(), _ctx())

        assert ran == ["send_email"]
