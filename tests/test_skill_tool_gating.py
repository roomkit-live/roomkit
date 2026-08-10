"""Skill tool gating actually gates (RFC §24.2).

`allowed_tools` entries are ToolPolicy globs. Two things broke that: the
frontmatter parser stringified a YAML list into its own repr, and both gating
sites compared tool names for exact membership. A skill written the way the
RFC's own example writes it therefore restricted nothing at all — silently.
"""

from __future__ import annotations

import pytest

from roomkit.skills.models import SkillMetadata
from roomkit.skills.parser import parse_frontmatter

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
    def test_ai_policy_matcher_is_glob_aware(self) -> None:
        from roomkit.channels._ai_policy import _matches_any

        assert _matches_any("search_web", {"search_*"}) is True
        assert _matches_any("send_email", {"search_*"}) is False

    def test_realtime_matcher_is_glob_aware(self) -> None:
        from roomkit.channels._realtime_skills import _matches_any

        assert _matches_any("search_web", {"search_*"}) is True
        assert _matches_any("send_email", {"search_*"}) is False
