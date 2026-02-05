"""Tests for AI documentation helpers."""

from __future__ import annotations


class TestAIDocs:
    def test_get_llms_txt(self) -> None:
        """get_llms_txt returns llms.txt content."""
        from roomkit import get_llms_txt

        content = get_llms_txt()
        assert "# RoomKit" in content
        assert "pure async python library" in content.lower()

    def test_get_agents_md(self) -> None:
        """get_agents_md returns AGENTS.md content."""
        from roomkit import get_agents_md

        content = get_agents_md()
        assert "# RoomKit" in content
        assert "pytest" in content
        assert "ruff" in content

    def test_get_ai_context(self) -> None:
        """get_ai_context returns combined content."""
        from roomkit import get_ai_context

        content = get_ai_context()
        assert "AGENTS.md" in content
        assert "llms.txt" in content
        assert "# RoomKit" in content
