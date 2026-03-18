"""Tests for ToolAuditEntry, JSONLToolAuditor, and audit wrappers."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from roomkit.orchestration.tool_audit import (
    JSONLToolAuditor,
    ToolAuditEntry,
    audit_tool_handler,
)

# ---------------------------------------------------------------------------
# ToolAuditEntry model
# ---------------------------------------------------------------------------


def test_audit_entry_model_dump() -> None:
    entry = ToolAuditEntry(
        ts="t",
        agent_id="a",
        tool_name="search",
        arguments={"q": "hello"},
        result="found",
        status="ok",
        duration_ms=42.5,
    )
    d = entry.model_dump()
    assert d["tool_name"] == "search"
    assert d["status"] == "ok"
    assert d["metadata"] == {}


def test_audit_entry_model_validate() -> None:
    data = {
        "ts": "t",
        "agent_id": "a",
        "tool_name": "search",
        "arguments": {},
        "result": "ok",
        "status": "ok",
        "duration_ms": 10,
    }
    entry = ToolAuditEntry.model_validate(data)
    assert entry.tool_name == "search"


# ---------------------------------------------------------------------------
# JSONLToolAuditor
# ---------------------------------------------------------------------------


def test_jsonl_auditor_record_and_summary(tmp_path: Path) -> None:
    p = tmp_path / "audit.jsonl"
    auditor = JSONLToolAuditor(p)
    auditor.record(
        ToolAuditEntry(
            ts="t",
            agent_id="a",
            tool_name="search",
            arguments={"q": "x"},
            result="done",
            status="ok",
            duration_ms=10,
        )
    )
    assert len(auditor.entries) == 1

    lines = p.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["tool_name"] == "search"

    summary = auditor.summary()
    assert "search" in summary
    assert "1 calls" in summary


def test_jsonl_auditor_empty_summary(tmp_path: Path) -> None:
    p = tmp_path / "audit.jsonl"
    auditor = JSONLToolAuditor(p)
    assert "No tool calls recorded" in auditor.summary()


# ---------------------------------------------------------------------------
# audit_tool_handler
# ---------------------------------------------------------------------------


async def test_audit_tool_handler_records(tmp_path: Path) -> None:
    auditor = JSONLToolAuditor(tmp_path / "audit.jsonl")

    async def handler(name: str, args: dict[str, Any]) -> str:
        return '{"status": "ok", "data": "hello"}'

    wrapped = audit_tool_handler(handler, auditor, "test-agent")
    result = await wrapped("my_tool", {"x": 1})
    assert "hello" in result
    assert len(auditor.entries) == 1
    assert auditor.entries[0].status == "ok"
    assert auditor.entries[0].tool_name == "my_tool"


async def test_audit_tool_handler_detects_failed(tmp_path: Path) -> None:
    auditor = JSONLToolAuditor(tmp_path / "audit.jsonl")

    async def handler(name: str, args: dict[str, Any]) -> str:
        return '{"status": "failed", "error": "not found"}'

    wrapped = audit_tool_handler(handler, auditor, "a")
    await wrapped("tool", {})
    assert auditor.entries[0].status == "failed"


async def test_audit_tool_handler_error(tmp_path: Path) -> None:
    auditor = JSONLToolAuditor(tmp_path / "audit.jsonl")

    async def handler(name: str, args: dict[str, Any]) -> str:
        raise ValueError("boom")

    wrapped = audit_tool_handler(handler, auditor, "a")
    with contextlib.suppress(ValueError):
        await wrapped("tool", {})
    assert auditor.entries[0].status == "error"
    assert "boom" in auditor.entries[0].result


async def test_audit_tool_handler_truncates(tmp_path: Path) -> None:
    auditor = JSONLToolAuditor(tmp_path / "audit.jsonl")

    async def handler(name: str, args: dict[str, Any]) -> str:
        return "x" * 1000

    wrapped = audit_tool_handler(handler, auditor, "a")
    result = await wrapped("tool", {})
    # The returned result is NOT truncated
    assert len(result) == 1000
    # But the recorded entry IS truncated
    assert len(auditor.entries[0].result) == 500
