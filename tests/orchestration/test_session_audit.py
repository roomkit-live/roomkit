"""Tests for SessionAuditEntry, JSONLSessionAuditor, and hook attachment."""

from __future__ import annotations

import json
from pathlib import Path

from roomkit.orchestration.session_audit import (
    JSONLSessionAuditor,
    SessionAuditEntry,
)
from roomkit.orchestration.tool_audit import ToolAuditEntry

# ---------------------------------------------------------------------------
# SessionAuditEntry model
# ---------------------------------------------------------------------------


def test_session_entry_model_dump() -> None:
    entry = SessionAuditEntry(
        ts="2026-03-20T12:00:00+00:00",
        type="speech",
        role="user",
        content="Hello world",
    )
    d = entry.model_dump()
    assert d["type"] == "speech"
    assert d["role"] == "user"
    assert d["content"] == "Hello world"
    assert d["metadata"] == {}
    assert d["duration_ms"] is None


def test_session_entry_with_metadata() -> None:
    entry = SessionAuditEntry(
        ts="t",
        type="tool_call",
        role="assistant",
        content="result text",
        duration_ms=42.0,
        metadata={"tool_name": "search", "status": "ok"},
    )
    assert entry.metadata["tool_name"] == "search"
    assert entry.duration_ms == 42.0


# ---------------------------------------------------------------------------
# JSONLSessionAuditor — basic recording
# ---------------------------------------------------------------------------


def test_jsonl_session_auditor_records_entries(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    auditor = JSONLSessionAuditor(path)

    auditor.record(
        SessionAuditEntry(
            ts="2026-03-20T12:00:00+00:00",
            type="speech",
            role="user",
            content="Hello",
        )
    )
    auditor.record(
        SessionAuditEntry(
            ts="2026-03-20T12:00:01+00:00",
            type="speech",
            role="assistant",
            content="Hi there!",
        )
    )

    assert len(auditor.entries) == 2
    assert auditor.entries[0].role == "user"
    assert auditor.entries[1].role == "assistant"

    # Verify JSONL file
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["type"] == "speech"
    assert first["content"] == "Hello"


def test_jsonl_session_auditor_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deep" / "session.jsonl"
    auditor = JSONLSessionAuditor(path)
    auditor.record(
        SessionAuditEntry(
            ts="t",
            type="session",
            role="system",
            content="started",
        )
    )
    assert path.exists()


# ---------------------------------------------------------------------------
# ToolAuditor bridge
# ---------------------------------------------------------------------------


def test_record_tool_adds_to_timeline(tmp_path: Path) -> None:
    auditor = JSONLSessionAuditor(tmp_path / "s.jsonl")

    # Record a speech turn first
    auditor.record(
        SessionAuditEntry(
            ts="2026-03-20T12:00:00+00:00",
            type="speech",
            role="user",
            content="Search for roomkit",
        )
    )

    # Record a tool call via record_tool
    auditor.record_tool(
        ToolAuditEntry(
            ts="2026-03-20T12:00:01+00:00",
            agent_id="screen-assistant",
            tool_name="describe_screen",
            arguments={"query": "what's on screen"},
            result="Chrome with Google open",
            status="ok",
            duration_ms=5000,
        )
    )

    assert len(auditor.entries) == 2
    tool_entry = auditor.entries[1]
    assert tool_entry.type == "tool_call"
    assert tool_entry.metadata["tool_name"] == "describe_screen"
    assert tool_entry.duration_ms == 5000
    assert tool_entry.content == "Chrome with Google open"


def test_tool_auditor_bridge_entries(tmp_path: Path) -> None:
    auditor = JSONLSessionAuditor(tmp_path / "s.jsonl")

    auditor.record_tool(
        ToolAuditEntry(
            ts="t",
            agent_id="a",
            tool_name="search",
            arguments={"q": "hello"},
            result="found",
            status="ok",
            duration_ms=42,
        )
    )

    bridge = auditor.tool_auditor
    tool_entries = bridge.entries
    assert len(tool_entries) == 1
    assert tool_entries[0].tool_name == "search"
    assert tool_entries[0].result == "found"
    assert tool_entries[0].agent_id == "a"


def test_tool_auditor_bridge_record(tmp_path: Path) -> None:
    """Bridge.record() feeds into the session auditor."""
    auditor = JSONLSessionAuditor(tmp_path / "s.jsonl")
    bridge = auditor.tool_auditor

    bridge.record(
        ToolAuditEntry(
            ts="t",
            agent_id="a",
            tool_name="click",
            arguments={},
            result="ok",
            status="ok",
            duration_ms=10,
        )
    )

    assert len(auditor.entries) == 1
    assert auditor.entries[0].type == "tool_call"


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------


def test_summary_empty(tmp_path: Path) -> None:
    auditor = JSONLSessionAuditor(tmp_path / "s.jsonl")
    assert "No session events" in auditor.summary()


def test_summary_full_conversation(tmp_path: Path) -> None:
    auditor = JSONLSessionAuditor(tmp_path / "s.jsonl")

    auditor.record(
        SessionAuditEntry(
            ts="2026-03-20T12:00:00+00:00",
            type="session",
            role="system",
            content="Session started",
        )
    )
    auditor.record(
        SessionAuditEntry(
            ts="2026-03-20T12:00:05+00:00",
            type="speech",
            role="user",
            content="Open Chrome and search for roomkit",
        )
    )
    auditor.record(
        SessionAuditEntry(
            ts="2026-03-20T12:00:07+00:00",
            type="speech",
            role="assistant",
            content="Let me check your screen first.",
        )
    )
    auditor.record_tool(
        ToolAuditEntry(
            ts="2026-03-20T12:00:08+00:00",
            agent_id="screen-assistant",
            tool_name="describe_screen",
            arguments={},
            result="Chrome with Google open",
            status="ok",
            duration_ms=5886,
        )
    )
    auditor.record(
        SessionAuditEntry(
            ts="2026-03-20T12:00:14+00:00",
            type="vision",
            role="system",
            content="Chrome browser showing Google search page",
        )
    )
    auditor.record(
        SessionAuditEntry(
            ts="2026-03-20T12:00:20+00:00",
            type="barge_in",
            role="user",
            content="User interrupted",
        )
    )
    auditor.record(
        SessionAuditEntry(
            ts="2026-03-20T12:00:30+00:00",
            type="speech",
            role="user",
            content="Never mind, I found it",
        )
    )

    summary = auditor.summary()

    # Check structure
    assert "Session Audit" in summary
    assert "SESSION Session started" in summary
    assert 'USER: "Open Chrome' in summary
    assert 'ASSISTANT: "Let me check' in summary
    assert "TOOL describe_screen" in summary
    assert "5886ms" in summary
    assert "VISION" in summary
    assert "BARGE-IN" in summary

    # Check stats
    assert "Turns: 2 user, 1 assistant" in summary
    assert "Tool calls: 1" in summary
    assert "Vision: 1" in summary
    assert "Interruptions: 1" in summary


def test_summary_duration_formatting(tmp_path: Path) -> None:
    auditor = JSONLSessionAuditor(tmp_path / "s.jsonl")

    auditor.record(
        SessionAuditEntry(
            ts="2026-03-20T12:00:00+00:00",
            type="session",
            role="system",
            content="started",
        )
    )
    auditor.record(
        SessionAuditEntry(
            ts="2026-03-20T12:03:42+00:00",
            type="speech",
            role="user",
            content="done",
        )
    )

    summary = auditor.summary()
    assert "3m 42s" in summary


# ---------------------------------------------------------------------------
# Hook attachment (unit test — no real RoomKit)
# ---------------------------------------------------------------------------


def test_attach_registers_hooks(tmp_path: Path) -> None:
    """Verify attach() calls kit.hook() for the expected triggers."""
    auditor = JSONLSessionAuditor(tmp_path / "s.jsonl")

    registered: list[str] = []

    class FakeKit:
        def hook(self, trigger: object, **kwargs: object) -> object:
            registered.append(str(trigger))

            def decorator(fn: object) -> object:
                return fn

            return decorator

    auditor.attach(FakeKit())  # type: ignore[arg-type]

    trigger_values = [r.split(".")[-1] for r in registered]
    assert "on_transcription" in trigger_values
    assert "on_barge_in" in trigger_values
    assert "on_session_started" in trigger_values
    assert "on_vision_result" in trigger_values
