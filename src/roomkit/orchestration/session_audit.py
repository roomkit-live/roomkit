"""Full session auditing — captures speech, tools, vision, and interruptions.

Records every conversation event into a unified timeline, producing both
a JSONL file and a human-readable conversation transcript.

Usage::

    from roomkit.orchestration.session_audit import JSONLSessionAuditor

    auditor = JSONLSessionAuditor("/tmp/session.jsonl")
    auditor.attach(kit)          # auto-capture speech, vision, barge-in
    # ... tool calls recorded manually via auditor.record_tool(...)
    auditor.print_summary()
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from roomkit.orchestration.tool_audit import ToolAuditEntry, ToolAuditor

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration.session_audit")

_MAX_CONTENT_LEN = 500


class SessionAuditEntry(BaseModel):
    """A single event in the session timeline."""

    ts: str
    """ISO timestamp."""

    type: str
    """Event type: speech | tool_call | vision | barge_in | session | error."""

    role: str | None = None
    """Who produced this: user | assistant | system."""

    content: str
    """Text content, transcription, tool result, or description."""

    duration_ms: float | None = None
    """Duration for timed events (tool calls)."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Extra data (tool name/args, vision terms, etc.)."""


class SessionAuditor(ABC):
    """Abstract base class for full session auditing.

    Extends :class:`ToolAuditor` compatibility so it can replace
    a tool-only auditor as a drop-in.
    """

    @abstractmethod
    def record(self, entry: SessionAuditEntry) -> None:
        """Record a session event."""
        ...

    @abstractmethod
    def summary(self) -> str:
        """Return a human-readable conversation transcript."""
        ...

    @property
    @abstractmethod
    def entries(self) -> list[SessionAuditEntry]:
        """All recorded entries in chronological order."""
        ...

    def print_summary(self) -> None:
        """Log the summary via the module logger."""
        logger.info(self.summary())

    # --- ToolAuditor compatibility -------------------------------------------

    def record_tool(self, entry: ToolAuditEntry) -> None:
        """Record a tool call (ToolAuditor compatibility)."""
        self.record(
            SessionAuditEntry(
                ts=entry.ts,
                type="tool_call",
                role="assistant",
                content=entry.result,
                duration_ms=entry.duration_ms,
                metadata={
                    "tool_name": entry.tool_name,
                    "arguments": entry.arguments,
                    "status": entry.status,
                    "agent_id": entry.agent_id,
                    **entry.metadata,
                },
            )
        )

    @property
    def tool_auditor(self) -> _SessionToolAuditorBridge:
        """A :class:`ToolAuditor` that feeds entries into this session audit.

        Use this when an API expects a ``ToolAuditor``::

            auditor = JSONLSessionAuditor("/tmp/session.jsonl")
            handler = audit_tool_handler(my_handler, auditor.tool_auditor, "agent-1")
        """
        return _SessionToolAuditorBridge(self)

    # --- Hook attachment -----------------------------------------------------

    def attach(self, kit: RoomKit) -> None:
        """Register hooks on *kit* to auto-capture conversation events.

        Captures:

        - **ON_TRANSCRIPTION** — final user and assistant speech turns
        - **ON_BARGE_IN** — user interruptions
        - **ON_SESSION_STARTED** — voice session lifecycle
        - **ON_VISION_RESULT** — periodic vision descriptions

        Tool calls are NOT auto-captured (the result isn't available
        in the hook). Use :meth:`record_tool` from your tool handler.
        """
        from roomkit.models.enums import HookExecution, HookTrigger

        @kit.hook(
            HookTrigger.ON_TRANSCRIPTION,
            execution=HookExecution.ASYNC,
            name="session_audit_transcription",
        )
        async def _on_transcription(event: object, ctx: object) -> None:
            ev = event  # RealtimeTranscriptionEvent
            is_final = getattr(ev, "is_final", True)
            if not is_final:
                return
            text = getattr(ev, "text", "")
            role = getattr(ev, "role", "user")
            if not text or not text.strip():
                return
            self.record(
                SessionAuditEntry(
                    ts=datetime.now().isoformat(),
                    type="speech",
                    role=role,
                    content=text,
                )
            )

        @kit.hook(
            HookTrigger.ON_BARGE_IN,
            execution=HookExecution.ASYNC,
            name="session_audit_barge_in",
        )
        async def _on_barge_in(event: object, ctx: object) -> None:
            self.record(
                SessionAuditEntry(
                    ts=datetime.now().isoformat(),
                    type="barge_in",
                    role="user",
                    content="User interrupted",
                )
            )

        @kit.hook(
            HookTrigger.ON_SESSION_STARTED,
            execution=HookExecution.ASYNC,
            name="session_audit_session",
        )
        async def _on_session_started(event: object, ctx: object) -> None:
            channel_id = getattr(event, "channel_id", "")
            participant = getattr(event, "participant_id", "")
            self.record(
                SessionAuditEntry(
                    ts=datetime.now().isoformat(),
                    type="session",
                    role="system",
                    content="Session started",
                    metadata={
                        "channel_id": channel_id,
                        "participant_id": participant,
                    },
                )
            )

        @kit.hook(
            HookTrigger.ON_VISION_RESULT,
            execution=HookExecution.ASYNC,
            name="session_audit_vision",
        )
        async def _on_vision(event: object, ctx: object) -> None:
            description = getattr(event, "description", "")
            if not description:
                return
            self.record(
                SessionAuditEntry(
                    ts=datetime.now().isoformat(),
                    type="vision",
                    role="system",
                    content=description[:_MAX_CONTENT_LEN],
                )
            )


class JSONLSessionAuditor(SessionAuditor):
    """Writes session events to a JSONL file and keeps them in memory.

    Args:
        path: Path to the JSONL output file.  Parent dirs are
            created automatically.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[SessionAuditEntry] = []

    def record(self, entry: SessionAuditEntry) -> None:
        self._entries.append(entry)
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(entry.model_dump(), default=str) + "\n")
        except Exception:
            logger.exception("Failed to write session entry to %s", self._path)

    @property
    def entries(self) -> list[SessionAuditEntry]:
        return list(self._entries)

    def summary(self) -> str:
        if not self._entries:
            return "\nNo session events recorded."

        lines = [f"\nSession Audit ({self._path})", "=" * 60]

        for entry in self._entries:
            ts_short = _format_time(entry.ts)
            line = _format_entry(entry, ts_short)
            lines.append(line)

        # --- Stats -----------------------------------------------------------
        lines.append("")
        stats = _compute_stats(self._entries)
        duration_str = _format_duration(stats["duration_s"]) if stats["duration_s"] else "?"
        parts = [f"Duration: {duration_str}"]
        if stats["user_turns"] or stats["assistant_turns"]:
            u, a = stats["user_turns"], stats["assistant_turns"]
            parts.append(f"Turns: {u} user, {a} assistant")
        if stats["tool_calls"]:
            parts.append(f"Tool calls: {stats['tool_calls']} ({stats['tool_ms']:.0f}ms)")
        if stats["vision_count"]:
            parts.append(f"Vision: {stats['vision_count']}")
        if stats["barge_ins"]:
            parts.append(f"Interruptions: {stats['barge_ins']}")
        lines.append("  " + " | ".join(parts))

        # Tool breakdown
        if stats["by_tool"]:
            tools_str = ", ".join(f"{k}({v})" for k, v in sorted(stats["by_tool"].items()))
            lines.append(f"  Tools: {tools_str}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ToolAuditor bridge
# ---------------------------------------------------------------------------


class _SessionToolAuditorBridge(ToolAuditor):
    """Bridges the ToolAuditor interface to a SessionAuditor."""

    def __init__(self, session: SessionAuditor) -> None:
        self._session = session

    def record(self, entry: ToolAuditEntry) -> None:
        self._session.record_tool(entry)

    @property
    def entries(self) -> list[ToolAuditEntry]:
        result = []
        for e in self._session.entries:
            if e.type == "tool_call":
                result.append(
                    ToolAuditEntry(
                        ts=e.ts,
                        agent_id=e.metadata.get("agent_id", ""),
                        tool_name=e.metadata.get("tool_name", ""),
                        arguments=e.metadata.get("arguments", {}),
                        result=e.content,
                        status=e.metadata.get("status", "ok"),
                        duration_ms=e.duration_ms or 0,
                        metadata={
                            k: v
                            for k, v in e.metadata.items()
                            if k not in ("tool_name", "arguments", "status", "agent_id")
                        },
                    )
                )
        return result

    def summary(self) -> str:
        return self._session.summary()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_time(ts: str) -> str:
    """Extract HH:MM:SS from an ISO timestamp."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return ts[:8]


def _format_entry(entry: SessionAuditEntry, ts: str) -> str:
    """Format a single entry as a transcript line."""
    if entry.type == "speech":
        label = "USER" if entry.role == "user" else "ASSISTANT"
        text = entry.content[:200]
        if len(entry.content) > 200:
            text += "..."
        return f'  [{ts}] {label}: "{text}"'

    if entry.type == "tool_call":
        tool_name = entry.metadata.get("tool_name", "?")
        status = entry.metadata.get("status", "ok").upper()
        ms = f" ({entry.duration_ms:.0f}ms)" if entry.duration_ms else ""
        args = entry.metadata.get("arguments", {})
        args_str = f" {json.dumps(args, default=str)}" if args else ""
        line = f"  [{ts}] TOOL {tool_name}{args_str} → {status}{ms}"
        # Add result preview on next line
        if entry.content:
            preview = entry.content[:120].replace("\n", " ")
            if len(entry.content) > 120:
                preview += "..."
            line += f"\n           {preview}"
        return line

    if entry.type == "vision":
        preview = entry.content[:150].replace("\n", " ")
        if len(entry.content) > 150:
            preview += "..."
        return f"  [{ts}] VISION {preview}"

    if entry.type == "barge_in":
        return f"  [{ts}] BARGE-IN {entry.content}"

    if entry.type == "session":
        return f"  [{ts}] SESSION {entry.content}"

    if entry.type == "error":
        return f"  [{ts}] ERROR {entry.content[:200]}"

    return f"  [{ts}] {entry.type.upper()} {entry.content[:200]}"


def _compute_stats(entries: list[SessionAuditEntry]) -> dict[str, Any]:
    """Compute summary statistics from entries."""
    user_turns = 0
    assistant_turns = 0
    tool_calls = 0
    tool_ms = 0.0
    by_tool: dict[str, int] = {}
    vision_count = 0
    barge_ins = 0

    for e in entries:
        if e.type == "speech":
            if e.role == "user":
                user_turns += 1
            else:
                assistant_turns += 1
        elif e.type == "tool_call":
            tool_calls += 1
            tool_ms += e.duration_ms or 0
            name = e.metadata.get("tool_name", "?")
            by_tool[name] = by_tool.get(name, 0) + 1
        elif e.type == "vision":
            vision_count += 1
        elif e.type == "barge_in":
            barge_ins += 1

    # Duration from first to last entry
    duration_s: float | None = None
    if len(entries) >= 2:
        try:
            first = datetime.fromisoformat(entries[0].ts)
            last = datetime.fromisoformat(entries[-1].ts)
            duration_s = (last - first).total_seconds()
        except (ValueError, TypeError):
            pass

    return {
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "tool_calls": tool_calls,
        "tool_ms": tool_ms,
        "by_tool": by_tool,
        "vision_count": vision_count,
        "barge_ins": barge_ins,
        "duration_s": duration_s,
    }


def _format_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs}s"
