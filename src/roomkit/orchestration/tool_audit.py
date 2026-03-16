"""Tool execution auditing ABC and built-in implementations.

Provides a standard interface for recording tool calls with input,
output, timing, and status. Plugs into AIChannel and RealtimeVoiceChannel
to automatically audit all tool executions.

Usage::

    from roomkit.orchestration.tool_audit import JSONLToolAuditor

    auditor = JSONLToolAuditor("/tmp/audit.jsonl")
    agent = Agent("my-agent", provider=..., auditor=auditor)

    # After session:
    auditor.print_summary()
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("roomkit.orchestration.tool_audit")


@dataclass
class ToolAuditEntry:
    """A single tool execution record."""

    ts: str
    agent_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: str
    status: str  # ok | failed | error
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result[:500],
            "status": self.status,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class ToolAuditor(ABC):
    """Abstract base class for tool execution auditing.

    Implementations record tool calls and provide summaries.
    """

    @abstractmethod
    def record(self, entry: ToolAuditEntry) -> None:
        """Record a tool execution entry."""
        ...

    @abstractmethod
    def summary(self) -> str:
        """Return a human-readable summary of all recorded entries."""
        ...

    @property
    @abstractmethod
    def entries(self) -> list[ToolAuditEntry]:
        """All recorded entries."""
        ...

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())


class JSONLToolAuditor(ToolAuditor):
    """Writes audit entries to a JSONL file and keeps them in memory.

    Args:
        path: Path to the JSONL output file.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[ToolAuditEntry] = []

    def record(self, entry: ToolAuditEntry) -> None:
        self._entries.append(entry)
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        except Exception:
            logger.exception("Failed to write audit entry to %s", self._path)

    @property
    def entries(self) -> list[ToolAuditEntry]:
        return list(self._entries)

    def summary(self) -> str:
        if not self._entries:
            return "\nNo tool calls recorded."
        lines = [f"\nTool Audit ({self._path})", "=" * 60]
        for i, e in enumerate(self._entries, 1):
            status_icon = "OK" if e.status == "ok" else "FAIL"
            result_preview = e.result[:100].replace("\n", " ")
            if len(e.result) > 100:
                result_preview += "..."
            lines.append(f"  {i:2d}. [{status_icon}] {e.tool_name}  ({e.duration_ms:.0f}ms)")
            lines.append(f"      → {result_preview}")
        # Stats
        by_tool: dict[str, int] = {}
        total_ms = 0.0
        for e in self._entries:
            by_tool[e.tool_name] = by_tool.get(e.tool_name, 0) + 1
            total_ms += e.duration_ms
        tools_str = ", ".join(f"{k}({v})" for k, v in sorted(by_tool.items()))
        lines.append(f"\n  Total: {len(self._entries)} calls, {total_ms:.0f}ms")
        lines.append(f"  Tools: {tools_str}")
        return "\n".join(lines)


class ConsoleToolAuditor(ToolAuditor):
    """Prints audit entries to the console in real-time.

    Also keeps entries in memory for summary.
    """

    def __init__(self, *, level: int = logging.INFO) -> None:
        self._entries: list[ToolAuditEntry] = []
        self._level = level

    def record(self, entry: ToolAuditEntry) -> None:
        self._entries.append(entry)
        status = "+" if entry.status == "ok" else "x"
        logger.log(
            self._level,
            "[AUDIT] [%s] %s.%s → %s (%dms) %s",
            status,
            entry.agent_id,
            entry.tool_name,
            entry.status,
            entry.duration_ms,
            entry.result[:80],
        )

    @property
    def entries(self) -> list[ToolAuditEntry]:
        return list(self._entries)

    def summary(self) -> str:
        if not self._entries:
            return "\nNo tool calls recorded."
        lines = ["\nTool Audit (console)", "=" * 60]
        for i, e in enumerate(self._entries, 1):
            status_icon = "OK" if e.status == "ok" else "FAIL"
            lines.append(f"  {i:2d}. [{status_icon}] {e.agent_id}.{e.tool_name}  ({e.duration_ms:.0f}ms)")
        total_ms = sum(e.duration_ms for e in self._entries)
        lines.append(f"\n  Total: {len(self._entries)} calls, {total_ms:.0f}ms")
        return "\n".join(lines)


def audit_tool_handler(
    handler: Any,
    auditor: ToolAuditor,
    agent_id: str,
) -> Any:
    """Wrap a tool handler to automatically record audit entries.

    Args:
        handler: The original async tool handler ``(name, args) -> str``.
        auditor: The ToolAuditor to record entries to.
        agent_id: Agent ID for the audit entries.

    Returns:
        A wrapped handler with the same signature.
    """

    async def _audited(name: str, arguments: dict[str, Any]) -> str:
        t0 = time.monotonic()
        status = "ok"
        result = ""
        try:
            result = await handler(name, arguments)
            # Detect failures from JSON result
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and parsed.get("status") == "failed":
                    status = "failed"
            except (json.JSONDecodeError, TypeError):
                pass
            return result
        except Exception as exc:
            status = "error"
            result = str(exc)
            raise
        finally:
            elapsed = (time.monotonic() - t0) * 1000
            auditor.record(ToolAuditEntry(
                ts=datetime.now(UTC).isoformat(),
                agent_id=agent_id,
                tool_name=name,
                arguments=dict(arguments),
                result=result,
                status=status,
                duration_ms=elapsed,
            ))

    return _audited
