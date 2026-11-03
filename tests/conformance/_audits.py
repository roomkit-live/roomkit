"""The run's auditors (README, "Setup" and "Scripted agents"): the
implementation's JSONL tool and session auditors, each over a file of its
own, and a scripted agent's tool handler wrapped to record to one of them."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from roomkit import RoomKit
from roomkit.orchestration import (
    JSONLSessionAuditor,
    JSONLToolAuditor,
    ToolAuditor,
    audit_tool_handler,
)


class Auditors:
    """The run's tool auditor and session auditor, each made at its first use."""

    def __init__(self) -> None:
        self._directory: Path | None = None
        self._tool: JSONLToolAuditor | None = None
        self._session: JSONLSessionAuditor | None = None

    def _path(self, name: str) -> Path:
        if self._directory is None:
            self._directory = Path(tempfile.mkdtemp(prefix="roomkit-conformance-audit-"))
        return self._directory / name

    @property
    def tool(self) -> JSONLToolAuditor:
        if self._tool is None:
            self._tool = JSONLToolAuditor(self._path("tools.jsonl"))
        return self._tool

    @property
    def session(self) -> JSONLSessionAuditor:
        if self._session is None:
            self._session = JSONLSessionAuditor(self._path("session.jsonl"))
        return self._session

    def set_up(self, spec: Mapping[str, Any] | None, kit: RoomKit) -> None:
        """The setup's `session_auditor`: made, and attached to *kit* when it says so."""
        if spec is not None and spec.get("attached"):
            self.session.attach(kit)

    def audited(self, handler: Any, agent: Mapping[str, Any]) -> Any:
        """*handler* wrapped to record to the auditor the agent's `audit` names."""
        named = agent.get("audit")
        if named is None:
            return handler
        auditor: ToolAuditor = self.tool if named == "tool_auditor" else self.session.tool_auditor
        return audit_tool_handler(handler, auditor, agent["id"])

    def observe(self) -> tuple[list[Any], list[Any], list[str]]:
        """Each auditor's entries, and how its JSONL file differs from them."""
        problems: list[str] = []
        tools = self._entries(self._tool, "tools.jsonl", "tool_audit", problems)
        sessions = self._entries(self._session, "session.jsonl", "session_audit", problems)
        return tools, sessions, problems

    def _entries(
        self,
        auditor: JSONLToolAuditor | JSONLSessionAuditor | None,
        name: str,
        label: str,
        problems: list[str],
    ) -> list[Any]:
        if auditor is None:
            return []
        entries = [entry.model_dump(mode="json") for entry in auditor.entries]
        file = self._path(name)
        written = file.read_text().splitlines() if file.exists() else []
        lines = [json.loads(line) for line in written]
        if lines != entries:
            problems.append(f"{label}: its JSONL file holds {lines!r}, its entries {entries!r}")
        return entries

    def close(self) -> None:
        if self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)
