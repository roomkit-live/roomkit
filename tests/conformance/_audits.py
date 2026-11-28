"""The run's auditors (README, "Setup" and "Scripted agents"): the
implementation's JSONL tool and session auditors, each over a file of its
own, the runner's own tool auditor, and a scripted agent's tool handler
wrapped to record to one of them.

``audit_files`` sets how the files stand before the run (§15.8.3): under
directories not made yet, made unwritable, or holding lines of an earlier run.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit
from roomkit.orchestration import (
    JSONLSessionAuditor,
    JSONLToolAuditor,
    SessionAuditEntry,
    ToolAuditEntry,
    ToolAuditor,
    audit_tool_handler,
)
from tests.conformance._calls import Call, read

if TYPE_CHECKING:
    from tests.conformance._harness import Harness

_FILES = {"tool_audit": "tools.jsonl", "session_audit": "session.jsonl"}


class OwnToolAuditor(ToolAuditor):
    """A ToolAuditor of the runner's own, as an integrator's backend
    implements the interface (README, ``own_auditor``): it keeps each entry
    it is handed."""

    def __init__(self) -> None:
        self._entries: list[ToolAuditEntry] = []

    def record(self, entry: ToolAuditEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[ToolAuditEntry]:
        return list(self._entries)

    def summary(self) -> str:
        return f"{len(self._entries)} tool calls"


class Auditors:
    """The run's tool auditor and session auditor, each made at its first use,
    and its own tool auditor."""

    def __init__(self) -> None:
        self._directory: Path | None = None
        self._files: Mapping[str, Any] = {}
        self._tool: JSONLToolAuditor | None = None
        self._session: JSONLSessionAuditor | None = None
        self.own = OwnToolAuditor()

    def _path(self, name: str) -> Path:
        if self._directory is None:
            self._directory = Path(tempfile.mkdtemp(prefix="roomkit-conformance-audit-"))
        if self._files.get("nested"):
            # Directories the auditor makes itself (§15.8.3).
            return self._directory / "missing" / "one" / name
        return self._directory / name

    @property
    def tool(self) -> JSONLToolAuditor:
        if self._tool is None:
            self._tool = JSONLToolAuditor(self._path(_FILES["tool_audit"]))
        return self._tool

    @property
    def session(self) -> JSONLSessionAuditor:
        if self._session is None:
            self._session = JSONLSessionAuditor(self._path(_FILES["session_audit"]))
        return self._session

    def set_up(
        self,
        spec: Mapping[str, Any] | None,
        files: Mapping[str, Any] | None,
        kit: RoomKit,
    ) -> None:
        """The setup's ``audit_files``, then its ``session_auditor``: made,
        and attached to *kit* when it says so."""
        if files is not None:
            self._set_up_files(files)
        if spec is not None and spec.get("attached"):
            self.session.attach(kit)

    def _set_up_files(self, files: Mapping[str, Any]) -> None:
        """Each file as ``audit_files`` says it stands, then both auditors made."""
        self._files = files
        for kind, name in _FILES.items():
            path = self._path(name)
            if files.get("unwritable"):
                path.mkdir(parents=True)
            lines = files.get("lines", {}).get(kind)
            if lines:
                path.write_text("".join(json.dumps(line) + "\n" for line in lines))
        self._tool = JSONLToolAuditor(self._path(_FILES["tool_audit"]))
        self._session = JSONLSessionAuditor(self._path(_FILES["session_audit"]))

    def audited(self, handler: Any, agent: Mapping[str, Any]) -> Any:
        """*handler* wrapped to record to the auditor the agent's `audit` names."""
        named = agent.get("audit")
        if named is None:
            return handler
        auditors: dict[str, ToolAuditor] = {"own_auditor": self.own}
        auditor = auditors.get(named) or (
            self.tool if named == "tool_auditor" else self.session.tool_auditor
        )
        return audit_tool_handler(handler, auditor, agent["id"])

    def observe(self) -> tuple[list[Any], list[Any], list[str]]:
        """Each auditor's entries, and how its JSONL file differs from them."""
        problems: list[str] = []
        tools = self._entries(self._tool, "tool_audit", problems)
        sessions = self._entries(self._session, "session_audit", problems)
        return tools, sessions, problems

    def own_entries(self) -> list[Any]:
        """What the runner's own tool auditor was handed (README, ``own_audit``)."""
        return [entry.model_dump(mode="json") for entry in self.own.entries]

    def _entries(
        self,
        auditor: JSONLToolAuditor | JSONLSessionAuditor | None,
        label: str,
        problems: list[str],
    ) -> list[Any]:
        if auditor is None:
            return []
        entries = [entry.model_dump(mode="json") for entry in auditor.entries]
        if self._files.get("unwritable"):
            return entries
        file = self._path(_FILES[label])
        written = file.read_text().splitlines() if file.exists() else []
        lines = [json.loads(line) for line in written]
        held = [*self._files.get("lines", {}).get(label, []), *entries]
        if lines != held:
            problems.append(f"{label}: its JSONL file holds {lines!r}, its entries {held!r}")
        return entries

    def close(self) -> None:
        if self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)


async def _record(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    """The application records an entry of its own (README, ``session_audit.record``)."""
    entry = read(lambda: SessionAuditEntry.model_validate(spec["entry"]))
    harness.auditors.session.record(entry)


AUDIT_CALLS: dict[str, Call] = {"session_audit.record": _record}

# Each call's arguments, beside `call` and `organization_id`.
AUDIT_ARGUMENTS: dict[str, set[str]] = {"session_audit.record": {"entry"}}
