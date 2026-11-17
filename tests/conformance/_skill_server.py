"""Script the public MCP skill-server protocol without interpreting its replies."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from roomkit.skills.mcp import SKILLS_EXTENSION

if TYPE_CHECKING:
    from roomkit.skills import SkillExecutionRequest


class ScriptedSkillServer:
    """Replies are consumed exactly once, with exact method/parameter checks."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self.server_extensions = copy.deepcopy(spec.get("extensions", {SKILLS_EXTENSION: {}}))
        self.replies = list(spec["replies"])
        self.requests: list[dict[str, Any]] = []
        self.problems: list[str] = []
        self._position = 0
        self.approval_answers = spec.get("approvals")
        self.approvals: list[dict[str, Any]] = []

    async def request(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._reply(method, dict(params) if params is not None else None)

    async def read_resource(self, uri: str) -> Any:
        return self._reply("resources/read", {"uri": uri})

    async def approve(self, request: SkillExecutionRequest) -> bool:
        self.approvals.append(asdict(request))
        position = len(self.approvals) - 1
        if self.approval_answers is None or position >= len(self.approval_answers):
            self.problems.append("approval fixture exhausted")
            raise RuntimeError("approval fixture exhausted")
        answer = self.approval_answers[position]
        if isinstance(answer, Mapping):
            raise RuntimeError(answer["error"])
        return answer

    def _reply(self, method: str, params: Any) -> Any:
        self.requests.append({"method": method, "params": copy.deepcopy(params)})
        if self._position >= len(self.replies):
            self.problems.append(f"unexpected {method}: no scripted reply remains")
            raise RuntimeError("skill server fixture exhausted")
        reply = self.replies[self._position]
        self._position += 1
        if (reply["method"], reply["params"]) != (method, params):
            self.problems.append(
                f"expected {reply['method']} {reply['params']!r}, got {method} {params!r}"
            )
            raise RuntimeError("skill server fixture request mismatch")
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return copy.deepcopy(reply["result"])

    def finish(self) -> list[str]:
        if self.approval_answers is not None and len(self.approvals) != len(self.approval_answers):
            self.problems.append("scripted approvals were not consumed")
        if self._position != len(self.replies):
            self.problems.append(
                f"{len(self.replies) - self._position} scripted replies were not read"
            )
        return self.problems
