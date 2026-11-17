"""Record a host sandbox call without running a command or applying policy."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from roomkit import SandboxExecutor, SandboxResult


def sandbox_options(channel: Mapping[str, Any], observed: dict[str, Any] | None) -> dict[str, Any]:
    if "sandbox" not in channel:
        return {}
    if observed is None:
        raise AssertionError("sandbox: no observation sink")
    record = observed.setdefault(channel["id"], {"calls": [], "closes": 0})
    return {"sandbox": ScriptedSandbox(channel["sandbox"], record)}


class ScriptedSandbox(SandboxExecutor):
    def __init__(self, spec: Mapping[str, Any], record: dict[str, Any]) -> None:
        if not isinstance(spec, Mapping):
            raise ValueError("sandbox: not an object")
        self.spec = spec
        self.record = record

    def tool_definitions(self) -> list[dict[str, Any]]:
        # Prefix checks and policy belong to the channel, not this fixture.
        return deepcopy(self.spec.get("tools", []))

    async def execute(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> SandboxResult:
        self.record["calls"].append({"command": command, "arguments": deepcopy(arguments)})
        if "raises" in self.spec:
            failure = self.spec["raises"]
            error = type(failure["kind"], (Exception,), {})
            raise error(failure["message"])
        return SandboxResult.model_validate(self.spec.get("result", {"exit_code": 0}))

    async def close(self) -> None:
        self.record["closes"] += 1
