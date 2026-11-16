"""A host executor fixture; it records invocations without starting a process."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from roomkit.skills import ScriptExecutor, ScriptResult, Skill


class ScriptedScriptExecutor(ScriptExecutor):
    def __init__(self, spec: Mapping[str, Any], record: dict[str, Any]) -> None:
        self.spec = spec
        self.record = record

    async def execute(
        self, skill: Skill, script_name: str, arguments: dict[str, str] | None = None
    ) -> ScriptResult:
        # Record entry before touching the file: a framework containment
        # regression must remain visible even if this fixture also rejects it.
        call = {
            "skill_name": skill.name,
            "script_name": script_name,
            "arguments": arguments,
        }
        self.record["calls"].append(call)
        path = skill.resolve_script(script_name)
        call["contents"] = await asyncio.to_thread(path.read_text, encoding="utf-8")
        if failure := self.spec.get("raises"):
            error = type(failure["kind"], (Exception,), {})
            raise error(failure["message"])
        return ScriptResult.model_validate(self.spec.get("result", {"exit_code": 0}))

    async def close(self) -> None:
        self.record["closes"] += 1
