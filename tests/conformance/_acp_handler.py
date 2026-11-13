"""A real external handler with scripted decisions and observable callbacks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from typing import Any

from roomkit.tools import PolicyExternalToolHandler, ToolDecision, ToolPolicy


class ScriptedExternalHandler(PolicyExternalToolHandler):
    """Delegate to the policy handler unless the script supplies a decision."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        super().__init__(policy=ToolPolicy(**spec["policy"]) if "policy" in spec else None)
        self.spec = spec
        self.calls: list[dict[str, Any]] = []
        self.problems: list[str] = []
        self._decisions = iter(spec.get("decisions", []))

    def _record(self, method: str, **values: Any) -> None:
        self.calls.append(deepcopy({"method": method, "channel_id": self.channel_id, **values}))

    async def start(self) -> None:
        self._record("start")

    async def stop(self) -> None:
        self._record("stop")

    async def process_tool_call(
        self, tool_name: str, tool_input: dict[str, Any], **kwargs: Any
    ) -> ToolDecision:
        self._record("process_tool_call", name=tool_name, arguments=tool_input, **kwargs)
        if "decisions" not in self.spec:
            return await super().process_tool_call(tool_name, tool_input, **kwargs)
        decision = next(self._decisions, None)
        if decision is None:
            self.problems.append("external_tool_handler.decisions exhausted")
            raise RuntimeError("external handler fixture exhausted")
        if "raise" in decision:
            raise RuntimeError(decision["raise"])
        return ToolDecision(**decision)

    async def _report(self, delegate: Callable[[], Awaitable[None]]) -> None:
        mode = self.spec.get("reports", "delegate")
        if mode == "raise_before":
            raise RuntimeError("audit sink down")
        if mode != "skip":
            await delegate()
        if mode == "raise_after":
            raise RuntimeError("audit sink down")

    async def on_tool_result(
        self, tool_name: str, tool_input: dict[str, Any], result: str, **kwargs: Any
    ) -> None:
        self._record(
            "on_tool_result", name=tool_name, arguments=tool_input, result=result, **kwargs
        )
        parent = super().on_tool_result
        await self._report(lambda: parent(tool_name, tool_input, result, **kwargs))

    async def on_tool_refused(
        self, tool_name: str, tool_input: dict[str, Any], reason: str, **kwargs: Any
    ) -> None:
        self._record(
            "on_tool_refused", name=tool_name, arguments=tool_input, reason=reason, **kwargs
        )
        parent = super().on_tool_refused
        await self._report(lambda: parent(tool_name, tool_input, reason, **kwargs))

    async def on_tool_cancelled(
        self, tool_name: str, tool_input: dict[str, Any], **kwargs: Any
    ) -> None:
        self._record("on_tool_cancelled", name=tool_name, arguments=tool_input, **kwargs)
        parent = super().on_tool_cancelled
        await self._report(lambda: parent(tool_name, tool_input, **kwargs))


def handler_problems(spec: Any, path: str) -> list[str]:
    """Refuse malformed controls before an SDK callback can hide the error."""
    if not isinstance(spec, dict) or set(spec) - {"policy", "decisions", "reports"}:
        return [f"{path}: expected policy, decisions or reports"]
    found = []
    if "policy" in spec:
        policy = spec["policy"]
        if (
            not isinstance(policy, dict)
            or set(policy) - {"allow", "deny"}
            or any(
                not isinstance(patterns, list) or any(not isinstance(p, str) for p in patterns)
                for patterns in policy.values()
            )
        ):
            found.append(f"{path}.policy: expected allow/deny pattern lists")
    if "decisions" in spec:
        decisions = spec["decisions"]
        if not isinstance(decisions, list) or any(not _decision(d) for d in decisions):
            found.append(f"{path}.decisions: expected ToolDecision objects or raise strings")
    if spec.get("reports", "delegate") not in ("delegate", "raise_before", "raise_after", "skip"):
        found.append(f"{path}.reports: unknown report behavior")
    return found


def _decision(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "raise" in value:
        return set(value) == {"raise"} and isinstance(value["raise"], str)
    return (
        isinstance(value.get("approved"), bool)
        and not set(value) - {"approved", "modified_input", "result", "reason", "detail"}
        and (value.get("modified_input") is None or isinstance(value["modified_input"], dict))
        and all(value.get(k) is None or isinstance(value[k], str) for k in ("result", "detail"))
        and isinstance(value.get("reason", ""), str)
    )
