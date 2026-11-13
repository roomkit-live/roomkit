"""Host context supplied through the ACP channel's public callback."""

from __future__ import annotations

from typing import Any

from roomkit import RoomContext, RoomEvent
from tests.conformance._script import Gates


def context_problems(spec: Any, path: str) -> list[str]:
    """Malformed fixtures must not pass as the callback's fail-open behaviour."""
    if not isinstance(spec, dict) or set(spec) != {"responses"}:
        return [f"{path}: expected a responses script"]
    if not isinstance(spec["responses"], list):
        return [f"{path}.responses: expected an array"]
    problems = []
    for index, reply in enumerate(spec["responses"]):
        valid = isinstance(reply, dict) and set(reply) <= {"blocks", "error", "gate"}
        if valid:
            valid = ("blocks" in reply) != ("error" in reply)
            if "blocks" in reply:
                valid &= isinstance(reply["blocks"], list) and all(
                    isinstance(block, str) for block in reply["blocks"]
                )
            if "error" in reply:
                valid &= isinstance(reply["error"], str)
            if "gate" in reply:
                valid &= isinstance(reply["gate"], str)
        if not valid:
            problems.append(f"{path}.responses[{index}]: invalid context response")
    return problems


class ScriptedContextContributor:
    """Record the actual context before awaiting any scripted gate."""

    def __init__(self, spec: Any, gates: Gates) -> None:
        problems = context_problems(spec, "acp.context_contributor")
        if problems:
            raise ValueError("; ".join(problems))
        self._replies = spec["responses"]
        self._gates = gates
        self.requests: list[dict[str, Any]] = []
        self.problems: list[str] = []

    async def __call__(self, context: RoomContext, event: RoomEvent) -> list[str]:
        index = len(self.requests)
        self.requests.append(
            {
                "room_id": context.room.id,
                "history_start_index": context.history_start_index,
                "event": event.model_dump(mode="json"),
                "recent_events": [item.model_dump(mode="json") for item in context.recent_events],
            }
        )
        if index >= len(self._replies):
            problem = f"acp.context_contributor: callback script exhausted at call {index + 1}"
            self.problems.append(problem)
            raise RuntimeError(problem)
        reply = self._replies[index]
        if "gate" in reply:
            await self._gates.hold(reply["gate"])
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return list(reply["blocks"])
