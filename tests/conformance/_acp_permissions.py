"""Scripted permission exchanges, observed on the ACP wire."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any


class PermissionExchanges:
    """Await each requested decision before the agent continues its prompt."""

    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []
        self._pending: dict[str, tuple[asyncio.Future[None], str, str]] = {}
        self._next_id = 0

    def accept(self, message: Mapping[str, Any]) -> None:
        pending = self._pending.get(str(message.get("id")))
        if pending is None:
            return
        future, session, tool_id = pending
        if not future.done():
            self.received.append(
                {
                    "session_id": session,
                    "tool_call_id": tool_id,
                    **{key: message[key] for key in ("result", "error") if key in message},
                }
            )
            future.set_result(None)

    async def exchange(
        self, session: str, request: Mapping[str, Any], writer: asyncio.StreamWriter
    ) -> None:
        self._next_id += 1
        request_id = f"awaited-permission-{self._next_id}"
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = (future, session, request["toolCall"]["toolCallId"])
        try:
            writer.write(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "session/request_permission",
                        "params": {**request, "sessionId": session},
                    }
                ).encode()
                + b"\n"
            )
            await future
        finally:
            self._pending.pop(request_id, None)


def permission_problems(prompt: Mapping[str, Any], path: str) -> list[str]:
    """Reject unusable scripts before an agent task starts waiting."""
    if "permission_steps" not in prompt:
        return []
    steps = prompt["permission_steps"]
    if not isinstance(steps, list):
        return [f"{path}.permission_steps: expected a list"]
    found = []
    for index, step in enumerate(steps):
        at = f"{path}.permission_steps[{index}]"
        if not isinstance(step, dict) or set(step) - {"request", "updates"}:
            found.append(f"{at}: expected request and optional updates")
            continue
        request = step.get("request")
        if not isinstance(request, dict) or set(request) - {"toolCall", "options"}:
            found.append(f"{at}.request: expected an ACP permission request")
            continue
        call = request.get("toolCall")
        if not isinstance(call, dict) or not isinstance(call.get("toolCallId"), str):
            found.append(f"{at}.request.toolCall: expected a toolCallId string")
        options = request.get("options")
        if not isinstance(options, list):
            found.append(f"{at}.request.options: expected a list")
        elif any(not _valid_option(option) for option in options):
            found.append(f"{at}.request.options: expected named ACP permission options")
        updates = step.get("updates", [])
        if not isinstance(updates, list) or any(not isinstance(item, dict) for item in updates):
            found.append(f"{at}.updates: expected a list of updates")
    return found


def _valid_option(option: Any) -> bool:
    return (
        isinstance(option, dict)
        and isinstance(option.get("optionId"), str)
        and isinstance(option.get("name"), str)
        and option.get("kind") in ("allow_once", "allow_always", "reject_once", "reject_always")
    )
