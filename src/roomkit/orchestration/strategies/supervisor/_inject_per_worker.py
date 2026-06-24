"""Per-worker delegation wiring for the supervisor.

Mixin for :class:`Supervisor`: injects one ``delegate_to_<id>`` tool per worker
and lets the AI decide when to delegate (manual mode). Host attributes are
declared as annotations; they are set in ``Supervisor.__init__``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies.supervisor._common import (
    _post_worker_status,
    logger,
)
from roomkit.providers.ai.base import AITool

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit


class _PerWorkerToolMixin:
    """Inject per-worker ``delegate_to_<id>`` tools (the AI decides)."""

    _supervisor: Agent
    _workers: list[Agent]
    _wait_for_result: bool
    _share_channels: list[str]

    def _inject_per_worker_tools(self, kit: RoomKit, room_id: str) -> None:
        """Inject per-worker ``delegate_to_<id>`` tools (AI decides)."""
        from roomkit.orchestration.handoff import _room_id_var

        any_new = False
        for worker in self._workers:
            tool_name = f"delegate_to_{worker.channel_id}"

            if any(t.name == tool_name for t in self._supervisor._injected_tools):
                continue

            any_new = True
            desc = getattr(worker, "description", None) or f"Worker agent {worker.channel_id}"
            tool = AITool(
                name=tool_name,
                description=f"Delegate a task to {worker.channel_id}. {desc}",
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Description of the task to delegate",
                        },
                    },
                    "required": ["task"],
                },
            )
            self._supervisor._injected_tools.append(tool)

        if not any_new:
            return

        original = self._supervisor.tool_handler
        tool_to_worker = {f"delegate_to_{w.channel_id}": w.channel_id for w in self._workers}
        wait = self._wait_for_result
        share_channels = self._share_channels
        pending: set[str] = set()

        async def delegation_handler(name: str, arguments: dict[str, Any]) -> str:
            worker_id = tool_to_worker.get(name)
            if worker_id is not None:
                rid = _room_id_var.get() or room_id
                task_desc = arguments.get("task", "")
                try:
                    if wait:
                        _post_worker_status(
                            kit,
                            worker_id,
                            StatusLevel.PENDING,
                            detail=task_desc,
                            metadata={"room_id": rid, "mode": "per_worker_wait"},
                        )
                        try:
                            delegated = await kit.delegate(
                                rid,
                                worker_id,
                                task_desc,
                                wait=True,
                                notify=self._supervisor.channel_id,
                                share_channels=share_channels,
                            )
                        except Exception as exc:
                            _post_worker_status(
                                kit,
                                worker_id,
                                StatusLevel.FAILED,
                                detail=str(exc),
                                metadata={"room_id": rid, "mode": "per_worker_wait"},
                            )
                            raise
                        result = delegated.result
                        result_status = result.status if result else "failed"
                        result_output = (result.output or result.error or "") if result else ""
                        _post_worker_status(
                            kit,
                            worker_id,
                            StatusLevel.COMPLETED
                            if result_status == "completed"
                            else StatusLevel.FAILED,
                            detail=result_output,
                            metadata={
                                "room_id": rid,
                                "mode": "per_worker_wait",
                                "task_id": delegated.id,
                            },
                        )
                        return json.dumps(
                            {
                                "status": result_status,
                                "worker": worker_id,
                                "result": result_output,
                            }
                        )

                    if worker_id in pending:
                        return json.dumps(
                            {
                                "status": "already_running",
                                "worker": worker_id,
                                "message": (
                                    f"{worker_id} is already working on this. "
                                    "Do NOT call this tool again. "
                                    "Tell the user to ask again shortly."
                                ),
                            }
                        )

                    delegated = await kit.delegate(
                        rid,
                        worker_id,
                        task_desc,
                        notify=self._supervisor.channel_id,
                        share_channels=share_channels,
                    )
                    pending.add(worker_id)
                    _post_worker_status(
                        kit,
                        worker_id,
                        StatusLevel.PENDING,
                        detail=task_desc,
                        metadata={
                            "room_id": rid,
                            "mode": "per_worker_async",
                            "task_id": delegated.id,
                        },
                    )

                    original_set = delegated._set_result
                    _bus_kit = kit
                    _bus_room = rid
                    _bus_task_id = delegated.id

                    def _patched_set(r: Any, *, _wid: str = worker_id or "") -> None:
                        pending.discard(_wid)
                        output = ""
                        ok = False
                        if r is not None:
                            output = getattr(r, "output", None) or getattr(r, "error", None) or ""
                            ok = getattr(r, "status", None) == "completed"
                        _post_worker_status(
                            _bus_kit,
                            _wid,
                            StatusLevel.COMPLETED if ok else StatusLevel.FAILED,
                            detail=output,
                            metadata={
                                "room_id": _bus_room,
                                "mode": "per_worker_async",
                                "task_id": _bus_task_id,
                            },
                        )
                        original_set(r)

                    delegated._set_result = _patched_set  # ty: ignore[invalid-assignment]

                    return json.dumps(
                        {
                            "status": "delegated",
                            "task_id": delegated.id,
                            "worker": worker_id,
                            "message": (
                                f"Task dispatched to {worker_id}. "
                                "It is running in the background. "
                                "Do NOT call this tool again. "
                                "Tell the user to ask again shortly "
                                "for results."
                            ),
                        }
                    )
                except Exception as exc:
                    logger.exception("Delegation to %s failed", worker_id)
                    return json.dumps({"error": str(exc)})
            if original:
                return await original(name, arguments)
            return json.dumps({"error": f"Unknown tool: {name}"})

        self._supervisor.tool_handler = delegation_handler
