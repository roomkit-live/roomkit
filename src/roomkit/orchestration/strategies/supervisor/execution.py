"""Worker execution for the deterministic strategies (sequential / parallel).

Runs workers directly via ``kit.delegate`` with a per-task timeout and posts
lifecycle events. The supervised hub-&-spoke variant lives in
``supervised.py``.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies.supervisor._common import (
    _DEFAULT_TASK_TIMEOUT_SECONDS,
    _post_worker_status,
)
from roomkit.orchestration.strategies.supervisor.results import _worker_label

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit


def _compose_sequential_input(task_desc: str, prior_steps: list[tuple[str, str]]) -> str:
    """Build a worker's input for sequential delegation.

    The first worker (no prior steps) gets the task unchanged. Every later
    worker gets the original task plus each prior worker's labeled output, so
    the goal and accumulated work survive the chain — a worker handed only its
    predecessor's raw output has no task to act on and just converses with it.
    """
    if not prior_steps:
        return task_desc
    blocks = [f"Original task:\n{task_desc}", "", "Work already completed by the team:"]
    for label, output in prior_steps:
        blocks.append(f"\n--- {label} ---\n{output or '(no output)'}")
    blocks.append("\nBuild on the work above to complete your part of the original task.")
    return "\n".join(blocks)


async def _run_sequential(
    kit: RoomKit,
    room_id: str,
    workers: list[Agent],
    task_desc: str,
    *,
    share_channels: list[str] | None = None,
    task_timeout: float = _DEFAULT_TASK_TIMEOUT_SECONDS,
) -> str:
    """Run workers in order, each receiving the original task plus every prior
    worker's output (see :func:`_compose_sequential_input`).

    Each delegation is bounded by *task_timeout*; a worker that exceeds it is
    recorded as failed and the chain continues so the rest of the team — and
    the supervisor's review — still runs on partial results."""
    results: list[dict[str, str]] = []
    prior_steps: list[tuple[str, str]] = []

    for worker in workers:
        worker_input = _compose_sequential_input(task_desc, prior_steps)
        _post_worker_status(
            kit,
            worker.channel_id,
            StatusLevel.PENDING,
            detail=worker_input,
            metadata={"room_id": room_id, "strategy": "sequential"},
        )
        try:
            delegated = await asyncio.wait_for(
                kit.delegate(
                    room_id,
                    worker.channel_id,
                    worker_input,
                    wait=True,
                    share_channels=share_channels,
                ),
                timeout=task_timeout,
            )
        except TimeoutError:
            timeout_msg = f"Worker timed out after {task_timeout:.0f}s"
            _post_worker_status(
                kit,
                worker.channel_id,
                StatusLevel.FAILED,
                detail=timeout_msg,
                metadata={"room_id": room_id, "strategy": "sequential"},
            )
            results.append({"worker": worker.channel_id, "output": timeout_msg})
            prior_steps.append((_worker_label(worker), timeout_msg))
            continue
        except Exception as exc:
            _post_worker_status(
                kit,
                worker.channel_id,
                StatusLevel.FAILED,
                detail=str(exc),
                metadata={"room_id": room_id, "strategy": "sequential"},
            )
            raise
        output = ""
        status_ok = False
        if delegated.result:
            output = delegated.result.output or delegated.result.error or ""
            status_ok = delegated.result.status == "completed"
        _post_worker_status(
            kit,
            worker.channel_id,
            StatusLevel.COMPLETED if status_ok else StatusLevel.FAILED,
            detail=output,
            metadata={
                "room_id": room_id,
                "strategy": "sequential",
                "task_id": delegated.id,
            },
        )
        results.append({"worker": worker.channel_id, "output": output})
        prior_steps.append((_worker_label(worker), output))

    return json.dumps({"status": "completed", "results": results})


async def _run_parallel(
    kit: RoomKit,
    room_id: str,
    workers: list[Agent],
    task_desc: str,
    *,
    share_channels: list[str] | None = None,
    task_timeout: float = _DEFAULT_TASK_TIMEOUT_SECONDS,
) -> str:
    """Run all workers concurrently on the same task. Each is bounded by
    *task_timeout*; one that exceeds it is recorded as failed without aborting
    its siblings."""

    async def _delegate_one(worker: Agent) -> dict[str, str]:
        _post_worker_status(
            kit,
            worker.channel_id,
            StatusLevel.PENDING,
            detail=task_desc,
            metadata={"room_id": room_id, "strategy": "parallel"},
        )
        try:
            delegated = await asyncio.wait_for(
                kit.delegate(
                    room_id,
                    worker.channel_id,
                    task_desc,
                    wait=True,
                    share_channels=share_channels,
                ),
                timeout=task_timeout,
            )
        except TimeoutError:
            timeout_msg = f"Worker timed out after {task_timeout:.0f}s"
            _post_worker_status(
                kit,
                worker.channel_id,
                StatusLevel.FAILED,
                detail=timeout_msg,
                metadata={"room_id": room_id, "strategy": "parallel"},
            )
            return {"worker": worker.channel_id, "output": timeout_msg}
        except Exception as exc:
            _post_worker_status(
                kit,
                worker.channel_id,
                StatusLevel.FAILED,
                detail=str(exc),
                metadata={"room_id": room_id, "strategy": "parallel"},
            )
            raise
        output = ""
        status_ok = False
        if delegated.result:
            output = delegated.result.output or delegated.result.error or ""
            status_ok = delegated.result.status == "completed"
        _post_worker_status(
            kit,
            worker.channel_id,
            StatusLevel.COMPLETED if status_ok else StatusLevel.FAILED,
            detail=output,
            metadata={
                "room_id": room_id,
                "strategy": "parallel",
                "task_id": delegated.id,
            },
        )
        return {"worker": worker.channel_id, "output": output}

    results = await asyncio.gather(*[_delegate_one(w) for w in workers])
    return json.dumps({"status": "completed", "results": list(results)})
