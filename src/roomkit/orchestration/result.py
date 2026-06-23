"""Structured result handoff for orchestration.

A delegated agent returns its work by calling the ``submit_result`` tool rather
than ending its turn with a free-text message that gets scraped from the room.
This forces a structured, parseable handoff — and a result at all (a worker
can't punt with a question) — so the next step and the supervisor receive a
clean object instead of prose. Reusable by any orchestration that delegates.
"""

from __future__ import annotations

from typing import Any

from roomkit.providers.ai.base import AITool

SUBMIT_RESULT_TOOL_NAME = "submit_result"

#: Injected into a delegated worker so it delivers its result through a tool
#: call (forced structure) instead of a scraped free-text message.
SUBMIT_RESULT_TOOL = AITool(
    name=SUBMIT_RESULT_TOOL_NAME,
    description=(
        "Submit your FINAL result for this task. You MUST call this exactly once, "
        "when your work is done — it is the ONLY way to hand your work to the next "
        "step. Do NOT end your turn with a plain message or a question back to the "
        "user; call submit_result with your structured result."
    ),
    parameters={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["completed", "failed"],
                "description": (
                    "completed if you did the task; failed only if you genuinely could not."
                ),
            },
            "summary": {
                "type": "string",
                "description": "One or two sentences summarizing what you produced.",
            },
            "data": {
                "type": "object",
                "description": "Your structured result, for the next step to build on.",
            },
            "deliverables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                    },
                },
                "description": "Concrete artifacts you produced (e.g. a published report URL).",
            },
            "reason": {
                "type": "string",
                "description": "If status is failed, explain why.",
            },
        },
        "required": ["status", "summary"],
    },
)


def normalize_result(arguments: dict[str, Any]) -> dict[str, Any]:
    """Coerce a ``submit_result`` tool-call payload into the canonical shape."""
    data = arguments.get("data")
    deliverables = arguments.get("deliverables")
    return {
        "status": arguments.get("status") or "completed",
        "summary": str(arguments.get("summary") or ""),
        "data": data if isinstance(data, dict) else {},
        "deliverables": deliverables if isinstance(deliverables, list) else [],
        "reason": str(arguments.get("reason") or ""),
    }


def orchestration_fail(*, role: str, last_output: str, attempts: int) -> dict[str, Any]:
    """Fail payload the orchestration submits on a worker's behalf when it never
    called ``submit_result`` after *attempts* tries.

    Distinct from a worker self-reporting a task failure (``status="failed"`` via
    the tool): ``by="orchestration"`` marks a MECHANISM-level failure, carrying
    the worker's last raw output as explanatory context so the next step and the
    supervisor understand precisely what went wrong.
    """
    return {
        "status": "failed",
        "by": "orchestration",
        "reason": f"no_structured_result_after_{attempts}_attempts",
        "role": role,
        "last_output": last_output,
    }
