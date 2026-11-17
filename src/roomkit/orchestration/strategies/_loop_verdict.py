"""A Loop reviewer's verdict (RFC §19.7.4): the tool it is handed back
through, and how it is read.

A reviewer judges the producer's output by calling ``submit_review`` once,
forced as a worker's ``submit_result`` and a supervisor's ``submit_verdict``
are (re-prompts included). The verdict fails closed: a review that never
called the tool, a reviewer channel that answered in text, or a task that
failed approves nothing, whatever words its text holds.
"""

from __future__ import annotations

import json
from typing import Any

from roomkit.orchestration.result import ResultTool
from roomkit.providers.ai.base import AITool


def _normalize_review(arguments: dict[str, Any]) -> dict[str, Any]:
    """A ``submit_review`` call's arguments as a verdict: approved only when
    ``approved`` is the boolean true."""
    return {
        "approved": arguments.get("approved") is True,
        "feedback": str(arguments.get("feedback") or ""),
    }


def _no_review(*, role: str, last_output: str, attempts: int) -> dict[str, Any]:
    """The verdict of a reviewer that never called ``submit_review``: not
    approved, its last text kept as its feedback."""
    return {"approved": False, "feedback": last_output}


SUBMIT_REVIEW = ResultTool(
    tool=AITool(
        name="submit_review",
        description=(
            "Submit your review of the content. You MUST call this exactly once; it is "
            "the ONLY way your judgement reaches the producer."
        ),
        parameters={
            "type": "object",
            "properties": {
                "approved": {
                    "type": "boolean",
                    "description": "true only if the content meets the standard as it is.",
                },
                "feedback": {
                    "type": "string",
                    "description": "What to change, precisely, when not approved; else empty.",
                },
            },
            "required": ["approved", "feedback"],
        },
    ),
    normalize=_normalize_review,
    on_missing=_no_review,
    reminder=(
        "You did not submit your review. You MUST now call the `submit_review` tool "
        "with approved and feedback. Do NOT reply with plain text."
    ),
)


def read_review(output: str) -> dict[str, Any]:
    """The verdict a review's task handed back: the JSON payload of its
    ``submit_review`` call. Anything else (a reviewer channel that answered in
    text, a task that failed and handed back nothing) is no approval, its text
    the feedback."""
    try:
        payload = json.loads(output)
    except (TypeError, ValueError):
        payload = None
    if not isinstance(payload, dict) or "approved" not in payload:
        return {"approved": False, "feedback": output}
    return _normalize_review(payload)
