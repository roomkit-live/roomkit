"""Prompt, verdict, and digest builders for the supervised flow.

Pure functions: the verdict-format instruction, verdict parsing, rework
re-framing, the final digest, and the next-worker hand-off composition. No
delegation or kit access.
"""

from __future__ import annotations

import json
from typing import Any

from roomkit.orchestration.strategies.supervisor._common import logger

_VERDICT_INSTRUCTIONS = (
    "Respond with ONLY a JSON object, no other text, of exactly this shape:\n"
    '{"approved": true_or_false, "feedback": "what to fix if not approved, else empty", '
    '"next_task": "task for the next worker if approved and one exists, else empty"}'
)


def _parse_verdict(raw: str) -> dict[str, Any]:
    """Parse the supervisor's review JSON.

    Fails CLOSED (``approved=False``) on a parse miss: an unreadable verdict must
    not pass a step through unjudged. The miss is logged and fed back as feedback
    so the next attempt can correct itself; the surrounding rework loop is bounded
    by ``max_revisions``, so a persistently malformed verdict ends in an honest
    failure rather than a silent approval.
    """
    try:
        start = raw.index("{")
        end = raw.rindex("}")
        obj = json.loads(raw[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        logger.warning("Supervisor verdict unparseable; rejecting by default: %r", raw[:200])
        return {
            "approved": False,
            "feedback": "Your verdict was unreadable. Respond with ONLY the JSON verdict.",
            "next_task": None,
        }
    next_task = obj.get("next_task")
    return {
        "approved": bool(obj.get("approved", False)),
        "feedback": str(obj.get("feedback") or ""),
        "next_task": (str(next_task).strip() or None) if next_task else None,
    }


def _compose_rework(task: str, output: str, feedback: str) -> str:
    """Re-frame a worker's task after the supervisor rejected its output."""
    return (
        f"{task}\n\n"
        "--- Revision requested by the supervisor ---\n"
        f"Your previous attempt was NOT accepted. Feedback:\n{feedback}\n\n"
        f"Your previous output (for reference):\n{output}\n\n"
        "Produce a corrected, complete result that addresses the feedback."
    )


def _format_supervised_digest(goal: str, steps: list[dict[str, Any]], max_revisions: int) -> str:
    """Brief handed back to the supervisor's own turn so it writes the final
    user-facing summary — each step's validated output + validation status."""
    aborted = bool(steps) and not steps[-1]["approved"]
    if aborted:
        intro = (
            "Your team could NOT complete the task. The final step below FAILED after "
            f"{max_revisions} attempts, so the chain was STOPPED — later workers did not "
            "run. Tell the user HONESTLY that the task could not be completed: name the "
            "step that failed and why, and summarize what was accomplished before it. Do "
            "NOT fabricate a finished result or a deliverable that does not exist."
        )
    else:
        intro = (
            "Your team has finished and you have reviewed each step. Deliver ONE final "
            "summary to the user: what each step accomplished and the outcome. Reference "
            "any deliverables (published reports/artifacts) by their link."
        )
    lines = [intro, "", f"User request:\n{goal}", "", "Reviewed work:"]
    for step in steps:
        status = "validated" if step["approved"] else f"FAILED after {max_revisions} attempts"
        lines.append(f"\n--- {step['role']} ({status}) ---\n{step['output'] or '(no output)'}")
    return "\n".join(lines)


def _compose_supervised_handoff(framing: str, prior_steps: list[dict[str, Any]]) -> str:
    """Build the next worker's task: the supervisor's framing PLUS the team's
    validated work embedded verbatim.

    The supervisor's ``next_task`` says WHAT the next worker must do, but it
    references prior results in prose ("build the report from the analyst's
    data") — an LLM won't reliably paste the content. So the supervisor curates
    the instruction and the code carries the data: each prior worker's rendered
    result is attached. Without this the next worker gets a task pointing at data
    it never sees and reports it as missing."""
    if not prior_steps:
        return framing
    blocks = [framing, "", "--- Work already completed by the team (build on this) ---"]
    for step in prior_steps:
        blocks.append(f"\n[{step['role']}]:\n{step['output'] or '(no output)'}")
    return "\n".join(blocks)
