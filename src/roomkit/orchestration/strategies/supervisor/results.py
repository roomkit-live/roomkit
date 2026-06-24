"""Result rendering and presentation helpers for the supervisor strategy.

Pure text transforms: worker labels/profiles, structured ``submit_result``
rendering, and the briefs/digests handed to the supervisor for its
user-facing summary. No delegation or kit access.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from roomkit.models.channel import ChannelOutput
from roomkit.models.event import TextContent

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent


def _worker_label(worker: Agent) -> str:
    """Human-readable label for a worker, used to attribute its output when
    composing the next worker's input and the supervisor's review brief."""
    return (
        getattr(worker, "role", None) or getattr(worker, "description", None) or worker.channel_id
    )


_PROFILE_CAP = 600


def _worker_profile(worker: Agent) -> str:
    """Role + the agent's own configured instructions, so the supervisor frames
    tasks knowing each worker's real capabilities (e.g. that the report agent
    publishes HTML to the website) rather than just a label. Prefers ``purpose``
    (the agent's concise own instructions, set by the host) and falls back to
    ``description``; the full runtime system prompt is far too large to inject."""
    role = getattr(worker, "role", None) or worker.channel_id
    detail = getattr(worker, "purpose", None) or getattr(worker, "description", None)
    if detail and detail.strip() and detail.strip() != role.strip():
        text = detail.strip()
        if len(text) > _PROFILE_CAP:
            text = text[:_PROFILE_CAP] + "…"
        return f"{role} — {text}"
    return role


def _render_result(output: str) -> str:
    """Render a worker's structured ``submit_result`` payload (JSON) as readable
    text for the supervisor's review and the final digest. Falls back to the raw
    string when it isn't a structured payload."""
    try:
        payload = json.loads(output)
    except (ValueError, TypeError):
        return output
    if not isinstance(payload, dict) or "status" not in payload:
        return output
    if payload.get("by") == "orchestration":
        last = str(payload.get("last_output") or "")[:500]
        return (
            f"status: failed (orchestration: {payload.get('reason')})\n"
            f"the worker never returned a structured result; its last raw output:\n{last}"
        )
    parts = [f"status: {payload.get('status')}"]
    if payload.get("summary"):
        parts.append(f"summary: {payload['summary']}")
    if payload.get("data"):
        parts.append(f"data: {json.dumps(payload['data'], ensure_ascii=False)}")
    if payload.get("deliverables"):
        parts.append(f"deliverables: {json.dumps(payload['deliverables'], ensure_ascii=False)}")
    if payload.get("reason"):
        parts.append(f"reason: {payload['reason']}")
    return "\n".join(parts)


async def _extract_output_text(output: ChannelOutput) -> str:
    """Extract plain text from a ChannelOutput.

    Handles both synchronous response_events and streaming responses.
    For streaming, drains the stream to collect the full text.
    """
    # Check synchronous response first
    if output.response_events:
        for resp in output.response_events:
            if isinstance(resp.content, TextContent) and resp.content.body:
                return resp.content.body

    # Drain streaming response if present
    if output.response_stream is not None:
        parts: list[str] = []
        async for chunk in output.response_stream:
            if isinstance(chunk, str):
                parts.append(chunk)
        return "".join(parts)

    return ""


def _format_worker_results(results: list[dict[str, Any]]) -> str:
    """Format worker results as readable text for the supervisor's presentation.

    Prefers the worker's role as the label and, when the step carries a
    supervised verdict, surfaces its validation status so the supervisor's final
    summary can flag anything it could not validate.
    """
    parts: list[str] = []
    for r in results:
        label = r.get("role") or r.get("worker", "unknown")
        output = r.get("output", "")
        suffix = ""
        if "approved" in r:
            suffix = " (validated)" if r["approved"] else " (UNVALIDATED)"
        parts.append(f"--- {label}{suffix} ---\n{output}")
    return "\n\n".join(parts)


def _present_worker_results(worker_results: list[dict[str, Any]]) -> str:
    """The system message handed to the supervisor for its final user-facing
    summary. When every step passed it's a neutral hand-off; when any step FAILED
    it leads with an unmissable directive so the supervisor reports the failure
    instead of presenting the partial work as a finished answer (an LLM given rich
    upstream data will otherwise just answer the question and bury the failure)."""
    results_text = _format_worker_results(worker_results)
    failed = [r for r in worker_results if "approved" in r and not r["approved"]]
    if not failed:
        return f"Here are the results from your workers:\n\n{results_text}"
    names = ", ".join(str(r.get("role") or r.get("worker") or "a worker") for r in failed)
    return (
        f"⚠️ THE TASK DID NOT COMPLETE — this step FAILED: {names}.\n"
        "OPEN your reply by telling the user plainly that the task could not be "
        "completed, naming the step that failed and why (the failure detail is in the "
        "results below). Do NOT present the partial work as a finished result, do NOT "
        "imply the task succeeded, and do NOT silently answer the original question from "
        "the partial data as if nothing went wrong.\n\n"
        f"Worker results:\n{results_text}"
    )


def _format_supervisor_review(task_desc: str, result_json: str, workers: list[Agent]) -> str:
    """Re-present the workers' combined results to the supervisor as a review
    brief: the original request, then each worker's labeled output, with an
    instruction to verify the work and deliver one final answer (or flag what's
    missing). Returns the raw payload unchanged if it can't be parsed."""
    try:
        parsed = json.loads(result_json)
    except (ValueError, TypeError):
        return result_json
    labels = {w.channel_id: _worker_label(w) for w in workers}
    lines = [
        "Your team has finished. Review their work against the user's request "
        "below, then deliver ONE final answer to the user. If the work is "
        "incomplete or wrong, say what's missing — do not invent content.",
        "",
        f"User request:\n{task_desc}",
        "",
        "Team output:",
    ]
    for item in parsed.get("results", []):
        cid = item.get("worker", "")
        label = labels.get(cid) or cid or "worker"
        lines.append(f"\n--- {label} ---\n{item.get('output') or '(no output)'}")
    return "\n".join(lines)
