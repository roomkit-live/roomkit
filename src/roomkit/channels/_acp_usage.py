"""Normalization and provenance of ACP usage observations."""

from __future__ import annotations

import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from uuid import uuid4

from roomkit.channels._acp_client import _model_dump

_TOKEN_FIELDS = (
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "thought_tokens",
    "cached_read_tokens",
    "cached_write_tokens",
)


def _usage_tokens(usage: Any) -> dict[str, int]:
    """Read the ``Usage`` an agent returns when a prompt ends.

    Taken off the model by attribute rather than out of :func:`_model_dump`,
    whose camelCase aliases would not sit beside the counters an in-process
    provider reports under the same key.

    Relayed exactly as the agent sent it. The ACP schema annotates these
    fields as running session figures ("total input tokens across all turns")
    while the reference agent fills them per prompt — measured against it,
    ``cached_read_tokens`` is the whole prefix re-read on that turn, not a sum
    over turns. RoomKit cannot tell the two apart from one reading, and
    guessing wrong corrupts the number in the direction nobody can detect
    downstream, so it does no arithmetic on them at all.
    """
    counters: dict[str, int] = {}
    for name in _TOKEN_FIELDS:
        value = getattr(usage, name, None)
        if isinstance(value, int):
            counters[name] = value
    return counters


def _usage_context(update: Any) -> dict[str, Any]:
    """Read a usage notification: how full the context is, what it has cost.

    A different quantity from the token counters above — ``used``/``size``
    describe the window the session is living in, and ``cost`` is its running
    total. Occupancy/capacity can fall after compaction or reconfiguration;
    neither is a count of tokens consumed by the prompt.
    """
    context: dict[str, Any] = {}
    used = _field(update, "used")
    if type(used) is int:
        context["context_used"] = used
    size = _field(update, "size")
    if type(size) is int:
        context["context_size"] = size
    cost = _field(update, "cost")
    amount = _field(cost, "amount")
    if type(amount) in (int, float):
        context["cost"] = float(amount)
    currency = _field(cost, "currency")
    if isinstance(currency, str) and currency:
        context["currency"] = currency
    return context


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _report_context(report: Any) -> dict[str, Any]:
    """Normalize the context/cost carried by a session observation."""
    return _usage_context(_field(report, "update"))


def _transport_usage(value: Any) -> dict[str, Any] | None:
    """Read the opt-in transport envelope, never arbitrary agent extensions."""
    meta = _field(value, "field_meta") or _field(value, "_meta")
    envelope = _field(meta, "roomkit.live/usage")
    return deepcopy(dict(envelope)) if isinstance(envelope, Mapping) else None


_SESSION_IDENTITY_FIELDS = (
    "session_id",
    "session_epoch",
    "usage_protocol",
    "node_id",
    "agent_id",
    "adapter_info",
)
_USAGE_IDENTITY_FIELDS = (
    *_SESSION_IDENTITY_FIELDS,
    "result_id",
    "turn_id",
    "generation",
    "replayed",
)


def _apply_transport_usage(
    metadata: dict[str, Any],
    envelope: dict[str, Any],
    *,
    terminal: bool = True,
) -> None:
    """A terminal snapshot is authoritative, including an absent report.

    Recovery can return another session's result. Never pair its counters
    with the current session's live cost or model. A report's source_result_id
    is its origin, even when it differs from this response's result_id.
    """
    # A notification's result ID describes its origin, not the active prompt.
    # Only a prompt response can establish the result identity at the root.
    identities = _USAGE_IDENTITY_FIELDS if terminal else _SESSION_IDENTITY_FIELDS
    for key in (*identities, "usage_report"):
        metadata.pop(key, None)
        if key in envelope:
            metadata[key] = deepcopy(envelope[key])
    metadata["identity_source"] = "transport"


def _observe_usage(update: Any, model: str | bool | None) -> dict[str, Any]:
    """Identify a local receipt without inventing a source prompt or durability."""
    report: dict[str, Any] = {
        "report_id": uuid4().hex,
        "identity_source": "roomkit",
        "observed_at_ms": time.time_ns() // 1_000_000,
        "source": "session/update",
        "scope": "session",
        "update": deepcopy(_model_dump(update)),
    }
    if isinstance(model, str):
        report["model_at_observation"] = model
    return report


def _usage_report(tokens: dict[str, int], context: dict[str, Any]) -> dict[str, Any]:
    """The accounting a finished turn carries: what the agent counted, and where.

    Two readings side by side under distinct keys — the token counters from
    the prompt's own response, and the session's context occupancy and running
    cost. Both are the agent's figures, unaltered; see :func:`_usage_tokens`
    for why nothing here is differenced.
    """
    return {**tokens, **context}
