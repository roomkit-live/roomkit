"""Per-call tool execution context, and the public accessors for it.

An ``AIChannel`` object is registered once per ``channel_id`` and shared
by every room it serves, so any room-specific state stored on the channel
(or on a host's tool handler) goes stale the moment another room attaches.
The tool loop scopes its per-invocation state in a contextvar instead;
this module holds that contextvar and exposes the parts of the state a
host's tool handler may need to resolve the call's origin.

Contextvars propagate through the async call chain, so a handler invoked
from inside a tool loop sees the loop's context without any signature
change. Outside a tool loop (realtime voice pipelines, direct calls) the
accessors return ``None`` — hosts keep their own fallback for those paths.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any

from roomkit.models.response_metadata import ResponseMetadata


@dataclass
class ToolCallContext:
    """Contextvar payload carrying tool-call metadata.

    The ToolHandler protocol is ``(name, arguments) → str`` — it does not
    receive ``room_id``, ``tool_call_id`` or ``channel_id``.  This payload
    bridges the gap: ``_ai_tools._run_one()`` sets it before calling the
    handler, and a handler that needs the call's origin reads it.  Safe
    with :func:`asyncio.gather`, which creates Tasks with copied contexts.

    ``structured_content`` is the reverse channel: the ToolHandler contract
    returns only a string, but MCP tools can produce a structured result
    (``CallToolResult.structuredContent``) that UI surfaces need verbatim —
    the LLM-facing string may be truncated/evicted when large. A handler
    that has one sets it here; ``_run_one()`` reads it back after the call
    and carries it on the tool-call events untouched by eviction.
    """

    room_id: str = ""
    tool_call_id: str = ""
    channel_id: str = ""
    structured_content: dict[str, Any] | None = None


_current_tool_call: contextvars.ContextVar[ToolCallContext | None] = contextvars.ContextVar(
    "_current_tool_call", default=None
)


def current_tool_room_id() -> str | None:
    """Room id of the tool loop the caller is executing under.

    Returns ``None`` when called outside a tool loop.
    """
    from roomkit.channels.ai import _current_loop_ctx

    ctx = _current_loop_ctx.get()
    return ctx.room_id if ctx is not None else None


def current_tool_actor_id() -> str | None:
    """Participant id of whoever's turn the caller is executing under.

    The author of the event that woke the channel this round. Read it rather
    than the identity a handler captured when it was built — one channel
    object serves every room and every speaker, so a captured identity is
    whoever happened to attach it.

    It names the turn; it does not authenticate it. The value is a room
    ``Participant.id``, and the inbound pipeline only substitutes the resolved
    ``Identity.id`` for it once identification succeeds — a turn still pending,
    ambiguous or unknown carries whatever the channel supplied, or a synthetic
    ``pending-…``, and reads back just as non-``None``. A handler that reaches
    a person's data with it resolves it first: load the participant, require
    ``Participant.identification`` to be ``IDENTIFIED``, and take
    ``Participant.identity_id`` as the principal.

    The author need not be human, either. In a multi-agent room the waking
    event may be another agent's, whose participant id reads back the same
    way — compare the participant's ``role`` against ``ParticipantRole.AGENT``
    when that distinction matters.

    ``None`` outside a tool loop, and ``None`` when the turn has no
    participant behind it (a system injection, a webhook, a scheduled run).
    A caller that needs a person then decides for itself — refuse, or fall
    back to a principal it configured on purpose — rather than borrow whoever
    spoke last.
    """
    from roomkit.channels.ai import _current_loop_ctx

    ctx = _current_loop_ctx.get()
    return ctx.actor_id if ctx is not None else None


def current_tool_allowed_names() -> set[str] | None:
    """Names of every tool in the current turn's resolved toolset.

    ``_build_context`` stamps the turn's full toolset (config-provider
    result plus channel-injected tools) into the loop context; a host's
    tool handler can validate an incoming call against it instead of an
    attach-time snapshot that goes stale on shared channels. Includes
    skill-gated tools whose *visibility* is filtered per round — gating
    is presentation, not an execution boundary.

    Returns ``None`` outside a tool loop or before context build, so
    hosts can fall back to their own allowlist.
    """
    from roomkit.channels.ai import _current_loop_ctx

    ctx = _current_loop_ctx.get()
    if ctx is None or ctx.all_context_tools is None:
        return None
    return {t.name for t in ctx.all_context_tools if getattr(t, "name", None)}


def current_response_metadata() -> ResponseMetadata | None:
    """The response-metadata record of the turn the caller is executing under.

    The one mapping RoomKit merges into every MESSAGE event the turn produces
    (see :mod:`roomkit.models.response_metadata`): a memory provider writing it
    during context build, a ``BEFORE_AI_GENERATION`` hook writing
    ``event.ai_context.response_metadata`` and a tool handler writing here all
    reach the same object. A tool handler is the case this exists for — the
    ``ToolHandler`` protocol hands it nothing but ``(name, arguments)``, and a
    document it read is a fact about the turn, not about the tool's string
    result.

    Returns ``None`` outside a turn (a realtime pipeline, a direct call): the
    caller then has nothing to attribute to, and writes nothing.
    """
    from roomkit.channels.ai import _current_loop_ctx

    ctx = _current_loop_ctx.get()
    return ctx.response_metadata if ctx is not None else None
