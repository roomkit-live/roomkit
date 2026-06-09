"""Public accessors for per-call tool execution context.

An ``AIChannel`` object is registered once per ``channel_id`` and shared
by every room it serves, so any room-specific state stored on the channel
(or on a host's tool handler) goes stale the moment another room attaches.
The tool loop scopes its per-invocation state in a contextvar instead;
this module exposes the parts of that state a host's tool handler may
need to resolve the call's origin.

Contextvars propagate through the async call chain, so a handler invoked
from inside a tool loop sees the loop's context without any signature
change. Outside a tool loop (realtime voice pipelines, direct calls) the
accessors return ``None`` — hosts keep their own fallback for those paths.
"""

from __future__ import annotations


def current_tool_room_id() -> str | None:
    """Room id of the tool loop the caller is executing under.

    Returns ``None`` when called outside a tool loop.
    """
    from roomkit.channels.ai import _current_loop_ctx

    ctx = _current_loop_ctx.get()
    return ctx.room_id if ctx is not None else None


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
    if ctx is None or not ctx.all_context_tools:
        return None
    return {t.name for t in ctx.all_context_tools if getattr(t, "name", None)}
