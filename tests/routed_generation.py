"""Run one generation of an AIChannel the way its turn runs it.

A turn's provider is the one its channel's router picked when it started
(RFC §6.4), held by the turn's loop context. A test that drives the
channel's retry wrapper outside a turn opens that context first with
:func:`routed_generation`, so a channel built with a router has a provider to
run on, as it would in a turn.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from roomkit.channels._ai_router import messages_request
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext
from roomkit.tools.context import _current_loop_ctx, _ToolLoopContext


async def routed_generation(channel: AIChannel, context: AIContext) -> AsyncIterator[Any]:
    """The events of one generation on *context* through *channel*'s retry
    and re-routing, in a turn routed on its messages."""
    loop_ctx = _ToolLoopContext()
    token = _current_loop_ctx.set(loop_ctx)
    try:
        await channel._route_turn(loop_ctx, messages_request(context.messages, None))
        async for event in channel._generate_stream_with_retry(context):
            yield event
    finally:
        _current_loop_ctx.reset(token)
