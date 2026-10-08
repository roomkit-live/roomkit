"""One ACP turn: a prompt sent to a session, streamed until the agent ends it.

The agent's updates arrive on the connection (``ACPEventsMixin``) and are put
on the turn's queue; this mixin is the other end of that queue. It composes the
prompt, runs it, yields what the agent says as stream deltas, records how the
turn ended on the response metadata, and closes whatever the turn left open.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from roomkit.channels._acp_client import _SDK, _TurnDone, _TurnState
from roomkit.channels._acp_context import compose_prompt, labelled_request, room_context_block
from roomkit.channels._acp_usage import (
    _apply_transport_usage,
    _report_context,
    _transport_usage,
    _usage_report,
    _usage_tokens,
)
from roomkit.channels.acp_transport import ACPSessionInvalidatedError
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.models.response_metadata import ResponseMetadata, recorded_turn_end
from roomkit.models.streaming import StreamDelta
from roomkit.models.tool_call import AfterResponseCallback, AIResponseEvent, response_transcript
from roomkit.providers.ai.base import ProviderError
from roomkit.realtime.base import EphemeralEventType

if TYPE_CHECKING:
    from roomkit.channels.acp_transport import ACPTransport

logger = logging.getLogger("roomkit.channels.acp")

_CLEAN_STOP_REASON = "end_turn"
"""The one ACP stop reason that means the agent finished what it was asked."""

_UNSEEN = -1
"""No prompt has left for this room yet — event indices start at 0."""


class ACPTurnMixin:
    """Run one prompt per turn and stream its outcome."""

    channel_id: str
    _transport: ACPTransport
    _agent_info: dict[str, Any] | None
    _turns: dict[str, _TurnState]
    _prompted_index: dict[str, int]
    _room_history: int
    _after_response_hook: AfterResponseCallback | None
    _closed: bool

    # Implemented by the other mixins of the channel. Annotations, never stub
    # methods: a stub would shadow an implementation later in the MRO. An
    # ``async def`` is declared as returning a ``Coroutine``, not an
    # ``Awaitable``: mypy requires a name two bases define to be compatible in
    # MRO order, so it rejects an application's ``class X(Mixin, ACPChannel)``
    # when an ``Awaitable`` annotation precedes the implementation.
    _sdk: Callable[[], _SDK]
    _ensure_connection: Callable[[], Coroutine[Any, Any, Any]]
    _drain_session_updates: Callable[[str], Coroutine[Any, Any, None]]
    _room_turn_lock: Callable[[str], AbstractAsyncContextManager[None]]
    _session_for: Callable[[str, Any], Coroutine[Any, Any, str]]
    _discard_room_session: Callable[[str, Any], Coroutine[Any, Any, bool]]
    _open_turn_session: Callable[[str, Any], Coroutine[Any, Any, str]]
    _close_turn_session: Callable[[str, str, Any], Coroutine[Any, Any, None]]
    session_config: Callable[[str], dict[str, str | bool]]
    _close_open_tools: Callable[..., Coroutine[Any, Any, bool]]
    _publish: Callable[..., Coroutine[Any, Any, None]]
    _room_locks: dict[str, asyncio.Lock]

    async def _prompt_stream(
        self,
        room_id: str,
        event_id: str,
        blocks: Sequence[str],
        context: RoomContext,
        trigger: RoomEvent,
        text: str,
        seen_index: int,
        metadata: ResponseMetadata,
        *,
        standalone: bool = False,
    ) -> AsyncIterator[StreamDelta]:
        async with self._room_turn_lock(room_id):
            connection = await self._ensure_connection()
            if standalone:
                session_id = await self._open_turn_session(room_id, connection)
            else:
                session_id = await self._session_for(room_id, connection)
            recovering = False
            completed = False
            try:
                for attempt in range(2):
                    turn_stream = self._turn_stream(
                        room_id,
                        session_id,
                        connection,
                        event_id,
                        blocks,
                        context,
                        trigger,
                        text,
                        seen_index,
                        metadata,
                        standalone=standalone,
                    )
                    try:
                        async for item in turn_stream:
                            yield item
                        completed = not metadata["acp"].get("interrupted", False)
                        return
                    except ACPSessionInvalidatedError as exc:
                        if standalone or attempt or exc.recovery_authorized is not True:
                            raise
                    finally:
                        # Cleanup (runner, tools, maps) precedes reconstruction.
                        await turn_stream.aclose()
                    recovering = True
                    rebuilt = await self._rebuild_session(room_id, session_id, connection)
                    if rebuilt is None:
                        return
                    session_id = rebuilt
                    # A new turn owns its outcome; keep the first failure marked
                    # until opening succeeds, then let the retry write its own.
                    for key in ("interrupted", "stop_reason", "prompt_returned"):
                        metadata["acp"].pop(key, None)
            finally:
                if standalone:
                    await self._close_turn_session(room_id, session_id, connection)
                elif recovering and not completed:
                    metadata["acp"]["interrupted"] = True
                    try:
                        await self._discard_room_session(room_id, connection)
                    finally:
                        self._room_locks.pop(room_id, None)

    async def _rebuild_session(self, room_id: str, session_id: str, connection: Any) -> str | None:
        """Keep reconstruction in the turn registry so public cancel can stop it."""
        turn = _TurnState(room_id=room_id, rebuilding=True)
        turn.runner = asyncio.create_task(self._replace_session(room_id, connection))
        self._turns[session_id] = turn
        try:
            replacement = await turn.runner
            # A cancel can land after the runner finishes but before we wake.
            return None if self._closed or turn.cancel_requested else replacement
        except asyncio.CancelledError:
            if not turn.cancel_requested:
                raise
            return None
        finally:
            if self._turns.get(session_id) is turn:
                self._turns.pop(session_id, None)

    async def _replace_session(self, room_id: str, connection: Any) -> str:
        await self._discard_room_session(room_id, connection)
        if self._closed:
            raise asyncio.CancelledError
        session_id = await self._session_for(room_id, connection)
        if self._closed:
            raise asyncio.CancelledError
        return session_id

    async def _turn_stream(
        self,
        room_id: str,
        session_id: str,
        connection: Any,
        event_id: str,
        blocks: Sequence[str],
        context: RoomContext,
        trigger: RoomEvent,
        text: str,
        seen_index: int,
        metadata: ResponseMetadata,
        *,
        standalone: bool,
    ) -> AsyncGenerator[StreamDelta]:
        """One prompt in *session_id*, streamed until the agent ends it."""
        prompt_source: dict[str, Any] = {"source": "session/prompt", "scope": "unspecified"}
        model = self.session_config(room_id).get("model")
        if isinstance(model, str):
            prompt_source["model_at_start"] = model
        turn = _TurnState(
            room_id=room_id,
            usage_metadata={
                "protocol": "acp",
                "transport": self._transport.name,
                "session_id": session_id,
                "event_id": event_id,
                "prompt": prompt_source,
            },
        )
        if self._agent_info is not None:
            turn.usage_metadata["adapter_info"] = deepcopy(self._agent_info)
        self._turns[session_id] = turn
        # A standalone turn reads nothing of the room: no catch-up, and
        # its session was born empty.
        catch_up = (
            ""
            if standalone
            else room_context_block(
                context,
                self.channel_id,
                after_index=self._prompted_index.get(room_id, _UNSEEN),
                trigger=trigger,
                limit=self._room_history,
            )
        )
        request = labelled_request(context, trigger, text, self.channel_id)
        prompt_text = compose_prompt(blocks, catch_up, request)
        prompt = [self._sdk().acp.text_block(prompt_text)]
        # The cursor commits only after the agent accepts the prompt. A
        # generator body that never runs, or a prompt rejected before
        # delivery, leaves the mark untouched so the next turn can replay
        # the missing room context instead of silently losing it.
        turn.runner = asyncio.create_task(
            self._run_prompt(
                connection,
                session_id,
                event_id,
                prompt,
                turn,
                room_id,
                None if standalone else seen_index,
                metadata,
            )
        )

        try:
            while True:
                item = await turn.queue.get()
                if isinstance(item, _TurnDone):
                    # Whatever the turn left open closes here, in the
                    # stream, because the stored TOOL_CALL_END is
                    # persisted from the marker — and the finally below
                    # runs too late to yield one. The closing markers go
                    # through the same queue, so the terminal item is put
                    # back to be read after them; the second pass finds
                    # nothing open and falls through.
                    if await self._close_open_tools(turn, room_id, stream=True):
                        turn.queue.put_nowait(item)
                        continue
                    if item.error is not None:
                        if isinstance(item.error, asyncio.CancelledError):
                            return
                        if (
                            isinstance(item.error, ACPSessionInvalidatedError)
                            and turn.activity_seen
                        ):
                            # Even an authorized signal cannot make observed
                            # activity safe to repeat (including plans/tools).
                            raise ProviderError(
                                "ACP session invalidated after turn activity; recovery refused",
                                provider="acp",
                            ) from item.error
                        if isinstance(item.error, ProviderError):
                            raise item.error
                        raise ProviderError(
                            f"ACP agent prompt failed: {item.error}",
                            provider="acp",
                        ) from item.error
                    turn.completed = True
                    return
                yield item
        finally:
            if turn.runner is not None and not turn.runner.done():
                with contextlib.suppress(Exception):
                    await connection.cancel(session_id)
                turn.runner.cancel()
                await asyncio.gather(turn.runner, return_exceptions=True)
            if turn.thinking_open:
                await self._publish(
                    room_id,
                    EphemeralEventType.THINKING_END,
                    {"thinking": "", "round": 0},
                )
            # A stream closed from the outside — the consumer was
            # cancelled, a muted binding dropped it — never reaches the
            # terminal item above. Its tools still have to stop spinning
            # for live surfaces; the stored row is beyond reach from here,
            # nothing can be yielded into a generator already closing.
            await self._close_open_tools(turn, room_id, stream=False)
            if self._turns.get(session_id) is turn:
                self._turns.pop(session_id, None)
            if turn.completed:
                await self._report_response(turn, metadata)

    async def _report_response(self, turn: _TurnState, metadata: Mapping[str, Any]) -> None:
        """Announce a finished turn to whatever observes agent responses.

        Reached only from a turn that ran to its terminal item without an
        error, and once — an abandoned or failed turn produced no response to
        report. Observational, like the same report on an in-process AI
        channel: a hook that raises does not disturb the turn that is ending.
        """
        if self._after_response_hook is None:
            return
        segments, transcript = response_transcript("".join(chunks) for chunks in turn.segments)
        try:
            await self._after_response_hook(
                AIResponseEvent(
                    channel_id=self.channel_id,
                    response_content=transcript,
                    segments=segments,
                    thinking="".join(turn.thinking),
                    room_id=turn.room_id,
                    tool_calls_count=len(turn.tools),
                    usage=_usage_report(turn.tokens, turn.context),
                    usage_metadata=turn.usage_metadata,
                    latency_ms=int((time.monotonic() - turn.started_at) * 1000),
                    streaming=True,
                    # Its stop reason, ``completed`` for ``end_turn`` (RFC §6.4).
                    loop_end_reason=recorded_turn_end(metadata),
                )
            )
        except Exception:
            logger.debug("After-response hook failed (acp)", exc_info=True)

    async def _run_prompt(
        self,
        connection: Any,
        session_id: str,
        event_id: str,
        prompt: list[Any],
        turn: _TurnState,
        room_id: str,
        seen_index: int | None,
        metadata: ResponseMetadata,
    ) -> None:
        """Run one prompt to its end, recording how that end came about.

        The turn's outcome rides ``metadata`` so that the MESSAGE segments
        persisted for this turn carry it: a caller with nobody watching (a
        scheduled run) has to tell an answer from a turn that stopped early,
        and the text alone cannot say which it is. Only an outcome that is
        *not* a clean end is written as such (``stop_reason``,
        ``interrupted``), the way an interrupted segment is marked
        ``cancelled`` and a finished one is marked by nothing;
        ``prompt_returned`` says the prompt ran to its end, which a turn never
        prompted cannot say (RFC §6.4).
        """
        acp_meta = metadata["acp"]
        try:
            response = await connection.prompt(
                session_id,
                prompt,
                **{"roomkit.live/eventId": event_id},
            )
            # The evidence the turn ran to its end: a turn never prompted (a
            # transport that read nothing) names no end (RFC §6.4).
            acp_meta["prompt_returned"] = True
            # A standalone turn told the room's session nothing: its cursor stays.
            if seen_index is not None:
                self._prompted_index[room_id] = max(
                    seen_index, self._prompted_index.get(room_id, _UNSEEN)
                )
            # The turn's own accounting, and the only place it is offered:
            # the usage notifications describe the context window, not what
            # answering cost.
            turn.tokens = _usage_tokens(getattr(response, "usage", None))
            # ``end_turn`` is the agent saying it finished; every other reason
            # (``refusal``, ``max_tokens``, ``cancelled``) means the work
            # stopped for a cause the caller must be able to act on. A reason
            # the agent did not name stays unwritten rather than being read as
            # either outcome.
            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason:
                turn.usage_metadata["prompt"]["stop_reason"] = stop_reason
            if stop_reason and stop_reason != _CLEAN_STOP_REASON:
                acp_meta["stop_reason"] = stop_reason
            await self._drain_session_updates(session_id)
            envelope = (
                _transport_usage(response) if self._transport.provides_usage_metadata else None
            )
            if envelope is not None:
                _apply_transport_usage(turn.usage_metadata, envelope)
                # The model of a recovered result is not the model we just
                # asked. Only the transport can supply that historical fact.
                turn.usage_metadata["prompt"].pop("model_at_start", None)
                if isinstance(envelope.get("model"), str):
                    turn.usage_metadata["prompt"]["model"] = envelope["model"]
                report = turn.usage_metadata.get("usage_report")
                turn.context = _report_context(report)
        except BaseException as exc:
            # The prompt never returned, so no stop reason exists to record:
            # the turn ended on the way, and that is the fact to carry.
            acp_meta["interrupted"] = True
            if isinstance(exc, ACPSessionInvalidatedError):
                # SDK callbacks may still be queued when prompt() raises. They
                # must count as activity before the stream considers recovery.
                try:
                    await self._drain_session_updates(session_id)
                except BaseException as drain_error:
                    exc = drain_error
            turn.usage_finalized = True
            turn.queue.put_nowait(_TurnDone(error=exc))
        else:
            turn.usage_finalized = True
            turn.queue.put_nowait(_TurnDone())
