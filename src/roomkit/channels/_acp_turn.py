"""One ACP turn: a prompt sent to a session, streamed until the agent ends it.

The agent's updates arrive on the connection (``ACPEventsMixin``) and are put
on the turn's queue; this mixin is the other end of that queue. It composes the
prompt, runs it, yields what the agent says as stream deltas, records how the
turn ended on the response metadata, and closes whatever the turn left open.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Mapping,
    Sequence,
)
from contextlib import AbstractAsyncContextManager
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from roomkit.channels._acp_client import _SDK, _TurnDone, _TurnState
from roomkit.channels._acp_context import (
    compose_prompt,
    instructions_block,
    labelled_request,
    room_context_block,
    window_speakers,
)
from roomkit.channels._acp_usage import (
    _apply_transport_usage,
    _report_context,
    _transport_usage,
    _usage_report,
    _usage_tokens,
)
from roomkit.channels._speaker import labels_of, people_taking_part
from roomkit.channels._turn_notes import without_header_copies
from roomkit.channels.acp_transport import ACPSessionInvalidatedError
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.models.response_metadata import ResponseMetadata, recorded_turn_end
from roomkit.models.streaming import StreamDelta
from roomkit.models.tool_call import AfterResponseCallback, AIResponseEvent, response_transcript
from roomkit.providers.ai.base import ProviderError
from roomkit.realtime.base import EphemeralEventType
from roomkit.tools.context import _with_response_record

if TYPE_CHECKING:
    from roomkit.channels.acp_transport import ACPTransport

logger = logging.getLogger("roomkit.channels.acp")


def _opening_key(room_id: str) -> str:
    """The turn registry key of a turn of *room_id* whose session is being opened.

    Turns are filed by session id, and this one has none yet.
    """
    return f"roomkit.live/opening/{room_id}"


_CLEAN_STOP_REASON = "end_turn"
"""The one ACP stop reason that means the agent finished what it was asked."""

_UNSEEN = -1
"""No prompt has left for this room yet — event indices start at 0."""


async def _within_record[T](record: ResponseMetadata, work: Awaitable[T]) -> T:
    """Run a closing callback with its turn's record in this consumer's task."""
    with _with_response_record(record):
        return await work


class ACPTurnMixin:
    """Run one prompt per turn and stream its outcome."""

    channel_id: str
    _transport: ACPTransport
    _connection: Any
    _connection_generation: int
    _sessions: dict[str, str]
    _agent_info: dict[str, Any] | None
    _turns: dict[str, _TurnState]
    _prompted_index: dict[str, int]
    _labelled_rooms: set[str]
    _instructed_rooms: set[str]
    _instructions: str | None
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
    _session_for: Callable[..., Coroutine[Any, Any, str]]
    _discard_room_session: Callable[[str, Any], Coroutine[Any, Any, bool]]
    _open_turn_session: Callable[[str, Any], Coroutine[Any, Any, str]]
    _close_turn_session: Callable[[str, str, Any], Coroutine[Any, Any, None]]
    session_config: Callable[[str], dict[str, str | bool]]
    _close_open_tools: Callable[..., Coroutine[Any, Any, bool]]
    _publish: Callable[..., Coroutine[Any, Any, None]]
    _room_locks: dict[str, asyncio.Lock]
    _detached: dict[str, int]

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
        notes: str | None = None,
        standalone: bool = False,
        generation: int = 0,
    ) -> AsyncIterator[StreamDelta]:
        """The turn's prompt, under the room's turn lock; *generation* is the
        number of the room's detaches when the turn started, past which it
        never reaches the agent (RFC §10.6.3)."""
        async with self._room_turn_lock(room_id):
            if self._detached_since(room_id, generation, metadata):
                return
            opened = await self._start_turn(room_id, standalone, metadata)
            if opened is None:
                return
            connection, session_id = opened
            recovering = False
            completed = False
            try:
                if self._detached_since(room_id, generation, metadata):
                    return
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
                        notes=notes,
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
                elif self._detached.get(room_id, 0) != generation:
                    # Detached while it ran: the detach left the session to it.
                    await self._discard_room_session(room_id, connection)

    def _detached_since(self, room_id: str, generation: int, metadata: ResponseMetadata) -> bool:
        """Whether the channel was detached from *room_id* since the turn
        started; such a turn is marked interrupted and never prompts."""
        if self._detached.get(room_id, 0) == generation:
            return False
        metadata["acp"]["interrupted"] = True
        return True

    async def _start_turn(
        self, room_id: str, standalone: bool, metadata: ResponseMetadata
    ) -> tuple[Any, str] | None:
        """The connection and the session a turn prompts, opened where a cancel reaches them.

        None when the room's turn was cancelled first, or the channel closed:
        the turn is marked interrupted and never prompts.
        """
        discard = None
        if standalone:
            discard = functools.partial(self._close_opened, room_id)
        opened = await self._prepared(
            room_id, _opening_key(room_id), self._open_for_turn(room_id, standalone), discard
        )
        if opened is None:
            metadata["acp"]["interrupted"] = True
        return opened

    async def _open_for_turn(self, room_id: str, standalone: bool) -> tuple[Any, str]:
        """Connect, then open the session a turn prompts: the room's, or its own if standalone."""
        connection = await self._ensure_connection()
        if standalone:
            return connection, await self._open_turn_session(room_id, connection)
        return connection, await self._session_for(room_id, connection)

    async def _close_opened(self, room_id: str, opened: tuple[Any, str]) -> None:
        """Close the standalone session a cancelled turn had opened."""
        connection, session_id = opened
        await self._close_turn_session(room_id, session_id, connection)

    async def _rebuild_session(self, room_id: str, session_id: str, connection: Any) -> str | None:
        """Rebuild the room's session where public cancel can stop it."""
        return await self._prepared(
            room_id, session_id, self._replace_session(room_id, connection)
        )

    async def _prepared[T](
        self,
        room_id: str,
        key: str,
        preparation: Coroutine[Any, Any, T],
        discard: Callable[[T], Awaitable[None]] | None = None,
    ) -> T | None:
        """Run *preparation*, a turn's session being opened or rebuilt, as a preparing turn.

        Registered under *key* in the turn registry, so :meth:`ACPChannel.cancel`
        stops the runner: no prompt is out yet for the agent to cancel.
        Returns None once the turn is cancelled or the channel closed; a
        result that came too late is handed to *discard*.
        """
        turn = _TurnState(room_id=room_id, preparing=True)
        turn.runner = asyncio.create_task(preparation)
        self._turns[key] = turn
        try:
            result = await turn.runner
            # A cancel can land after the runner finishes but before we wake.
            if not (self._closed or turn.cancel_requested):
                return result
            if discard is not None:
                await discard(result)
            return None
        except asyncio.CancelledError:
            if not turn.cancel_requested:
                raise
            return None
        finally:
            if self._turns.get(key) is turn:
                self._turns.pop(key, None)

    async def _replace_session(self, room_id: str, connection: Any) -> str:
        await self._discard_room_session(room_id, connection)
        if self._closed:
            raise asyncio.CancelledError
        session_id = await self._session_for(room_id, connection)
        if self._closed:
            raise asyncio.CancelledError
        return session_id

    def _prompt_text(
        self,
        room_id: str,
        blocks: Sequence[str],
        context: RoomContext,
        trigger: RoomEvent,
        text: str,
        *,
        notes: str | None,
        standalone: bool,
    ) -> tuple[str, bool, bool]:
        """The prompt for *trigger*, whether its request opens with its
        sender's label, and whether it carries the channel's instructions:
        the instructions on a session's first prompt, the host's blocks, the
        room context it missed, the request (RFC §6.4, §19.3.2), one ranking
        naming each source one way in both, then a discussion's *notes*. A
        standalone turn reads nothing of the room: no catch-up, and its
        session was born empty. A copy of the notes' header anywhere but the
        notes is replaced, so no one passes their words off as the
        runtime's (RFC §6.4)."""
        speakers = window_speakers(context, trigger, self.channel_id)
        catch_up = (
            ""
            if standalone
            else room_context_block(
                context,
                self.channel_id,
                after_index=self._prompted_index.get(room_id, _UNSEEN),
                trigger=trigger,
                limit=self._room_history,
                labels=labels_of(speakers),
            )
        )
        request, labelled = labelled_request(
            speakers,
            trigger,
            text,
            people=people_taking_part(context, self.channel_id),
            session_labelled=room_id in self._labelled_rooms,
        )
        instructions = self._instructions
        instructed = instructions is not None and (
            standalone or room_id not in self._instructed_rooms
        )
        head = [instructions_block(instructions)] if instructions and instructed else []
        prompt = compose_prompt(
            [*head, *map(without_header_copies, blocks)],
            without_header_copies(catch_up),
            without_header_copies(request),
            notes,
        )
        return prompt, labelled, instructed

    def _turn_state(
        self, room_id: str, session_id: str, event_id: str, metadata: ResponseMetadata
    ) -> _TurnState:
        """Capture the prompt's context and record for the session callbacks."""
        prompt_source: dict[str, Any] = {"source": "session/prompt", "scope": "unspecified"}
        model = self.session_config(room_id).get("model")
        if isinstance(model, str):
            prompt_source["model_at_start"] = model
        with _with_response_record(metadata):
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
        return turn

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
        notes: str | None = None,
        standalone: bool,
    ) -> AsyncGenerator[StreamDelta]:
        """One prompt in *session_id*, streamed until the agent ends it."""
        turn = self._turn_state(room_id, session_id, event_id, metadata)
        self._turns[session_id] = turn
        prompt_text, turn.labelled, turn.instructed = self._prompt_text(
            room_id, blocks, context, trigger, text, notes=notes, standalone=standalone
        )
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
                self._connection_generation,
            ),
            context=turn.scope.copy(),
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
                    if await _within_record(
                        metadata, self._close_open_tools(turn, room_id, stream=True)
                    ):
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
            await _within_record(metadata, self._close_open_tools(turn, room_id, stream=False))
            if self._turns.get(session_id) is turn:
                self._turns.pop(session_id, None)
            if turn.completed:
                await _within_record(metadata, self._report_response(turn, metadata))

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

    def _acknowledge_prompt(
        self,
        connection: Any,
        generation: int,
        session_id: str,
        turn: _TurnState,
        seen_index: int | None,
    ) -> None:
        """A reply acknowledges only the connection and room session it read.

        A custom transport may reuse its connection object or session IDs.
        The generation captured before prompting distinguishes its reopenings.
        Standalone prompts never acknowledge the room's session.
        """
        if (
            seen_index is None
            or self._closed
            or self._connection is not connection
            or self._connection_generation != generation
            or self._sessions.get(turn.room_id) != session_id
        ):
            return
        room_id = turn.room_id
        self._prompted_index[room_id] = max(seen_index, self._prompted_index.get(room_id, _UNSEEN))
        if turn.labelled:
            self._labelled_rooms.add(room_id)
        if turn.instructed:
            self._instructed_rooms.add(room_id)

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
        connection_generation: int,
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
            self._acknowledge_prompt(
                connection, connection_generation, session_id, turn, seen_index
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
