"""RoomKit exception hierarchy."""

from __future__ import annotations

from typing import Any

from roomkit.models.delivery import ProviderResult


class RoomKitError(Exception):
    """Base exception for all RoomKit errors."""


class ProviderDeliveryError(RoomKitError):
    """Raised when a provider returns an explicit unsuccessful result.

    Providers use :class:`~roomkit.models.delivery.ProviderResult` for both
    accepted and rejected sends. Turning a negative result into an exception
    at the channel boundary lets the existing retry and circuit-breaker path
    treat it exactly like a transport exception while retaining the structured
    provider response for the caller.
    """

    def __init__(self, result: ProviderResult) -> None:
        message = result.error or "provider_send_failed"
        super().__init__(message)
        self.provider_result = result
        self.code = str(result.metadata.get("code") or result.error or "ProviderDeliveryFailed")
        declared_retryable = result.metadata.get("retryable")
        self.retryable = declared_retryable if isinstance(declared_retryable, bool) else True


class RoomNotFoundError(RoomKitError):
    """Room does not exist."""


class RoomClosedError(RoomKitError):
    """Room's status refuses new events (RFC §5.1).

    Raised by the APIs whose return value has no place for a refusal: direct
    injection, which returns the committed ``RoomEvent`` — returning an event
    marked DELIVERED for a write that never happened would be worse than
    raising — and ``regenerate_target``, which returns the event a regenerate
    would replay. The inbound path and ``regenerate_response``, whose result
    type can say so, return ``InboundResult(blocked=True, reason="room_closed")``
    instead.
    """


class ProcessTimeoutError(RoomKitError):
    """``process_timeout`` expired before the commit point (RFC §13.6): the
    event was not written.

    Raised by direct injection, which returns the committed ``RoomEvent`` and
    has no place for a refusal (RFC §10.5), once the ``process_timeout``
    framework event has been emitted. The inbound path and
    ``regenerate_response`` return ``InboundResult(blocked=True,
    reason="process_timeout")`` instead.
    """


class ChannelNotFoundError(RoomKitError):
    """Channel binding not found in room."""


class ChannelNotRegisteredError(RoomKitError):
    """Channel type not registered."""


class ChannelAlreadyRegisteredError(RoomKitError):
    """A channel with this ID is already registered.

    Silently replacing a live channel would leave existing room bindings
    routing to an object the framework no longer knows about. Call
    ``unregister_channel()`` first to swap an implementation deliberately.
    """


class ParticipantNotFoundError(RoomKitError):
    """Participant not found in room."""


class IdentityNotFoundError(RoomKitError):
    """Identity not found."""


class SourceAlreadyAttachedError(RoomKitError):
    """Source already attached to channel."""


class SourceNotFoundError(RoomKitError):
    """Source not found for channel."""


class VoiceNotConfiguredError(RoomKitError):
    """Raised when voice operation attempted without configured provider."""


class VoiceBackendNotConfiguredError(RoomKitError):
    """Raised when voice backend operation attempted without configured backend."""


class VoiceSessionEndedError(RoomKitError):
    """Raised when a voice session is moved out of ENDED (RFC §12.1).

    ENDED is terminal. A participant who reconnects gets a new session — the
    old one's audio paths, recorders and lanes have already been released.
    """


class RoomNotAttachedError(RoomKitError):
    """Raised when a channel acts on a room it is no longer attached to.

    Detaching leaves the conference running for the humans in it, so backend
    callbacks keep arriving. Acting on them would reconnect a bot nobody asked
    for.
    """


class ParticipantNotAdmittedError(RoomKitError):
    """Raised when a room's participant is barred from what was asked for them.

    Distinct from :class:`ParticipantNotFoundError`, which says the room has
    never heard of them. This one says it has, and the answer is still no —
    ``BANNED`` is "removed and blocked" (RFC section 5.5), and a caller told
    "not found" would reasonably create the participant and try again.
    """


class ConferenceCapabilityError(RoomKitError):
    """Raised when a conference operation needs a capability the backend, or the
    channel's configuration, lacks (a bot asked of a channel with nothing to
    consume or say).

    Refusing at the boundary rather than degrading silently: a moderation UI
    that offers an unmute the SFU will reject, or a recording that never
    materialises, is worse than a configuration that fails immediately.
    """


class ConferenceAlreadyAttachedError(RoomKitError):
    """Raised when a second conference channel is attached to a room.

    A conference maps 1:1 to a Room (RFC section 12.10.1, principle 2), and
    the attachment is where that is enforceable: a second conference channel
    is a second bot session, a second transcription of every utterance and a
    second AI voice speaking the same deliveries — duplicates the roster, the
    transcript and the meeting have no way to express. Re-attaching the
    *same* conference channel is an ordinary attach and is not refused.
    """


class ConferenceCloseError(RoomKitError):
    """Raised when a conference channel did not close all of its resources.

    Raised at the very end of ``ConferenceChannel.close()``, after every step
    has run. It names sessions that could not be taken out, joins or lanes
    retained past their budget, and backend or provider shutdown calls that
    failed. Sessions remain on the channel's books, where ``info()`` reports
    them; resources still used by an abandoned task remain alive until that
    task settles. Raised rather than summarised into a log because a clean
    return would misreport potentially live conference media as released.
    ``RoomKit.close()`` collects it into its ``ExceptionGroup`` instead of
    letting it stop the other channels' closes.

    ``issues`` carries the structured report the message was rendered from —
    one entry per step that failed, timed out, was abandoned, or left a
    resource retained — so operator tooling can match on component and
    status without parsing prose.
    """

    def __init__(self, message: str, *, issues: tuple[Any, ...] = ()) -> None:
        super().__init__(message)
        self.issues = issues


class ToolRefusedError(RoomKitError):
    """Raised by a tool handler for a call it declined to serve.

    The outcome of a tool call is carried, never inferred
    (:attr:`~roomkit.providers.ai.base.AIToolResultPart.is_error`), and a
    handler that *raises* already states it: the tool loop catches the
    exception and marks the part. What it cannot state that way is a refusal
    it wants the model to read in its own words, because the generic branch
    reads ``{"error": "Tool '<name>' failed (<ExceptionClass>)"}``, the
    message withheld from the model (RFC §9.3): the wording a host tuned for a
    small model is gone, and with it the reason.

    This is that branch with the message kept. A handler raises it to say two
    things at once: nothing ran, and here is what the model should read. The
    loop marks the part failed, fires the ON_TOOL_CALL observers, and hands
    :attr:`message` to the model unchanged.

    Returning a refusal as an ordinary string cannot express this: the loop
    would have to recognise a failure in the body, and the bodies do not agree
    — which is the guesswork ``is_error`` exists to end.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ToolFailedError(RoomKitError):
    """Raised by a tool handler for a call that ran and failed, with the words
    the model should read.

    The sibling of :class:`ToolRefusedError` on the other side of the line
    between a refusal and a failure (RFC §9.3): a refusal says nothing ran, a
    failure says the tool ran and could not do it. Any other exception a
    handler raises is a failure too, but its message is withheld from the
    model (it can hold anything the failing code held); this one hands
    :attr:`message` to the model unchanged. The call is marked failed, its
    ON_TOOL_CALL observers read ``refused=False`` and the message as
    ``error_detail``. An MCP tool whose result says ``isError`` raises it.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class HumanInputRejectedError(RoomKitError, RuntimeError):
    """Raised by :meth:`~roomkit.tools.human_input.HumanInputHandler.wait` for a
    request that was rejected, by a human or an ``ON_USER_INPUT_REQUIRED``
    hook, with the reason given.

    A :class:`RuntimeError`, so a caller that catches ``RuntimeError`` around
    ``wait()`` still catches it. The human-input tool reads this one as a
    refusal; a timeout is a failure, and any other error (the handler closing
    or releasing the request before an answer included) takes the generic
    failure path, its message withheld from the model (RFC §9.3).
    """


class ChannelRefusalError(ToolRefusedError):
    """A refusal the channel decided itself, before any tool ran (RFC §9.3).

    A repeat of the same call with the same arguments that the channel stops,
    or a tool outside the turn's toolset. Refused like a handler's refusal,
    but it is not the tool's answer: the room's tool memory does not keep it,
    so it cannot stand in for the result of an earlier, identical call.
    """


class UnservedToolCallError(RoomKitError):
    """Raised by a tool handler for a call that is not its to serve (RFC §21.4).

    The typed way to say "this tool is not mine": a composition of handlers
    (:func:`~roomkit.tools.compose.compose_tool_handlers`) passes the call to
    the next one, and a channel reads the call as served by nothing, on every
    path. A channel's dispatcher raises it too when nothing serves a call.

    Not a refusal: ON_TOOL_CALL's SYNC hooks may still serve the call (RFC
    §9.3), and it fails, reported once, when none does. Raised rather than
    returned so a handler keeps its contract (it answers with a result). A
    call dispatched outside a tool loop may see it.

    A handler that answers ``{"error": "Unknown tool: ..."}`` instead, as
    text or as a mapping, gives the same signal.
    """


class ToolTimeoutError(RoomKitError):
    """A tool handler that did not answer within its call's bound (RFC §21.6).

    The handler was cancelled, and the call fails like one whose handler
    raised (RFC §9.3): the model reads the tool's failure and this class, the
    observers the detail. A ``TimeoutError`` the handler raises itself is that
    handler's own failure, never this.
    """

    def __init__(self, name: str, timeout: float) -> None:
        super().__init__(f"tool {name!r} did not answer within {timeout:g} s")
        self.name = name
        self.timeout = timeout


class ToolNameCollisionError(RoomKitError, ValueError):
    """A tool given under a name a channel already serves where it would be
    declared (RFC §21.1).

    A name is served by one tool in a room: declared once and served by
    another, the model would call one tool's schema on the other's server.
    Raised when the tool is given (a strategy's install, ``setup_handoff``,
    ``setup_delegation``, ``configure(tools=)``), naming the tool.
    """


class TurnCutShortError(RoomKitError):
    """An AI turn that ended before its answer (RFC §6.4): it ended for any
    reason but ``completed`` (its round cap, deadline or budget cut it, a stop
    cancelled it, its answer was cut or never came). An expected outcome, not
    a code defect: logged as a warning, without a traceback (RFC §15.2).

    Attributes:
        reason: The turn's ``loop_end_reason``.
    """

    def __init__(self, message: str, reason: str | None) -> None:
        super().__init__(message)
        self.reason = reason

    def __reduce__(self) -> tuple[Any, ...]:
        # Rebuilt from its own arguments, so a copy or a pickle of a task's
        # result holds it.
        return type(self), (str(self), self.reason), self.__dict__


class TaskCutShortError(TurnCutShortError):
    """A delegated worker's turn ended before its answer (RFC §23.3).

    The turn has no answer, and its last narration is none (RFC §6.4): the
    task fails, its error naming how the turn ended and its output the
    narration, which the caller may still read.

    Attributes:
        reason: The turn's ``loop_end_reason`` (``max_rounds``, ``timeout``,
            ``budget_exceeded``...), or an ACP worker's unclean outcome: its
            stop reason (``max_tokens``, ``max_turn_requests``, ``refusal``,
            ``cancelled``) or ``interrupted`` when its prompt never returned.
        narration: What the worker said last, or ``None``.
    """

    def __init__(self, reason: str, narration: str | None) -> None:
        super().__init__(self.message_for(reason), reason)
        self.narration = narration

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self.reason, self.narration), self.__dict__

    @staticmethod
    def message_for(reason: str) -> str:
        """The error a task cut short by *reason* carries."""
        return f"The worker's turn ended {reason} before its answer"


class TaskTurnFailedError(RoomKitError):
    """A delegated worker's turn that failed after it began (RFC §23.3).

    The task fails with the turn's error, whose message this one keeps and
    which is its cause, and carries what the child room's record holds of the
    turn: how it ended and what the worker said last.

    Attributes:
        reason: The turn's ``loop_end_reason`` (``error``), or an ACP worker's
            ``interrupted`` when its prompt failed.
        narration: What the worker said last, or ``None``.
    """

    def __init__(self, error: BaseException, reason: str, narration: str | None) -> None:
        super().__init__(str(error))
        self.reason = reason
        self.narration = narration
        self.__cause__ = error

    def __reduce__(self) -> tuple[Any, ...]:
        cause = self.__cause__ if self.__cause__ is not None else RoomKitError(str(self))
        return type(self), (cause, self.reason, self.narration), self.__dict__
