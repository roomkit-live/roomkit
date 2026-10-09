"""The one log line a failed turn, delivery or task gets (RFC §15.2).

A ``ProviderError`` is not a code defect: it is logged without a traceback, at
the level its status calls for, with the provider and the status in the line
(RFC §15.3). Anything else is unexpected and keeps its full traceback. The
component that raises leaves the line to the one that catches, so an incident
is logged once.
"""

from __future__ import annotations

import logging
from typing import Any

from roomkit.core.exceptions import NoRecipientError, TaskTurnFailedError, TurnCutShortError
from roomkit.providers.ai.base import ProviderError

_REPORTED = "_roomkit_reported"


def mark_reported[E: BaseException](exc: E) -> E:
    """Mark *exc* as a failure its own turn already reported to ON_ERROR:
    whoever receives it next (a strategy's caller, a pass's room turn) hands
    it on, its type unchanged, without a second report (RFC §19.7.3,
    §19.7.4). Whoever catches it on its way still logs it once."""
    setattr(exc, _REPORTED, True)
    return exc


def was_reported(exc: BaseException) -> bool:
    """Whether *exc* is a failure its own turn already reported to ON_ERROR
    (:func:`mark_reported`)."""
    return bool(getattr(exc, _REPORTED, False))


def needs_reporting(exc: BaseException | None) -> bool:
    """Whether a turn's failure *exc* is still to be reported to ON_ERROR and
    logged where it surfaces: not a turn that ended before its answer (an
    expected end, :class:`TurnCutShortError`), nor one its own turn already
    reported (:func:`mark_reported`)."""
    if isinstance(exc, TurnCutShortError):
        return False
    return exc is None or not was_reported(exc)


def provider_error_level(exc: ProviderError) -> int:
    """ERROR for a missing model or a server fault, WARNING for a transient.

    A 404 (the model does not exist) or a 5xx needs someone to act; no status
    (connect refused, timeout), a 429 or another 4xx is expected now and then.
    """
    status = exc.status_code
    if status == 404 or (status is not None and status >= 500):
        return logging.ERROR
    return logging.WARNING


def log_failure(
    log: logging.Logger,
    exc: Exception,
    what: str,
    *,
    caller_logs: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log that *what* failed with *exc*, once, at the level its cause calls for.

    ``caller_logs`` is for a caller that receives the error itself and owns its
    log line (``InboundResult.error``): a provider error is then only a DEBUG
    line here, so the incident is not reported twice.
    """
    if isinstance(exc, TaskTurnFailedError) and isinstance(exc.__cause__, Exception):
        # A worker's turn that failed: logged as its error is.
        log_failure(log, exc.__cause__, what, caller_logs=caller_logs, extra=extra)
        return
    if isinstance(exc, TurnCutShortError):
        # A turn that ended before its answer: expected, never a defect.
        log.warning("%s failed: %s", what, exc, extra=extra)
        return
    if isinstance(exc, NoRecipientError):
        # Refused before any send: the binding's configuration, not a defect.
        log.warning("%s refused: the binding has no %s", what, exc.recipient_key, extra=extra)
        return
    if not isinstance(exc, ProviderError):
        # The traceback is *exc*'s, whether or not a handler is still active.
        log.error("%s failed", what, exc_info=exc, extra=extra)
        return
    level = logging.DEBUG if caller_logs else provider_error_level(exc)
    log.log(
        level,
        "%s failed (provider=%s, status=%s): %s",
        what,
        exc.provider,
        exc.status_code,
        exc,
        extra=extra,
    )
