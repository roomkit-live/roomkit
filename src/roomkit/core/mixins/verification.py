"""VerificationMixin — identity verification requests and their outcome (RFC §11.7).

RoomKit keeps the state and the lifecycle; the integrator keeps the method (how
a person proves who they are) and calls :meth:`complete_verification` once its
own check passed. A request's secret is returned once and stored only as a
salted hash.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import hmac
import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.core._authors import People
from roomkit.core._changes import withdrawn
from roomkit.core.exceptions import VerificationRefusedError
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.core.visibility import committed_visibility, visibility_allows
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import (
    ChannelCategory,
    EventType,
    HookTrigger,
    IdentificationStatus,
    VerificationRequestStatus,
    VerificationStatus,
    Visibility,
)
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import VERIFICATION_DECISION_KEY
from roomkit.models.verification import Verification, VerificationPolicy, VerificationRequest

if TYPE_CHECKING:
    from roomkit.core._delivery_keys import NamedDelivery
    from roomkit.core._reentrance import RoomWork
    from roomkit.core.hooks import HookEngine
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.context import RoomContext
    from roomkit.models.identity import Identity
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")

METHOD_LINK = "link"
"""A secret carried in a link to the integrator's page (high entropy)."""
METHOD_CODE = "code"
"""A short code the person replies with in the conversation (RFC §11.7.4)."""

ENDED_VERIFICATION = "verification"
"""What an end names (``ended``): the identity's verification in force ended,
by expiry, revocation, or a request that failed while it held."""
ENDED_REQUEST = "request"
"""What an end names (``ended``): a request ended (expired, failed, cancelled),
the identity's verification left as it was or never reached."""

_REQUEST_TIMER = "request:"
"""Key prefix of a request's expiry timer, beside the identities' timers."""
_LINK_SECRET_BYTES = 32  # 256 bits; RFC §11.7.7 asks for at least 128
HELD_KEY = "roomkit:verification_held"
"""Event metadata marking a message held for verification (RFC §11.7.3); its
value records the visibility the message had, which its release restores."""
# scrypt, not a fast hash: a 6-digit code read from a leaked store would
# otherwise be found in under a second. At n=2**15 one hash costs ~55 ms, so
# the million codes take hours, against a code's minutes of life.
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1


@runtime_checkable
class VerificationHost(Protocol):
    """Contract: capabilities a host class must provide for VerificationMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation persistence backend (stores verifications).
        _lock_manager: Serialises the transitions of one identity.
        _verification_policy: Lifetimes and attempts.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _verification_policy: VerificationPolicy
    _verification_timers: dict[str, asyncio.Task[None]]


class VerificationMixin(HelpersMixin):
    """Identity verification: requests, completion, revocation, state.

    Host contract: :class:`VerificationHost`.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _verification_policy: VerificationPolicy
    _verification_timers: dict[str, asyncio.Task[None]]
    _hook_engine: HookEngine
    _room_work: RoomWork

    # Cross-mixin methods — attribute annotation avoids MRO shadowing
    _rerun_agents_on: Any  # see RegenerateMixin
    _delivery_keys: Any  # see InboundMixin
    get_room: Any  # see RoomLifecycleMixin

    async def request_verification(
        self,
        identity_id: str,
        *,
        room_id: str,
        channel_id: str,
        method: str = METHOD_LINK,
        window_start: datetime | None = None,
        organization_id: str | None = None,
    ) -> tuple[VerificationRequest, str]:
        """Open a verification request for an identity; return it and its secret.

        The secret is returned once and never stored: put it in the link to
        your page, or send it as a code (``method="code"``: digits the person
        replies with). Opening a request cancels the identity's earlier open
        one. An identity that is not verified becomes PENDING; one already
        verified stays so until this request completes or its verification
        ends. *window_start* is when the message that called for it was
        received (now by default): the held messages released on completion
        are those from then on (RFC §11.7.6).

        The request is scoped to the room's organization, the one whose
        messages it verifies. *organization_id* is the organization the
        caller acts for (RFC §17.2): a room of another raises
        :class:`RoomNotFoundError`.
        """
        room = await self.get_room(room_id, organization_id=organization_id)
        organization_id = room.organization_id
        policy = self._verification_policy
        now = datetime.now(UTC)
        code = method == METHOD_CODE
        secret = _code(policy.code_length) if code else secrets.token_urlsafe(_LINK_SECRET_BYTES)
        ttl = policy.code_request_ttl if code else policy.request_ttl
        request_id = uuid4().hex
        request = VerificationRequest(
            id=request_id,
            identity_id=identity_id,
            organization_id=organization_id,
            room_id=room_id,
            channel_id=channel_id,
            method=method,
            secret_hash=await _secret_hash(request_id, secret),
            window_start=window_start or now,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl),
            max_attempts=policy.max_attempts,
        )
        async with self._lock_manager.locked(_lock_key(identity_id, organization_id)):
            current = await self._stored_verification(identity_id, organization_id)
            cancelled = await self._cancel_open_request(current)
            await self._store.create_verification_request(request)
            in_force = _in_force(current, now)
            await self._store.put_verification(
                current.model_copy(
                    update={
                        "status": VerificationStatus.VERIFIED
                        if in_force
                        else VerificationStatus.PENDING,
                        "request_id": request.id,
                    }
                )
            )
        # Announced once the lock is released: a handler may call back in.
        if cancelled is not None:
            await self._announce_superseded(cancelled)
        await self._emit_verification_event("verification_requested", request)
        self._arm_request_expiry(request)
        return request, secret

    async def complete_verification(
        self,
        request_id: str,
        secret: str,
        *,
        method: str | None = None,
        organization_id: str | None = None,
    ) -> Verification:
        """Complete a request once your own check passed; return the verification.

        Raises :class:`VerificationRefusedError` when the request is unknown
        (or another organization's than *organization_id*, when given), no
        longer open, expired, or the secret does not match (a mismatch counts
        as an attempt; the last one fails it).
        *method* records how the person proved it, the request's by default.
        """
        request = await self._store.get_verification_request(request_id)
        if request is None or not _in_scope(request, organization_id):
            raise VerificationRefusedError("not_found")
        key = _lock_key(request.identity_id, request.organization_id)
        async with self._lock_manager.locked(key):
            outcome, request, verification = await self._attempt(request_id, secret, method)
        if outcome == "completed":
            self._disarm_request_expiry(request.id)
            self._arm_expiry(verification)
            await self._emit_verification_event("verification_completed", request)
            await self._verification_hook(
                HookTrigger.ON_VERIFICATION_COMPLETED,
                request,
                {"method": verification.method, "expires_at": _iso(verification.expires_at)},
            )
            if self._verification_policy.release_on_verify:
                await self._release_held(request)
            return verification
        await self._after_unfulfilled(outcome, request, verification)
        raise VerificationRefusedError(_refusal(outcome, request))

    async def record_failed_attempt(
        self, request_id: str, *, organization_id: str | None = None
    ) -> VerificationRequest:
        """Count a failed check of your own method against a request; return it.

        Your page's wrong PIN or refused login is an attempt like a wrong
        secret: the request's bound applies to it, so a PIN cannot be guessed
        through your page. The request comes back OPEN with one attempt more,
        or FAILED when that was the last one (which ends it as a wrong secret
        does, a verification in force included). Raises
        :class:`VerificationRefusedError` when the request is unknown (or
        another organization's than *organization_id*, when given), no longer
        open, or expired.
        """
        request = await self._store.get_verification_request(request_id)
        if request is None or not _in_scope(request, organization_id):
            raise VerificationRefusedError("not_found")
        key = _lock_key(request.identity_id, request.organization_id)
        async with self._lock_manager.locked(key):
            outcome, request, current = await self._open_request(request_id)
            if outcome is None:
                outcome, request = await self._count_failure(request, current)
        await self._after_unfulfilled(outcome, request, current)
        if outcome in ("not_open", "expired"):
            raise VerificationRefusedError(_refusal(outcome, request))
        return request

    async def revoke_verification(
        self,
        identity_id: str,
        *,
        reason: str = "revoked",
        organization_id: str | None = None,
    ) -> Verification | None:
        """End an identity's verification before its time; None if none was in force.

        Its open request is cancelled in every case, verified or still pending:
        a link sent before the revocation can never verify the person after it
        (RFC §11.7.7).
        """
        now = datetime.now(UTC)
        self._disarm_expiry(identity_id, organization_id)
        async with self._lock_manager.locked(_lock_key(identity_id, organization_id)):
            current = await self._stored_verification(identity_id, organization_id)
            completing = await self._completing_request(current)
            cancelled = await self._cancel_open_request(current)
            ended: Verification | None = None
            if _in_force(current, now):
                ended = current.model_copy(
                    update={
                        "status": VerificationStatus.REVOKED,
                        "ended_reason": reason,
                        "request_id": None,
                    }
                )
                await self._store.put_verification(ended)
            elif cancelled is not None:
                # Pending, now with nothing open: back to unverified.
                await self._store.put_verification(
                    current.model_copy(
                        update={"status": VerificationStatus.UNVERIFIED, "request_id": None}
                    )
                )
        if cancelled is not None:
            await self._announce_superseded(cancelled)
        if ended is not None:
            await self._emit_framework_event(
                "verification_ended",
                data={"identity_id": identity_id, "reason": reason, "ended": ENDED_VERIFICATION},
            )
            if completing is not None:
                await self._verification_hook(
                    HookTrigger.ON_VERIFICATION_ENDED,
                    completing,
                    {"reason": reason, "ended": ENDED_VERIFICATION},
                )
        return ended

    async def check_verifications(self) -> list[Verification]:
        """End the verifications whose time passed; arm the timers of the others.

        A verification ends when its time runs out (RFC §11.7.5), by a timer
        this process arms when it completes. Call this when the process
        starts (``async with RoomKit(...)`` does): it ends what expired while
        nothing was running and re-arms the rest. Returns the ones it ended.
        """
        ended: list[Verification] = []
        now = datetime.now(UTC)
        for verification in await self._store.list_verifications_in_force():
            if _in_force(verification, now):
                self._arm_expiry(verification)
                continue
            expired = await self._expire(verification)
            if expired is not None:
                ended.append(expired)
        return ended

    async def verification_of(
        self, identity_id: str, *, organization_id: str | None = None
    ) -> Verification:
        """The identity's verification as it stands now: UNVERIFIED when it has none.

        A verification whose time has passed reads EXPIRED, and an open
        request past its expiry reads as ended (no ``request_id``, a PENDING
        identity back to UNVERIFIED), whether or not the end has been
        recorded yet.
        """
        current = await self._without_lapsed_request(
            await self._stored_verification(identity_id, organization_id)
        )
        if current.status == VerificationStatus.VERIFIED and not _in_force(
            current, datetime.now(UTC)
        ):
            return current.model_copy(
                update={"status": VerificationStatus.EXPIRED, "ended_reason": "expired"}
            )
        return current

    async def sender_verification(
        self, event: RoomEvent, context: RoomContext
    ) -> Verification | None:
        """The verification of *event*'s sender, in its room's organization.

        None when the pipeline did not label the event (its sender is not
        identified, or the kit has no verification policy). What an
        ON_VERIFICATION_REQUIRED hook reads before opening a request: whose
        verification it is (``identity_id``) and whether a request is open
        already (``request_id``). The sender's participant may not be in
        *context* yet on their first message; this reads it all the same.
        """
        if event.source.verification is None:
            return None
        identity_id = _participant_identity(event, context) or event.source.participant_id
        if identity_id is None:
            return None
        return await self.verification_of(
            identity_id, organization_id=context.room.organization_id
        )

    # -- inbound pipeline (RFC §10.1 step 5b) -----------------------------------

    async def _apply_verification(
        self,
        event: RoomEvent,
        room_id: str,
        context: RoomContext,
        resolved_identity: Identity | None,
    ) -> RoomEvent | InboundResult:
        """Label an identified sender's message, consume a code, or hold it.

        Returns the event to go on with, or the result of a message that stops
        here before anything of it is stored (a code reply, a block).
        """
        identity = _sender_identity(event, context, resolved_identity)
        if identity is None:
            return event
        identity_id, organization_id = identity
        state = await self.verification_of(identity_id, organization_id=organization_id)
        if await self._consume_code(event, room_id, state):
            # Its text is never stored nor broadcast, and no agent reads it.
            return InboundResult(blocked=True, reason="verification_code")
        event = event.model_copy(
            update={"source": event.source.model_copy(update={"verification": state.status})}
        )
        if state.status == VerificationStatus.VERIFIED:
            return event
        decided = await self._hook_engine.run_sync_hooks(
            room_id, HookTrigger.ON_VERIFICATION_REQUIRED, event, context
        )
        if not decided.allowed:
            return InboundResult(blocked=True, reason=decided.reason or "verification_required")
        event = decided.event if isinstance(decided.event, RoomEvent) else event
        passed = decided.metadata.get(VERIFICATION_DECISION_KEY) == "pass"
        if passed or not self._verification_policy.hold_unverified:
            return event
        if event.type in (EventType.EDIT, EventType.DELETE):
            # An edit or a deletion changes a message the agents may already
            # read; held, it would still rewrite it. Refused instead.
            return InboundResult(blocked=True, reason="verification_required")
        return _held(event, context, identity_id, organization_id)

    async def _consume_code(self, event: RoomEvent, room_id: str, state: Verification) -> bool:
        """Whether *event* is a reply to the sender's open code request here, attempted."""
        if state.request_id is None or not isinstance(event.content, TextContent):
            return False
        text = event.content.body.strip()
        if not text.isdigit() or len(text) != self._verification_policy.code_length:
            return False
        request = await self._store.get_verification_request(state.request_id)
        if (
            request is None
            or request.method != METHOD_CODE
            or request.status != VerificationRequestStatus.OPEN
            or request.room_id != room_id
        ):
            return False
        # A wrong code counts as an attempt and is consumed all the same.
        with contextlib.suppress(VerificationRefusedError):
            await self.complete_verification(
                request.id, text, organization_id=request.organization_id
            )
        return True

    # -- releasing held messages (RFC §11.7.6) ----------------------------------

    async def _release_held(self, request: VerificationRequest) -> list[RoomEvent]:
        """Release what the request's sender wrote while it was open; the released.

        Each comes back with the visibility it had before it was held, through
        the store (ON_EVENT_UPDATED fires), and the room's agents then run on
        the last one as on any inbound message of its author: the earlier ones
        are its context. Nothing is copied, and nothing becomes an instruction.
        A last one whose delivery was cancelled gets no answer (RFC §10.1).
        """
        held = await self._held_window(request)
        if not held:
            return []
        released, cancelled = await self._restore_held(request.room_id, held)
        if not released:
            return []
        context = await self._build_context(request.room_id)
        # The triggers of a change the framework makes for the room's work run
        # on that work's budget, as a direct change's do (RFC §9.5.1, §10.3).
        budget = self._room_work.direct_change(request.room_id).budget
        for event in released:
            self._hook_engine._start_async_hooks(
                request.room_id, HookTrigger.ON_EVENT_UPDATED, event, context, budget=budget
            )
        await self._answer_released(request.room_id, released, cancelled)
        return released

    async def _restore_held(
        self, room_id: str, held: list[RoomEvent]
    ) -> tuple[list[RoomEvent], set[str]]:
        """Restore *held* to the visibility each had before it was held: the
        restored, and the ids of those whose delivery was cancelled."""
        released: list[RoomEvent] = []
        cancelled: set[str] = set()
        async with self._lock_manager.locked(room_id):
            for event in held:
                current = await self._store.get_event(event.id)
                marker = current.metadata.get(HELD_KEY) if current is not None else None
                if current is None or not isinstance(marker, dict):
                    continue  # released already, or changed under us
                if marker.get("cancelled"):
                    cancelled.add(current.id)
                metadata = {k: v for k, v in current.metadata.items() if k != HELD_KEY}
                restored = current.model_copy(
                    update={
                        "visibility": marker.get("visibility") or Visibility.ALL,
                        "metadata": metadata,
                    }
                )
                released.append(await self._store.update_event(restored))
        return released, cancelled

    async def _answer_released(
        self, room_id: str, released: list[RoomEvent], cancelled: set[str]
    ) -> None:
        """Run the room's agents on the last of *released*, named by its
        delivery's key, unless that delivery was cancelled (RFC §10.1): no
        agent answers it then. The earlier ones are its context."""
        last = self._unhold_released(released)
        answer = released[-1]
        if answer.id in cancelled or (last is not None and last.cancelled is not None):
            if last is not None:
                last.release()
            return
        await self._rerun_agents_on(room_id, answer.id, named=last)

    def _unhold_released(self, released: list[RoomEvent]) -> NamedDelivery | None:
        """Stop holding the deliveries of *released*: the last one's, which
        names its answer, or ``None`` when this process holds none."""
        named = [self._delivery_keys.unhold(event.id) for event in released]
        for earlier in named[:-1]:
            if earlier is not None:
                earlier.release()
        return named[-1]

    def _name_held(self, named: NamedDelivery, result: InboundResult) -> None:
        """Keep *named* naming a delivery whose committed event is held, until
        its release or the policy's last moment for it (RFC §11.7.6). Called
        under the room's lock, so no release reaches the event before."""
        event = result.event
        if (
            result.blocked
            or event is None
            or not self._verification_policy.release_on_verify
            or not isinstance(event.metadata.get(HELD_KEY), dict)
        ):
            return  # nothing will release it
        lifetime = self._verification_policy.release_max_age
        self._delivery_keys.hold(named, event.room_id, event.id, lifetime)

    def _record_held_cancellation(self, held: tuple[str, str], reason: str) -> None:
        """Record on a held event that its delivery was cancelled, so that its
        release answers it with no agent in any process. Off the canceller,
        in a context of its own: a cancellation returns at once, under the
        room's lock too, and the write takes that lock in its turn. A write
        still pending when the kit closes is dropped with its other pending
        work; the process itself no longer releases anything then."""
        task = asyncio.get_running_loop().create_task(
            self._write_held_cancellation(*held, reason), context=contextvars.Context()
        )
        task.add_done_callback(self._pending_hook_tasks.discard)
        self._pending_hook_tasks.add(task)

    async def _write_held_cancellation(self, room_id: str, event_id: str, reason: str) -> None:
        """Write *reason* on *event_id*'s hold marker, if it is still held."""
        try:
            async with self._lock_manager.locked(room_id):
                current = await self._store.get_event(event_id)
                marker = current.metadata.get(HELD_KEY) if current is not None else None
                if current is None or not isinstance(marker, dict) or marker.get("cancelled"):
                    return  # released already, or recorded
                metadata = {**current.metadata, HELD_KEY: {**marker, "cancelled": reason}}
                await self._store.update_event(current.model_copy(update={"metadata": metadata}))
        except Exception:
            logger.exception("Recording the cancellation of held event %s failed", event_id)

    async def _held_window(self, request: VerificationRequest) -> list[RoomEvent]:
        """The held messages a completed request releases, oldest first.

        Its sender's own (the identity the marker names), in its room, received
        from the request's window start on, within the policy's bounds of age
        and number; never one released before (its marker is gone).
        """
        policy = self._verification_policy
        oldest = max(
            request.window_start,
            datetime.now(UTC) - timedelta(seconds=policy.release_max_age),
        )
        recent = await self._store.list_events(
            request.room_id, limit=max(200, policy.release_max_messages * 5), newest_first=True
        )
        # A held message deleted since is no longer said (RFC §10.3).
        held = [
            event
            for event in recent
            if _held_for(event, request) and event.created_at >= oldest and not withdrawn(event)
        ]
        # A newest-first page is still rendered in ascending order (RFC §14.1):
        # the latest ones are taken by index, then given back oldest first.
        latest = sorted(held, key=lambda event: event.index, reverse=True)
        return sorted(latest[: policy.release_max_messages], key=lambda event: event.index)

    # -- internal ------------------------------------------------------------

    async def _resume_verifications(self) -> None:
        """Resume the verifications in force as the kit starts.

        A store that does not keep verifications (RFC §11.7 is Level 2) has
        none to resume: the kit starts all the same. So does one that cannot
        answer yet (a ``PostgresStore`` before ``init()``, a database down):
        a verification whose time passed reads EXPIRED all the same, only its
        end is announced late, at the next ``check_verifications()``.
        """
        try:
            await self.check_verifications()
        except NotImplementedError:
            return
        except Exception as exc:
            logger.warning("Verifications were not resumed: the store did not answer (%s)", exc)

    def _arm_expiry(self, verification: Verification) -> None:
        """Arm (or re-arm) the timer that ends *verification* when it expires."""
        if verification.expires_at is None:
            return
        key = _lock_key(verification.identity_id, verification.organization_id)
        self._disarm_expiry(verification.identity_id, verification.organization_id)
        task = asyncio.get_running_loop().create_task(self._expire_when_due(verification))
        self._verification_timers[key] = task
        task.add_done_callback(lambda done: _forget_timer(self._verification_timers, key, done))

    def _disarm_expiry(self, identity_id: str, organization_id: str | None) -> None:
        task = self._verification_timers.pop(_lock_key(identity_id, organization_id), None)
        if task is not None:
            task.cancel()

    def _arm_request_expiry(self, request: VerificationRequest) -> None:
        """Arm the timer that ends *request* if it is still open at its expiry."""
        key = _REQUEST_TIMER + request.id
        task = asyncio.get_running_loop().create_task(self._end_request_when_due(request))
        self._verification_timers[key] = task
        task.add_done_callback(lambda done: _forget_timer(self._verification_timers, key, done))

    def _disarm_request_expiry(self, request_id: str) -> None:
        task = self._verification_timers.pop(_REQUEST_TIMER + request_id, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _end_request_when_due(self, request: VerificationRequest) -> None:
        delay = (request.expires_at - datetime.now(UTC)).total_seconds()
        await asyncio.sleep(max(0.0, delay))
        key = _lock_key(request.identity_id, request.organization_id)
        try:
            async with self._lock_manager.locked(key):
                outcome, ended, before = await self._open_request(request.id)
        except VerificationRefusedError:
            return  # gone from the store
        await self._after_unfulfilled(outcome or "", ended, before)

    def _cancel_verification_timers(self) -> None:
        """Stop every expiry timer (the kit is closing)."""
        for task in self._verification_timers.values():
            task.cancel()
        self._verification_timers.clear()

    async def _expire_when_due(self, verification: Verification) -> None:
        if verification.expires_at is None:
            return
        delay = (verification.expires_at - datetime.now(UTC)).total_seconds()
        await asyncio.sleep(max(0.0, delay))
        await self._expire(verification)

    async def _expire(self, verification: Verification) -> Verification | None:
        """End *verification* if it is still the one in force and its time passed."""
        key = _lock_key(verification.identity_id, verification.organization_id)
        async with self._lock_manager.locked(key):
            current = await self._stored_verification(
                verification.identity_id, verification.organization_id
            )
            still_it = (
                current.status == VerificationStatus.VERIFIED
                and current.expires_at == verification.expires_at
            )
            if not still_it or _in_force(current, datetime.now(UTC)):
                return None
            completing = await self._completing_request(current)
            ended = current.model_copy(
                update={"status": VerificationStatus.EXPIRED, "ended_reason": "expired"}
            )
            await self._store.put_verification(ended)
        await self._emit_framework_event(
            "verification_ended",
            data={
                "identity_id": ended.identity_id,
                "reason": "expired",
                "ended": ENDED_VERIFICATION,
            },
        )
        if completing is not None:
            await self._verification_hook(
                HookTrigger.ON_VERIFICATION_ENDED,
                completing,
                {"reason": "expired", "ended": ENDED_VERIFICATION},
            )
        return ended

    async def _completing_request(self, current: Verification) -> VerificationRequest | None:
        """The request that completed the verification in force, if any."""
        if current.status != VerificationStatus.VERIFIED or current.completed_request_id is None:
            return None
        return await self._store.get_verification_request(current.completed_request_id)

    async def _verification_hook(
        self, trigger: HookTrigger, request: VerificationRequest, data: dict[str, Any]
    ) -> None:
        """Fire a verification hook in the room the request was opened in (ASYNC)."""
        await self._fire_lifecycle_hook(
            request.room_id,
            trigger,
            EventType.SYSTEM,
            code=str(trigger),
            message=f"Verification of {request.identity_id}",
            data={
                "identity_id": request.identity_id,
                "organization_id": request.organization_id,
                "request_id": request.id,
                "channel_id": request.channel_id,
                **data,
            },
        )

    async def _attempt(
        self, request_id: str, secret: str, method: str | None
    ) -> tuple[str, VerificationRequest, Verification]:
        """One attempt at a request, under the identity's lock: (outcome, request, state)."""
        outcome, request, current = await self._open_request(request_id)
        if outcome is not None:
            return outcome, request, current
        if not hmac.compare_digest(request.secret_hash, await _secret_hash(request.id, secret)):
            outcome, request = await self._count_failure(request, current)
            return outcome, request, current
        now = datetime.now(UTC)
        request = request.model_copy(update={"status": VerificationRequestStatus.COMPLETED})
        await self._store.update_verification_request(request)
        verification = Verification(
            identity_id=request.identity_id,
            organization_id=request.organization_id,
            status=VerificationStatus.VERIFIED,
            method=method or request.method,
            verified_via=request.channel_id,
            verified_at=now,
            expires_at=now + timedelta(seconds=self._verification_policy.ttl),
            completed_request_id=request.id,
        )
        await self._store.put_verification(verification)
        return "completed", request, verification

    async def _open_request(
        self, request_id: str
    ) -> tuple[str | None, VerificationRequest, Verification]:
        """A request read under its identity's lock, with the identity's state.

        The outcome is None when the request is open and in time; otherwise
        "not_open", or "expired" once its expiry is recorded and settled.
        """
        request = await self._store.get_verification_request(request_id)
        if request is None:
            raise VerificationRefusedError("not_found")
        current = await self._stored_verification(request.identity_id, request.organization_id)
        if request.status != VerificationRequestStatus.OPEN:
            return "not_open", request, current
        if _lapsed(request, datetime.now(UTC)):
            request = request.model_copy(update={"status": VerificationRequestStatus.EXPIRED})
            await self._store.update_verification_request(request)
            await self._settle_pending(current, request)
            return "expired", request, current
        return None, request, current

    async def _count_failure(
        self, request: VerificationRequest, current: Verification
    ) -> tuple[str, VerificationRequest]:
        """One failed attempt: "mismatch", or "failed" when it used the last one."""
        attempts = request.attempts + 1
        failed = attempts >= request.max_attempts
        status = VerificationRequestStatus.FAILED if failed else VerificationRequestStatus.OPEN
        request = request.model_copy(update={"attempts": attempts, "status": status})
        await self._store.update_verification_request(request)
        if not failed:
            return "mismatch", request
        await self._settle_pending(current, request)
        return "failed", request

    async def _after_unfulfilled(
        self, outcome: str, request: VerificationRequest, before: Verification
    ) -> None:
        """Announce a request that ended without completing (expired, failed).

        *before* is the identity's state when the attempt began: a failed
        request that was its open one ended the verification in force too
        (fail closed), and the end says so.
        """
        if outcome not in ("expired", "failed"):
            return
        ended = ENDED_REQUEST
        if outcome == "failed":
            # It may have ended a verification in force: no timer outlives it.
            self._disarm_expiry(request.identity_id, request.organization_id)
            if before.request_id == request.id and _in_force(before, datetime.now(UTC)):
                ended = ENDED_VERIFICATION
        await self._emit_verification_event(
            "verification_ended", request, reason=outcome, ended=ended
        )
        await self._verification_hook(
            HookTrigger.ON_VERIFICATION_ENDED, request, {"reason": outcome, "ended": ended}
        )

    async def _settle_pending(self, current: Verification, request: VerificationRequest) -> None:
        """Settle the identity once its request ended unfulfilled (see
        :func:`_settled`): asking a verified person again, on a suspicion, and
        getting wrong secrets must not leave the old verification standing.
        """
        if current.request_id != request.id:
            return
        failed = request.status == VerificationRequestStatus.FAILED
        await self._store.put_verification(current.model_copy(update=_settled(current, failed)))

    async def _cancel_open_request(self, current: Verification) -> VerificationRequest | None:
        """Cancel the identity's open request, if it has one; return it to announce.

        One past its expiry, whose timer did not run here, is recorded as
        expired: that is how it ended.
        """
        if current.request_id is None:
            return None
        request = await self._store.get_verification_request(current.request_id)
        if request is None or request.status != VerificationRequestStatus.OPEN:
            return None
        self._disarm_request_expiry(request.id)
        status = (
            VerificationRequestStatus.EXPIRED
            if _lapsed(request, datetime.now(UTC))
            else VerificationRequestStatus.CANCELLED
        )
        ended = request.model_copy(update={"status": status})
        await self._store.update_verification_request(ended)
        return ended

    async def _announce_superseded(self, request: VerificationRequest) -> None:
        """Announce the open request a new request or a revocation took over:
        cancelled, or expired when its time had passed (an end the hooks hear)."""
        if request.status != VerificationRequestStatus.EXPIRED:
            await self._emit_verification_event("verification_ended", request, reason="cancelled")
            return
        await self._emit_verification_event("verification_ended", request, reason="expired")
        await self._verification_hook(
            HookTrigger.ON_VERIFICATION_ENDED,
            request,
            {"reason": "expired", "ended": ENDED_REQUEST},
        )

    async def _without_lapsed_request(self, current: Verification) -> Verification:
        """*current* as it stands once its open request, past its expiry, ended.

        The timer that ends a request runs in the process that opened it: one
        that restarted, or another process, finds it still open in the store.
        """
        if current.request_id is None:
            return current
        request = await self._store.get_verification_request(current.request_id)
        if (
            request is None
            or request.status != VerificationRequestStatus.OPEN
            or not _lapsed(request, datetime.now(UTC))
        ):
            return current
        return current.model_copy(update=_settled(current, failed=False))

    async def _stored_verification(
        self, identity_id: str, organization_id: str | None
    ) -> Verification:
        stored = await self._store.get_verification(identity_id, organization_id)
        if stored is not None:
            return stored
        return Verification(identity_id=identity_id, organization_id=organization_id)

    async def _emit_verification_event(
        self,
        event_type: str,
        request: VerificationRequest,
        *,
        reason: str | None = None,
        ended: str = ENDED_REQUEST,
    ) -> None:
        data: dict[str, Any] = {
            "identity_id": request.identity_id,
            "request_id": request.id,
            "method": request.method,
            "expires_at": request.expires_at.isoformat(),
        }
        if reason is not None:
            data.update(reason=reason, ended=ended)
        await self._emit_framework_event(
            event_type, room_id=request.room_id, channel_id=request.channel_id, data=data
        )


def _code(length: int) -> str:
    """A code of *length* digits, drawn uniformly."""
    return f"{secrets.randbelow(10**length):0{length}d}"


async def _secret_hash(request_id: str, secret: str) -> str:
    """The stored form of a secret: scrypt, salted with its request's id.

    Computed off the event loop: tens of milliseconds of CPU on purpose.
    """
    digest = await asyncio.to_thread(
        hashlib.scrypt,
        secret.encode(),
        salt=request_id.encode(),
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        maxmem=64 * 1024 * 1024,
        dklen=32,
    )
    return f"scrypt${digest.hex()}"


def _in_force(verification: Verification, now: datetime) -> bool:
    return (
        verification.status == VerificationStatus.VERIFIED
        and verification.expires_at is not None
        and verification.expires_at > now
    )


def _in_scope(request: VerificationRequest, organization_id: str | None) -> bool:
    """Whether a caller acting for *organization_id* sees *request* (RFC §17.2):
    unscoped, any; scoped, its own organization's only."""
    return organization_id is None or request.organization_id == organization_id


def _lock_key(identity_id: str, organization_id: str | None) -> str:
    """One identity's transitions are serialised under this key (any lock manager)."""
    return f"verification:{organization_id or ''}:{identity_id}"


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment is not None else None


def _refusal(outcome: str, request: VerificationRequest) -> str:
    """The reason an attempt is refused: "expired" for an expired request,
    whether this attempt or its timer recorded the expiry."""
    if request.status == VerificationRequestStatus.EXPIRED:
        return "expired"
    return outcome


def _lapsed(request: VerificationRequest, now: datetime) -> bool:
    """Whether *request*'s time to be answered has passed."""
    return request.expires_at <= now


def _settled(current: Verification, failed: bool) -> dict[str, Any]:
    """The identity's update once its open request ended unfulfilled.

    An expired request proves nothing either way: the identity is left as it
    was before it. A failed one is someone who could not prove it, so a
    verification in force ends with it (fail closed).
    """
    update: dict[str, Any] = {"request_id": None}
    if failed:
        update.update(status=VerificationStatus.UNVERIFIED, ended_reason="failed")
    elif current.status == VerificationStatus.PENDING:
        update["status"] = VerificationStatus.UNVERIFIED
    return update


def _forget_timer(
    timers: dict[str, asyncio.Task[None]], key: str, done: asyncio.Task[None]
) -> None:
    """Drop a finished timer, unless a newer one took its key."""
    if timers.get(key) is done:
        del timers[key]


def _sender_identity(
    event: RoomEvent, context: RoomContext, resolved_identity: Identity | None
) -> tuple[str, str | None] | None:
    """(identity_id, organization_id) of an identified sender; None otherwise.

    The resolver's answer, or the room's IDENTIFIED participant when the
    sender was already identified and resolution was skipped (RFC §11.6).
    The scope is always the room's organization: a verification proven in
    another organization never labels this conversation (RFC §11.7.7).
    """
    organization_id = context.room.organization_id
    if resolved_identity is not None:
        return resolved_identity.id, organization_id
    identity_id = _participant_identity(event, context)
    return (identity_id, organization_id) if identity_id is not None else None


def _participant_identity(event: RoomEvent, context: RoomContext) -> str | None:
    """The identity of *event*'s sender as the room's IDENTIFIED participant
    its id names on its channel (RFC §5.5), if one.

    A sender just resolved is not one yet: the pipeline stamps the identity's
    id as the event's ``participant_id`` and records the participant later.
    """
    participant = People(context.participants).named(event)
    if participant is not None and participant.identification == IdentificationStatus.IDENTIFIED:
        return participant.identity_id
    return None


def _held(
    event: RoomEvent, context: RoomContext, identity_id: str, organization_id: str | None
) -> RoomEvent:
    """*event* held from the room's agents; the transports it reached still see it.

    Holding only narrows: the scope the event is committed with (its own
    intersected with its source binding's, RFC §7.3) keeps its transports and
    loses its intelligence channels, never gaining a channel it did not
    reach. The marker records that scope, which its release restores, and
    whose it is.
    """
    source_binding = context.get_binding(event.source.channel_id)
    scope = committed_visibility(event.visibility, source_binding, context.bindings)
    held = _transport_part(scope, context)
    marker = {
        "visibility": scope,
        "identity_id": identity_id,
        "organization_id": organization_id,
    }
    return event.model_copy(
        update={"visibility": held, "metadata": {**event.metadata, HELD_KEY: marker}}
    )


def _transport_part(visibility: str, context: RoomContext) -> str:
    """The part of *visibility* that reaches transport channels, and no more."""
    if visibility in (Visibility.ALL, Visibility.TRANSPORT):
        return Visibility.TRANSPORT
    if visibility in (Visibility.INTELLIGENCE, Visibility.NONE, Visibility.INTERNAL):
        return Visibility.NONE if visibility != Visibility.INTERNAL else Visibility.INTERNAL
    reached = [
        binding.channel_id
        for binding in context.bindings
        if binding.category == ChannelCategory.TRANSPORT and visibility_allows(visibility, binding)
    ]
    # The normal form of a list (RFC §7.3): its ids sorted.
    return ",".join(sorted(reached)) if reached else Visibility.NONE


def _held_for(event: RoomEvent, request: VerificationRequest) -> bool:
    """Whether *event* is held and belongs to *request*'s identity.

    Its marker alone is not enough: the pipeline strips one a sender supplies
    and labels what it holds, so a held event carries both, the label saying
    its sender was not verified.
    """
    marker = event.metadata.get(HELD_KEY)
    label = event.source.verification
    return (
        isinstance(marker, dict)
        and label is not None
        and label != VerificationStatus.VERIFIED
        and marker.get("identity_id") == request.identity_id
        and (marker.get("organization_id") or None) == (request.organization_id or None)
    )
